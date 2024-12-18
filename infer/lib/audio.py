import platform
import ffmpeg
import traceback

import librosa
import numpy as np
import av
from io import BytesIO
import traceback

import torch, torchcrepe

import os, sys

now_dir = os.getcwd()
sys.path.append(now_dir)


def wav2(i, o, format):
    inp = av.open(i, "r")
    if format == "m4a":
        format = "mp4"
    out = av.open(o, "w", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "mp4":
        format = "aac"

    ostream = out.add_stream(format)

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    for p in ostream.encode(None):
        out.mux(p)

    out.close()
    inp.close()


def load_audio(file, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = clean_path(file)  # 防止小白拷路径头尾带了空格和"和回车
        if os.path.exists(file) == False:
            raise RuntimeError(
                "You input a wrong audio path that does not exists, please fix it!"
            )
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()


def clean_path(path_str):
    if platform.system() == "Windows":
        path_str = path_str.replace("/", "\\")
    return path_str.strip(" ").strip('"').strip("\n").strip('"').strip(" ")


@torch.inference_mode()
def extract_features_simple_segment(
    audio, model, version, device, is_half=False, sr=16000
):
    if version == "mod":
        audio = torch.from_numpy(audio).unsqueeze(0).to(device)

        feats = model(audio).last_hidden_state
        if is_half:
            feats = feats.half()
        else:
            feats = feats.float()
    else:
        if sr != 16000:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=16000
            )  # , res_type="soxr_vhq"

        feats = torch.from_numpy(audio)
        if is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
            "source": (
                feats.half().to(device)
                if device not in ["mps", "cpu"]
                else feats.to(device)
            ),
            "padding_mask": padding_mask.to(device),
            "output_layer": 9 if version == "v1" else 12,  # layer 9
        }
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]
    return feats


def extract_features_simple(audio, model, version, device, is_half=False, sr=16000):
    if sr != 16000:
        audio = librosa.resample(
            audio, orig_sr=sr, target_sr=16000
        )  # , res_type="soxr_vhq"

    audio_pad = np.pad(audio, (150 * 320, 150 * 320))
    feats_segments = []
    block_size = round(60 * sr)
    max_offset = round(10 * sr)
    window_length = 160
    audio_sum = np.zeros((len(audio) - window_length,))
    for i in range(window_length):
        audio_sum += np.abs(audio[i : i - window_length])
    last_split = 0
    i = 0
    while True:
        center = (i + 1) * block_size
        if center >= len(audio) - max_offset:
            next_split = len(audio)
        else:
            next_split = np.argmin(
                audio_sum[center - max_offset - window_length : center - window_length]
            ) + round(center - max_offset - 0.5 * window_length)
            next_split = round(next_split / 320) * 320
        feats_segments.append(
            extract_features_simple_segment(
                audio_pad[last_split : min(len(audio), next_split + 160) + 300 * 320],
                model,
                version,
                device,
                is_half,
            )[:, 150:-150]
            .squeeze(0)
            .cpu()
            .numpy()
        )
        if next_split == len(audio):
            break
        else:
            last_split = next_split
            i += 1

    return np.concatenate(feats_segments, axis=0)


def extract_features_new(
    audio_original, audio_shifted, model, version, device, is_half=False
):
    feats_original = extract_features_simple_segment(
        audio_original,
        model=model,
        version=version,
        device=device,
        is_half=is_half,
        sr=16000,
    )
    if audio_shifted is None:
        return feats_original
    feats_shifted = extract_features_simple_segment(
        audio_shifted,
        model=model,
        version=version,
        device=device,
        is_half=is_half,
        sr=16000,
    )

    def get_pd(audio, target_length):
        # Pick a batch size that doesn't cause memory errors on your gpu
        torch_device_index = 0
        torch_device = None
        if torch.cuda.is_available():
            torch_device = torch.device(
                f"cuda:{torch_device_index % torch.cuda.device_count()}"
            )
        elif torch.backends.mps.is_available():
            torch_device = torch.device("mps")
        else:
            torch_device = torch.device("cpu")
        # Compute pitch using first gpu
        audio_tensor = torch.tensor(np.copy(audio_shifted))[None].float()
        f0_crepe, pd = torchcrepe.predict(
            audio_tensor,
            16000,
            160,
            50.0,
            1100.0,
            "full",
            batch_size=512,
            device=torch_device,
            return_periodicity=True,
        )
        pd = torchcrepe.filter.median(pd, 3)
        f0_crepe = torchcrepe.filter.mean(f0_crepe, 3)
        f0_crepe[pd < 0.1] = 0
        f0_crepe = f0_crepe[0].cpu().numpy()
        f0_crepe = f0_crepe[1:]  # Get rid of extra first frame
        pd = pd[0].cpu().numpy()
        pd = pd[1:]

        pd = np.interp(
            np.arange(0, len(pd) * target_len, len(pd)) / target_len,
            np.arange(0, len(pd)),
            pd,
        )
        return np.clip((pd - 0.5) / 0.2, 0, 1)

    target_len = min(feats_original.shape[1], feats_shifted.shape[1])
    feats_original, feats_shifted = (
        feats_original[:, :target_len],
        feats_shifted[:, :target_len],
    )

    pd_original = get_pd(audio_original, target_len)
    pd_shifted = get_pd(audio_shifted, target_len)
    mask = pd_original * pd_shifted + (1 - pd_original) * (1 - pd_shifted)
    mask = torch.tensor(mask, device=device).unsqueeze(-1)
    if is_half:
        mask = mask.half()
    else:
        mask = mask.float()
    return feats_shifted * mask + feats_original * (1 - mask)


def pitch_blur_mel(f0_mel, tf0=100, border=1 / 6.25, radius=1 / 12.5):
    from scipy.ndimage import gaussian_filter1d

    f0_mel_pad = np.concatenate(([0], f0_mel))
    f0_mel_segments = np.split(f0_mel_pad, np.where(f0_mel_pad < 0.001)[0])
    border_length = int(border * tf0 + 0.5)
    blurred_segments = []
    for segment in f0_mel_segments:
        if segment.shape[0] > 0:
            adjusted_border_length = min(border_length, segment.shape[0] // 5)
            if adjusted_border_length > 0:
                segment[1 : adjusted_border_length + 1] = segment[
                    adjusted_border_length + 1
                ]
                segment[-adjusted_border_length:] = segment[-adjusted_border_length - 1]
                segment[1:] = gaussian_filter1d(segment[1:], tf0 * radius)
            else:
                segment[1:] = segment[segment.shape[0] // 2]
            blurred_segments.append(segment)
    return np.concatenate(blurred_segments)[1:]


def pitch_blur(f0, tf0=100, border=1 / 6.25, radius=1 / 12.5):
    f0[np.where(f0 < 0.001)] = 0
    f0_mel = np.log(1 + f0 / 700)
    f0_mel_blurred = pitch_blur_mel(f0_mel, tf0, border, radius)
    f0_blurred = (np.exp(f0_mel_blurred) - 1) * 700
    f0_blurred[np.where(f0 < 0.001)] = 0
    return f0_blurred


def trim_sides_mel(f0_mel, tf0=100, border=0.05, threshold=75):
    from scipy.ndimage import gaussian_filter1d

    f0_mel_pad = np.concatenate(([0], f0_mel))
    f0_mel_segments = np.split(f0_mel_pad, np.where(f0_mel_pad < 0.001)[0])
    border_length = int(border * tf0 + 0.5)
    trimmed_segments = []
    for segment in f0_mel_segments:
        if segment.shape[0] > 0:
            adjusted_border_length = min(border_length, segment.shape[0] // 5)
            for i in range(adjusted_border_length):
                if abs(segment[i + 2] - segment[i + 1]) > threshold:
                    segment[1 : i + 2] = 0
                if abs(segment[-i - 1] - segment[-i - 2]) > threshold:
                    segment[-i - 1 :] = 0
            trimmed_segments.append(segment)
    return np.concatenate(trimmed_segments)[1:]


def trim_sides(f0, tf0=100, border=0.15, threshold=75):
    f0[np.where(f0 < 0.001)] = 0
    f0_mel = np.log(1 + f0 / 700)
    f0_mel_trimmed = trim_sides_mel(f0_mel, tf0, border, threshold)
    f0_trimmed = (np.exp(f0_mel_trimmed) - 1) * 700
    f0_trimmed[np.where(f0 < 0.001)] = 0
    return f0_trimmed
