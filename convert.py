import os
import numpy as np
from configs.config import Config
import soundfile as sf
import librosa
import math
import torch

from infer.lib.audio import load_audio
from infer.lib.train.mel_processing import spectrogram_torch
from infer.lib.train.utils import load_wav_to_torch
import infer.modules.vc.modules
infer.modules.vc.modules.ENC_Q = True

from dotenv import load_dotenv
load_dotenv()
load_dotenv("sha256.env")

eps = 1e-3

config = Config()
vc = infer.modules.vc.modules.VC(config)
vc.get_vc("D:/matthew99/rvc/Retrieval-based-Voice-Conversion-WebUI/assets/weights/tmp.pth")
net_g = vc.net_g

device = "cuda"
input_dir = "D:/matthew99/rvc/Retrieval-based-Voice-Conversion-WebUI/logs/mi-test/0_gt_wavs_orig"
output_dir = "D:/matthew99/rvc/Retrieval-based-Voice-Conversion-WebUI/logs/mi-test/0_gt_wavs"
pitch_dir = "D:/matthew99/rvc/Retrieval-based-Voice-Conversion-WebUI/logs/mi-test/2b-f0nsf"
os.makedirs(output_dir, exist_ok=True)

THRESHOLD = float("inf")
def shift_needed(contour_max, threshold=None):
    if threshold is None:
        threshold = THRESHOLD
    shift_needed = 0
    while contour_max * 2 ** (-shift_needed / 12) > threshold:
        shift_needed += 1
    return shift_needed

def work(input_file, pitch_file, output_file):
    global net_g
#    input_contour = np.load(input_file.replace(".wav", " out.npy"))[300:-300]
#    print(np.max(input_contour), input_file)

    with torch.no_grad():
        audio, _ = load_wav_to_torch(input_file)
        audio = audio.unsqueeze(0)
        spec = spectrogram_torch(
            audio,
            2048,
            40000,
            400,
            2048,
            center=False,
        )
        spec = spec.to(device)
        spec = torch.squeeze(spec, 0)
        spec = spec.half()
        pitchf = np.load(pitch_file)
        pitchf = pitchf[:spec.shape[-1]]
        pitchf = torch.tensor(pitchf, device=device).unsqueeze(0).float()
        len_spec = torch.tensor([spec.shape[-1]], device=device).long()
        sid = torch.tensor(0, device=device).unsqueeze(0).long()
        z, x_mask = net_g.get_hidden_features_q(
            sid, spec.unsqueeze(0), len_spec
        )
        output = (
            (
                net_g.infer_from_hidden_features(
                    sid,
                    pitchf,
                    z,
                    x_mask,
                )[0, 0]
            )
            .data.cpu()
            .float()
            .numpy()
        )

    abs_max = np.abs(output).max()
    if abs_max > 1:
        output = output / abs_max

    output = (output * np.iinfo(np.int16).max).astype(np.int16)
    sf.write(output_file, output, 40000)


for file_name in os.listdir(input_dir):
    if file_name.endswith(".wav"):
        try:
            input_file = os.path.join(input_dir, file_name)
            pitch_file = os.path.join(pitch_dir, file_name + ".npy")
            output_file = os.path.join(output_dir, file_name)
            work(input_file, pitch_file, output_file)
        except:
            pass
