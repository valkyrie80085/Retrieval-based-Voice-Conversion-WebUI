import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import numpy as np
import os

import librosa
import ffmpeg

from infer.lib.audio import load_audio, pitch_blur_mel, extract_features_simple, trim_sides_mel
import torchcrepe
import random

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.interpolate import CubicSpline

from configs.config import Config
from infer.modules.vc.utils import load_hubert
from f0_magic_gen import PitchContourGenerator, segment_size
from f0_magic_disc import PitchContourDiscriminator
from f0_magic_disc_s import PitchContourDiscriminator as PitchContourDiscriminatorS

padding_size = 2 * segment_size

config = Config()

eps = 1e-3
mel_min = 1127 * math.log(1 + 50 / 700)
mel_max = 1127 * math.log(1 + 1100 / 700)

multiplicity_target = 200
multiplicity_others = 20
max_offset = round(segment_size / 10)
min_ratio = 0.55
median_filter_size = 17
gaussian_filter_sigma = 8
preprocess_noise_amp_p = 10
preprocess_noise_amp_p_d = 10
preprocess_noise_amp_d = 0.1
data_noise_amp_p = 5
label_noise_amp_p = 0.1
d_clip_threshold = 4.5
BATCH_SIZE = 16

USE_TEST_SET = False
EPOCH_PER_BAK = 5

lr_g = 1e-5
lr_d = 1e-5
c_loss_factor = 1


def f0_magic_log(s):
    while True:
        try:
            with open("f0_magic.log", "a") as f:
                print(s, file=f)
            break
        except:
            pass
    print(s)


def gaussian_kernel1d_torch(sigma, width=None):
    if width is None:
        width = round(sigma * 4)
    distance = torch.arange(
        -width, width + 1, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    ).detach()
    gaussian = torch.exp(
        -(distance ** 2) / (2 * sigma ** 2)
    )
    gaussian /= gaussian.sum()
    kernel = gaussian[None, None]
    return kernel


def get_masks(left, right, mid, threshold=1.0, blur_mask=True):
    mid_mask = (F.pad(torch.abs(mid[:, 1:] - mid[:, :-1]), (0, 1)) <= threshold).float().detach()
    left_mask = (F.pad(torch.abs(left[:, 1:] - left[:, :-1]), (0, 1)) <= threshold).float().detach()
    left_mask[mid_mask > eps] = 0
    right_mask = (F.pad(torch.abs(right[:, 1:] - right[:, :-1]), (0, 1)) <= threshold).float().detach()
    right_mask[mid_mask > eps] = 0
    if blur_mask:
        mask_kernel = gaussian_kernel1d_torch(2)
        mask_kernel /= mask_kernel.sum()
        mid_mask = F.conv1d(mid_mask.unsqueeze(1), mask_kernel, padding="same").squeeze(1) + eps
        left_mask = F.conv1d(left_mask.unsqueeze(1), mask_kernel, padding="same").squeeze(1)
        right_mask = F.conv1d(right_mask.unsqueeze(1), mask_kernel, padding="same").squeeze(1)
    return left_mask, right_mask, mid_mask


def smooth_simple(x, size, extras, threshold=1.0, blur_mask=True):
    kernel = gaussian_kernel1d_torch(size)

    mid = F.conv1d(x.unsqueeze(1), kernel, padding="same").squeeze(1)
    mid_extras = [F.conv1d(extra.unsqueeze(1), kernel, padding="same").squeeze(1) for extra in extras]

    left_kernel = kernel.clone()
    left_kernel[:, :, :left_kernel.shape[2] // 2] = 0
    left_kernel /= left_kernel.sum()
    right_kernel = kernel.clone()
    right_kernel[:, :, right_kernel.shape[2] // 2 + 1:] = 0
    right_kernel /= right_kernel.sum()

    left = F.conv1d(x.unsqueeze(1), left_kernel, padding="same").squeeze(1)
    left_extras = [F.conv1d(extra.unsqueeze(1), left_kernel, padding="same").squeeze(1) for extra in extras]
    right = F.conv1d(x.unsqueeze(1), right_kernel, padding="same").squeeze(1)
    right_extras = [F.conv1d(extra.unsqueeze(1), right_kernel, padding="same").squeeze(1) for extra in extras]

    left_mask, right_mask, mid_mask = get_masks(left, right, mid, threshold=threshold, blur_mask=blur_mask)
    x_smoothed = (left_mask * left + right_mask * right + mid_mask * mid) / (left_mask + right_mask + mid_mask)
    extras_smoothed = [(left_mask * left_extra + right_mask * right_extra + mid_mask * mid_extra) / (left_mask + right_mask + mid_mask) for (left_extra, right_extra, mid_extra) in zip(left_extras, right_extras, mid_extras)]
    return x_smoothed, extras_smoothed


def smooth(x, threshold=1.0, blur_mask=True):
    x_ones = torch.zeros_like(x)
    x_ones[x > eps] = 1

    x_scale8, ones_scale8 = smooth_simple(x, 8, [x_ones], threshold=threshold)
#    x_scale4, ones_scale4 = smooth_simple(x, 4, [x_ones])
    x_scale4, ones_scale4 = x.clone(), [x_ones.clone()]

    ones_scale8 = ones_scale8[0]
    ones_scale4 = ones_scale4[0]

    window_size = 12
    x_max, x_min = x_scale8[:, :-window_size].clone(), x_scale8[:, :-window_size].clone()
    for i in range(window_size):
        x_max = torch.maximum(x_max, x_scale8[:, window_size - i:x_scale8.shape[1] - i])
        x_min = torch.minimum(x_min, x_scale8[:, window_size - i:x_scale8.shape[1] - i])
    x_max_hz = (torch.exp(x_max / 1127) - 1) * 700
    x_min_hz = (torch.exp(x_min / 1127) - 1) * 700
    x_diff = torch.log2(x_max_hz) - torch.log2(x_min_hz)
    threshold_diff = 1 / 25

    mask = torch.logical_or(F.pad(x_diff, (0, window_size)) <= threshold_diff, F.pad(x_diff, (window_size, 0)) <= threshold_diff).float().detach()
    if blur_mask:
        mask_kernel = gaussian_kernel1d_torch(2)
        mask_kernel /= mask_kernel.sum()
        mask = F.conv1d(mask.unsqueeze(1), mask_kernel, padding="same").squeeze(1)
    x_smoothed = x_scale8 * mask + x_scale4 * (1 - mask)
    ones_smoothed = ones_scale8 * mask + ones_scale4 * (1 - mask)

    x_ret = torch.zeros_like(x_smoothed)
    x_ret[x > eps] = x_smoothed[x > eps] / ones_smoothed[x > eps]
    return x_ret


def snap_helper(x, sensitivity):
    x_semitone = torch.log2(x / 440) * 12
    x_semitone_rounded = torch.floor(x_semitone)
    x_semitone_remainder = x_semitone - x_semitone_rounded
    x_semitone_remainder_snapped = torch.sin(torch.clip(x_semitone_remainder - 0.5, -(1 - sensitivity) / 2, (1 - sensitivity) / 2) * (math.pi / (1 - sensitivity))) / 2 + 0.5
    x_semitone_snapped = x_semitone_rounded + x_semitone_remainder_snapped
    x_snapped = torch.pow(2, x_semitone_snapped / 12) * 440
    x_snapped[x < eps] = 0
    return x_snapped


def snap(x, sensitivity):
    x = smooth(x.unsqueeze(0)).squeeze(0)
    x_hz = (torch.exp(x / 1127) - 1) * 700
    x_snapped_hz = snap_helper(x_hz, sensitivity)
    x_snapped = 1127 * torch.log(1 + x_snapped_hz / 700) 
    x_snapped[x < eps] = 0
    return x_snapped


mn_p, std_p = 550, 120
mn_d, std_d = 3.8, 1.7
std_s = 80
def preprocess_t(x, y, noise_p=None, noise_d=None):
    if noise_p is None:
        noise_p = preprocess_noise_amp_p
    if noise_d is None:
        noise_d = preprocess_noise_amp_d
    x_ret = smooth(x.squeeze(1), threshold=1.0, blur_mask=False).unsqueeze(1)
    if noise_p != 0:
        x_ret = x_ret + torch.randn_like(x_ret) * noise_p
    x_ret = (x_ret - mn_p) / std_p
    y_ret = y.clone()
    y_ret[x < eps] = 0
    y_ret = torch.clamp(y_ret, min=d_clip_threshold)
    if noise_d != 0:
        y_ret = y_ret + torch.randn_like(y_ret) * noise_d
    y_ret = (y_ret - mn_d) / std_d
    return torch.cat((x_ret, y_ret), dim=1)


def preprocess_s(x, y):
    x_ret = (x - mn_p) / std_p
    y_ret = (y - mn_d) / std_d
    return torch.cat((x_ret, y_ret), dim=1)


def preprocess(x, y):
    return preprocess_s(x, y)


def preprocess_disc_t(x, y, noise_p=None, noise_d=None):
    if noise_p is None:
        noise_p = preprocess_noise_amp_p_d
    if noise_d is None:
        noise_d = preprocess_noise_amp_d
    kernel = gaussian_kernel1d_torch(gaussian_filter_sigma)
    x_blurred = F.conv1d(x, kernel, padding="same")
    x_blurred[x < eps] = 0
    x_sharpened = x - x_blurred
    if noise_p != 0:
        x_blurred = x_blurred + torch.randn_like(x_blurred) * noise_p
        x_blurred[x < eps] = 0
    x_blurred = (x_blurred - mn_p) / std_p
    x_sharpened = x_sharpened / std_s
    y_ret = y.clone()
    y_ret[x < eps] = 0
    y_ret = torch.clamp(y_ret, min=d_clip_threshold)
    if noise_d != 0:
        y_ret = y_ret + torch.randn_like(y_ret) * noise_d
#    if random.randint(0, 1) == 0:
#        y_ret = torch.zeros_like(y_ret)
    y_ret = (y_ret - mn_d) / std_d
    return torch.cat((x_blurred, x_sharpened, y_ret), dim=1)


def preprocess_disc_s(x, y, z, noise_p=None):
    kernel = gaussian_kernel1d_torch(gaussian_filter_sigma)
    if noise_p is None:
        noise_p = preprocess_noise_amp_p_d
    x_blurred = F.conv1d(x, kernel, padding="same")
    x_blurred[x < eps] = 0
    x_sharpened = x - x_blurred
    if noise_p != 0:
        x_blurred = x_blurred + torch.randn_like(x_blurred) * noise_p
        x_blurred[x < eps] = 0
    x_blurred = (x_blurred - mn_p) / std_p
    x_sharpened = x_sharpened / std_s
    y_ret = (y - mn_d) / std_d
    z_ret = (z - mn_p) / std_p
    return torch.cat((x_blurred, x_sharpened, y, z), dim=1)


def postprocess(x):
    x_ret = x.clone()
    x_ret = x * std_p + mn_p
    x_ret[x_ret < mel_min * 0.5] = 0
#    x_ret[x < eps] = (2 * mel_min - mel_max - mn_p) / std_p
    return x_ret


sr = 16000
window_length = 160
frames_per_sec = sr // window_length
def resize_with_zeros(contour, target_len):
    a = contour.copy()
    a[a < eps] = np.nan
    a = np.interp(
        np.arange(0, len(a) * target_len, len(a)) / target_len,
        np.arange(0, len(a)),
        a
    )
    a = np.nan_to_num(a)
    return a


def resize(a, target_len):
    return np.interp(
        np.arange(0, len(a) * target_len, len(a)) / target_len,
        np.arange(0, len(a)),
        a
    )


hubert_model = None
def trim_f0(f0, audio, index_file, version="v2"):
    global hubert_model

    if not os.path.isfile(index_file):
        return f0
    import faiss
    try:
        index = faiss.read_index(index_file)
        # big_npy = np.load(file_big_npy)
        big_npy = index.reconstruct_n(0, index.ntotal)
    except:
        print("Failed to read index file: \"{index_file:s}\"")
        return f0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hubert_model is None:
        hubert_model = load_hubert(config)

    feats = extract_features_simple(audio, model=hubert_model, version=version, device=device, is_half=config.is_half)
    npy = feats[0].cpu().numpy()
    npy = np.concatenate((npy, np.full((npy.shape[0], 1), 0.5)), axis=1)

    score, ix = index.search(npy, k=8)
    weight = np.square(1 / score)
    weight /= weight.sum(axis=1, keepdims=True)
    npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

    pd = npy[:, -1]
    pd = np.interp(
        np.arange(0, len(pd) * len(f0), len(pd)) / len(f0),
        np.arange(0, len(pd)),
        pd
    )

    threshold = 0.5
    for it in (range(len(f0)), reversed(range(len(f0)))):
        keep = False
        for i in it:
            if f0[i] > eps:
                if pd[i] > threshold:
                    keep = True
                if not keep:
                    f0[i] = 0
            else:
                keep = False

    return f0


model_rmvpe = None
def compute_f0_inference(path, index_file=""):
    print("computing f0 for: " + path)
    x = load_audio(path, 44100)
    x = librosa.resample(
        x, orig_sr=44100, target_sr=sr
    )

    global model_rmvpe
    if model_rmvpe is None:
        from infer.lib.rmvpe import RMVPE
        print("Loading rmvpe model")
        model_rmvpe = RMVPE(
            "assets/rmvpe/rmvpe.pt", is_half=False, device="cuda")
    f0 = model_rmvpe.infer_from_audio(x, thred=0.03)

    # Pick a batch size that doesn't cause memory errors on your gpu
    torch_device_index = 0
    torch_device = None
    if torch.cuda.is_available():
        torch_device = torch.device(f"cuda:{torch_device_index % torch.cuda.device_count()}")
    elif torch.backends.mps.is_available():
        torch_device = torch.device("mps")
    else:
        torch_device = torch.device("cpu")
    model = "full"
    batch_size = 512
    # Compute pitch using first gpu
    audio_tensor = torch.tensor(np.copy(x))[None].float()
    f0_crepe, pd = torchcrepe.predict(
        audio_tensor,
        16000,
        160,
        50,
        1100,
        model,
        batch_size=batch_size,
        device=torch_device,
        return_periodicity=True,
    )
    pd = torchcrepe.filter.median(pd, 3)
    f0_crepe = torchcrepe.filter.mean(f0_crepe, 3)
    f0_crepe[pd < 0.1] = 0
    f0_crepe = f0_crepe[0].cpu().numpy()
    f0_crepe = f0_crepe[1:] # Get rid of extra first frame

    # Resize the pitch
    target_len = f0.shape[0]
    f0_crepe = resize_with_zeros(f0_crepe, target_len)

    f0_rmvpe_mel = np.log(1 + f0 / 700)
    f0_crepe_mel = np.log(1 + f0_crepe / 700)
    f0 = np.where(np.logical_and(f0_rmvpe_mel > eps, f0_crepe_mel - f0_rmvpe_mel > 0.05), f0_crepe, f0)

    f0_mel = 1127 * np.log(1 + f0 / 700)

    target_len = x.shape[0] // window_length
    f0_mel = resize_with_zeros(f0_mel, target_len)

    model_rmvpe = None
    f0_mel = trim_f0(f0_mel, x, index_file)

    f0_mel = trim_sides_mel(f0_mel, frames_per_sec)

    f0 = (np.exp(f0_mel / 1127) - 1) * 700 
    f0 = np.pad(f0, (300, 300))
    return f0


def compute_f0(path):
    print("computing f0 for: " + path)
    x = load_audio(path, 44100)
    x = librosa.resample(
        x, orig_sr=44100, target_sr=sr
    )

    global model_rmvpe
    if model_rmvpe is None:
        from infer.lib.rmvpe import RMVPE
        print("Loading rmvpe model")
        model_rmvpe = RMVPE(
            "assets/rmvpe/rmvpe.pt", is_half=False, device="cuda")
    f0 = model_rmvpe.infer_from_audio(x, thred=0.03)

    # Pick a batch size that doesn't cause memory errors on your gpu
    torch_device_index = 0
    torch_device = None
    if torch.cuda.is_available():
        torch_device = torch.device(f"cuda:{torch_device_index % torch.cuda.device_count()}")
    elif torch.backends.mps.is_available():
        torch_device = torch.device("mps")
    else:
        torch_device = torch.device("cpu")
    model = "full"
    batch_size = 512
    # Compute pitch using first gpu
    audio_tensor = torch.tensor(np.copy(x))[None].float()
    f0_crepe, pd = torchcrepe.predict(
        audio_tensor,
        16000,
        160,
        50,
        1100,
        model,
        batch_size=batch_size,
        device=torch_device,
        return_periodicity=True,
    )
    pd = torchcrepe.filter.median(pd, 3)
    f0_crepe = torchcrepe.filter.mean(f0_crepe, 3)
    f0_crepe[pd < 0.1] = 0
    f0_crepe = f0_crepe[0].cpu().numpy()
    f0_crepe = f0_crepe[1:] # Get rid of extra first frame

    # Resize the pitch
    target_len = f0.shape[0]
    f0_crepe = resize_with_zeros(f0_crepe, target_len)

    f0_rmvpe_mel = np.log(1 + f0 / 700)
    f0_crepe_mel = np.log(1 + f0_crepe / 700)
    f0 = np.where(np.logical_and(f0_rmvpe_mel > eps, f0_crepe_mel - f0_rmvpe_mel > 0.05), f0_crepe, f0)

    f0_mel = 1127 * np.log(1 + f0 / 700)

    target_len = x.shape[0] // window_length
    f0_mel = resize_with_zeros(f0_mel, target_len)
    return f0_mel



TARGET_PATH = "C:/datasets/singing_ai/f0_magic/target"
OTHERS_PATH = "C:/datasets/singing_ai/f0_magic/others"

def walk(path):
   return sum(([os.path.join(dirpath, file_name) for file_name in filenames] for (dirpath, dirnames, filenames) in os.walk(path)), [])


def compute_d(path):
    print("computing phone difference for: " + path)
    global hubert_model
    config = Config()
    if hubert_model is None:
        hubert_model = load_hubert(config)

    audio = load_audio(path, 44100)
    audio = librosa.resample(
        audio, orig_sr=44100, target_sr=16000
    )

    feats = extract_features_simple(audio, model=hubert_model, version="v2", device="cuda", is_half=config.is_half)
    npy = feats[0].cpu().numpy()

    feats_diff = np.pad(np.linalg.norm(npy[:-1] - npy[1:], axis=1), (1, 0))
    return feats_diff


def prepare_data():
    filenames = []
    for filename in walk(TARGET_PATH):
        if filename.endswith(".wav"): 
            filenames.append(filename)
    for filename in walk(OTHERS_PATH):
        if filename.endswith(".wav"):
            filenames.append(filename)
    for filename in filenames:
        npy_file_p = os.path.splitext(filename)[0] + " p.npy"
        npy_file_d = os.path.splitext(filename)[0] + " d.npy"
        if not os.path.isfile(npy_file_p):
            try:
                np.save(npy_file_p, compute_f0(filename))
            except:
                pass
        if not os.path.isfile(npy_file_d):
            try:
                np.save(npy_file_d, compute_d(filename))
            except:
                pass


def pitch_shift_mel(contour, semitones):
    contour_shifted = (np.exp(contour / 1127) - 1)
    contour_shifted *= 2 ** (semitones / 12)
    contour_shifted = 1127 * np.log(1 + contour_shifted)
    contour_shifted[contour < eps] = 0
    return contour_shifted


def pitch_shift_tensor(contour, semitones):
    contour_shifted = (torch.exp(contour / 1127) - 1)
    contour_shifted *= 2 ** (semitones / 12)
    contour_shifted = 1127 * torch.log(1 + contour_shifted)
    contour_shifted[contour < eps] = 0
    return contour_shifted


def pitch_invert_mel(contour, note):
    contour_inverted = (np.exp(contour / 1127) - 1) * 700
    contour_inverted[contour_inverted > 0] = (librosa.note_to_hz(note) ** 2) / contour_inverted[contour_inverted > 0]
    contour_inverted = 1127 * np.log(1 + contour_inverted / 700)
    contour_inverted[contour < eps] = 0
    return contour_inverted


def add_noise(contour, amp=5, scale=1):
    zeros = contour < eps
    length = int(contour.shape[0] / scale) + 1
    noise = np.random.normal(0, amp, length)
    if len(noise) != len(contour):
        noise = CubicSpline(np.arange(0, len(noise)), noise)(np.arange(0, len(noise) * len(contour), len(noise)) / len(contour))
    contour_with_noise = contour + noise
    contour_with_noise[zeros] = 0
    return contour_with_noise


def get_average(contour):
    try:
        return np.average(contour[contour > eps])
    except ZeroDivisionError:
        return 0


def change_vibrato(contour, factor):
    blurred = pitch_blur_mel(contour, frames_per_sec)
    modified_contour = blurred + factor * (contour - blurred)
    modified_contour[modified_contour < eps] = 0
    return modified_contour


def modify_ends(contour):
    from scipy.ndimage import gaussian_filter1d
    contour_pad = np.concatenate(([0], contour))
    contour_segments = np.split(contour_pad, np.where(contour_pad < eps)[0])
    border_length = random.randint(4, 24)
    amount = random.uniform(30, 60) * random.choice((-1, 1))
    t = random.randint(0, 1)
    mask = np.hanning(border_length * 2)
    if t == 0:
        mask = mask[border_length:]
    else:
        mask = mask[:border_length]
    mask *= amount
    modified_segments = []
    for segment in contour_segments:
        if segment.shape[0] > 0:
            if len(segment) > border_length:
                if t == 0:
                    segment[1:border_length + 1] += mask
                else:
                    segment[-border_length:] += mask
            modified_segments.append(segment)
    modified_contour = np.concatenate(modified_segments)[1:]
    return modified_contour


def load_data():
    prepare_data()
    train_target_data = []
    train_others_data = []
    test_target_data = []
    test_others_data = []
    test_set = set()
    for filename in walk(TARGET_PATH) + walk(OTHERS_PATH):
        if filename.endswith(".wav"): 
            if random.uniform(0, 1) < 0.2:
                test_set.add(filename)
    target_averages = []
    for filename in walk(TARGET_PATH):
        if filename.endswith(".wav"): 
            if filename in test_set:
                target_data, others_data = test_target_data, test_others_data
            else:
                target_data, others_data = train_target_data, train_others_data
            filename_p = os.path.splitext(filename)[0] + " p.npy"
            filename_d = os.path.splitext(filename)[0] + " d.npy"
            contour = np.load(filename_p)
            phone_diff = resize(np.load(filename_d), len(contour))
            if contour.shape[0] < segment_size:
                phone_diff = np.pad(phone_diff, (0, segment_size - contour.shape[0]))
                contour = np.pad(contour, (0, segment_size - contour.shape[0]))
            contour = np.pad(contour, (padding_size + max_offset, padding_size))
            phone_diff = np.pad(phone_diff, (padding_size + max_offset, padding_size))
            for i in range(round(multiplicity_target * (contour.shape[0] - max_offset - 2 * padding_size) / segment_size) + 1):
                start = random.randint(padding_size + max_offset, contour.shape[0] - padding_size - segment_size)
                contour_sliced = contour[start - padding_size - max_offset:start + padding_size + segment_size].copy()
                phone_diff_sliced = phone_diff[start - padding_size - max_offset:start + padding_size + segment_size].copy()
                if np.sum(contour_sliced[padding_size:-padding_size] > eps) > segment_size * min_ratio:
                    average = get_average(contour_sliced[contour_sliced > eps]) / 1127
                    target_averages.append(average)
                    contour_final = contour_sliced
                    phone_diff_final = phone_diff_sliced
                    target_data.append((torch.tensor(contour_final, dtype=torch.float32), torch.tensor(phone_diff_final, dtype=torch.float32)))

    if multiplicity_others > 0:
        for filename in walk(OTHERS_PATH):
            if filename.endswith(".wav"):
                if filename in test_set:
                    target_data, others_data = test_target_data, test_others_data
                else:
                    target_data, others_data = train_target_data, train_others_data
                filename_p = os.path.splitext(filename)[0] + " p.npy"
                filename_d = os.path.splitext(filename)[0] + " d.npy"
                contour = np.load(filename_p)
                phone_diff = resize(np.load(filename_d), len(contour))
                if contour.shape[0] < segment_size:
                    phone_diff = np.pad(phone_diff, (0, segment_size - contour.shape[0]))
                    contour = np.pad(contour, (0, segment_size - contour.shape[0]))
                contour = np.pad(contour, (padding_size + max_offset, padding_size))
                phone_diff = np.pad(phone_diff, (padding_size + max_offset, padding_size))
                for i in range(round(multiplicity_others * (contour.shape[0] - max_offset - 2 * padding_size) / segment_size)):
                    start = random.randint(padding_size + max_offset, contour.shape[0] - padding_size - segment_size)
                    use_original = False#random.randint(0, 4) == 0
                    contour_sliced = contour[start - padding_size - max_offset:start + padding_size + segment_size].copy()
                    phone_diff_sliced = phone_diff[start - padding_size - max_offset:start + padding_size + segment_size].copy()
                    if np.sum(contour_sliced[padding_size:-padding_size] > eps) > segment_size * min_ratio:
                        if use_original:
                            shift_real = 0
                        else:
                            average = get_average(contour_sliced[contour_sliced > eps]) / 1127
                            average_goal = random.choice(target_averages)
                            average = (math.exp(average) - 1) * 700
                            average_goal = (math.exp(average_goal) - 1) * 700
                            shift_real = math.log(average_goal / average) / math.log(2) * 12
                        contour_final = pitch_shift_mel(contour_sliced, shift_real)
                        phone_diff_final = phone_diff_sliced
                        others_data.append((torch.tensor(contour_final, dtype=torch.float32), torch.tensor(phone_diff_final, dtype=torch.float32)))

    print("Train target data count:", len(train_target_data))
    print("Train others data count:", len(train_others_data))
    print("Test target data count:", len(test_target_data))
    print("Test others data count:", len(test_others_data))
    return train_target_data, train_others_data, test_target_data, test_others_data


def median_filter1d_torch(x, size):
    return torch.median(torch.cat(tuple(x[:, i:x.shape[1] - size + i + 1].unsqueeze(2) for i in range(size)), dim=2), dim=2).values


def contrastive_loss(output, ref, size):
    ref_scale8, output_scale8 = smooth_simple(ref, 8, [output])
    ref_scale4, output_scale4 = smooth_simple(ref, 4, [output])

    output_scale8 = output_scale8[0]
    output_scale4 = output_scale4[0]

    mask = (F.pad(torch.abs(ref_scale8[:, 1:] - ref_scale8[:, :-1]), (0, 1)) <= 1.0).float().detach()
    mask_kernel = gaussian_kernel1d_torch(2)
    mask_kernel /= mask_kernel.sum()
    mask = F.conv1d(mask.unsqueeze(1), mask_kernel, padding="same").squeeze(1)
    ref_smoothed = ref_scale8 * mask + ref_scale4 * (1 - mask)
    output_smoothed = output_scale8 * mask + output_scale4 * (1 - mask)

    output_smoothed = output_smoothed[:, padding_size:-padding_size]
    ref_smoothed = ref_smoothed[:, padding_size:-padding_size]

    return F.mse_loss(output_smoothed, ref_smoothed)


def train_model(name, train_target_data, train_others_data, test_target_data, test_others_data):
    if train_target_data:
        train_target_data_p = torch.stack(tuple(pitch for pitch, phone_diff in train_target_data))
        train_target_data_d = torch.stack(tuple(phone_diff for pitch, phone_diff in train_target_data))
    if train_others_data:
        train_others_data_p = torch.stack(tuple(pitch for pitch, phone_diff in train_others_data))
        train_others_data_d = torch.stack(tuple(phone_diff for pitch, phone_diff in train_others_data))
    if test_target_data:
        test_target_data_p = torch.stack(tuple(pitch for pitch, phone_diff in test_target_data))
        test_target_data_d = torch.stack(tuple(phone_diff for pitch, phone_diff in test_target_data))
    if test_others_data:
        test_others_data_p = torch.stack(tuple(pitch for pitch, phone_diff in test_others_data))
        test_others_data_d = torch.stack(tuple(phone_diff for pitch, phone_diff in test_others_data))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_g_t = PitchContourGenerator().to(device)
    net_d_t = PitchContourDiscriminator().to(device)
    net_g_s = PitchContourGenerator().to(device)
    net_d_s = PitchContourDiscriminatorS().to(device)
    optimizer_g_t = optim.AdamW(net_g_t.parameters(), lr=lr_g)
    optimizer_d_t = optim.AdamW(net_d_t.parameters(), lr=lr_d)
    optimizer_g_s = optim.AdamW(net_g_t.parameters(), lr=lr_g)
    optimizer_d_s = optim.AdamW(net_d_t.parameters(), lr=lr_d)
    epoch = 0

    MODEL_FILE = name + ".pt"
    CHECKPOINT_FILE = name + " checkpoint.pt"
    TMP_FILE = name + "_tmp.pt"
    if os.path.isfile(TMP_FILE):
        if not os.path.isfile(CHECKPOINT_FILE):
            os.rename(TMP_FILE, CHECKPOINT_FILE)
        else:
            os.remove(TMP_FILE)
    try:
        if os.path.isfile(CHECKPOINT_FILE):
            checkpoint = torch.load(CHECKPOINT_FILE)
            epoch = checkpoint['epoch']
            net_g_t.load_state_dict(checkpoint['net_g_t'])
            net_d_t.load_state_dict(checkpoint['net_d_t'])
            net_g_s.load_state_dict(checkpoint['net_g_s'])
            net_d_s.load_state_dict(checkpoint['net_d_s'])
            optimizer_g_t.load_state_dict(checkpoint['optimizer_g_t'])
            optimizer_d_t.load_state_dict(checkpoint['optimizer_d_t'])
            optimizer_g_s.load_state_dict(checkpoint['optimizer_g_s'])
            optimizer_d_s.load_state_dict(checkpoint['optimizer_d_s'])
            print(f"Data loaded from '{CHECKPOINT_FILE:s}'")
        else:
            print("Model initialized with random weights")
    except:
        epoch = 0
        net_g_t = PitchContourGenerator().to(device)
        net_d_t = PitchContourDiscriminator().to(device)
        net_g_s = PitchContourGenerator().to(device)
        net_d_s = PitchContourDiscriminatorS().to(device)
        optimizer_g_t = optim.AdamW(net_g_t.parameters(), lr=lr_g)
        optimizer_d_t = optim.AdamW(net_d_t.parameters(), lr=lr_d)
        optimizer_g_s = optim.AdamW(net_g_t.parameters(), lr=lr_g)
        optimizer_d_s = optim.AdamW(net_d_t.parameters(), lr=lr_d)
        print("Model initialized with random weights")

    train_dataset = torch.utils.data.TensorDataset(train_target_data_p, train_target_data_d, torch.ones((len(train_target_data),)))
    if train_others_data:
        train_dataset += torch.utils.data.TensorDataset(train_others_data_p, train_others_data_d, torch.zeros((len(train_others_data),)))
    if USE_TEST_SET:
        test_dataset = torch.utils.data.TensorDataset(test_target_data_p, test_target_data_d, torch.ones((len(test_target_data),)))
        if test_others_data:
            test_dataset += torch.utils.data.TensorDataset(test_others_data_p, test_others_data_d, torch.zeros((len(test_others_data),)))
    else:
        if test_target_data:
            train_dataset += torch.utils.data.TensorDataset(test_target_data_p, test_target_data_d, torch.ones((len(test_target_data),)))
        if test_others_data:
            train_dataset += torch.utils.data.TensorDataset(test_others_data_p, test_others_data_d, torch.zeros((len(test_others_data),)))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    if USE_TEST_SET:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.BCELoss()

    best_loss = float("inf")

    while True:
        epoch += 1

        def work(data_p, data_d, labels, is_train, disc_loss, contrastive_loss_t, gen_loss, contrastive_loss_s, imitation_loss):
            if is_train:
                net_d_t.train()
                net_d_s.train()
            else:
                net_d_t.eval()
                net_d_s.eval()
            data_p, data_d, labels = data_p.to(device), data_d.to(device), labels.to(device)
            offset = torch.randint(0, max_offset, (1,))
            data_p = data_p[:, offset:data_p.shape[1] - max_offset + offset]
            data_d = data_d[:, offset:data_d.shape[1] - max_offset + offset]
            data_p = pitch_shift_tensor(data_p, torch.rand(1, device=data_p.device) - 0.5)

            fakes = postprocess(net_g_t(preprocess_t(data_p.unsqueeze(1), data_d.unsqueeze(1), noise_p=preprocess_noise_amp_p, noise_d=preprocess_noise_amp_d))).squeeze(1)
            fakes_s = postprocess(net_g_s(preprocess_s(data_p.unsqueeze(1), data_d.unsqueeze(1)))).squeeze(1)
            d_data_p = fakes.detach().clone()
            d_data_d = data_d.detach().clone()
            d_labels = torch.zeros((d_data_p.shape[0],), device=device)
            if torch.sum(labels > eps) > 0:
                target_data_p = data_p[labels > eps]
                target_labels = torch.ones((target_data_p.shape[0],), device=device)
                d_data_p = torch.cat((d_data_p, target_data_p), dim=0)
                d_data_d = torch.cat((d_data_d, data_d[labels > eps]), dim=0)
                d_labels = torch.cat((d_labels, target_labels), dim=0)

            outputs = net_d_t(preprocess_disc_t(d_data_p.unsqueeze(1), d_data_d.unsqueeze(1), noise_p=preprocess_noise_amp_p_d, noise_d=preprocess_noise_amp_d))
            loss = criterion(outputs, d_labels.unsqueeze(1).expand(-1, outputs.shape[1]))
            disc_loss.append(loss.item())

            if is_train:
                optimizer_d_t.zero_grad()
                loss.backward()
                optimizer_d_t.step()

                d_s_data_in = torch.cat((data_p, data_p), dim=0)
                d_s_data_d = torch.cat((data_d, data_d), dim=0)
                d_s_data_out = torch.cat((fakes.detach(), fakes_s.detach()), dim=0)
                d_s_labels = torch.cat((torch.ones((fakes.shape[0],), device=device), torch.zeros((fakes_s.shape[0],), device=device)), dim=0)

                outputs = net_d_s(preprocess_disc_s(d_s_data_in.unsqueeze(1), d_s_data_d.unsqueeze(1), d_s_data_out.unsqueeze(1), noise_p=preprocess_noise_amp_p_d))
                loss = criterion(outputs, d_s_labels.unsqueeze(1).expand(-1, outputs.shape[1]))

                optimizer_d_s.zero_grad()
                loss.backward()
                optimizer_d_s.step()

            net_d_t.eval()
            net_d_s.eval()

            g_data_p = fakes
            g_labels = torch.ones((g_data_p.shape[0],), device=device)
            outputs = net_d_t(preprocess_disc_t(g_data_p.unsqueeze(1), data_d.unsqueeze(1), noise_d=preprocess_noise_amp_d))

            loss_total = 0
            loss = criterion(outputs, g_labels.unsqueeze(1).expand(-1, outputs.shape[1]))
            loss_total = loss
            gen_loss.append(loss.item())

            loss = contrastive_loss(fakes, data_p, gaussian_filter_sigma)
            loss_total += loss * c_loss_factor
            contrastive_loss_t.append(loss.item())

            if is_train:
                optimizer_g_t.zero_grad()
                loss_total.backward()
                optimizer_g_t.step()

            g_s_data_p = fakes_s
            g_s_labels = torch.ones((g_s_data_p.shape[0],), device=device)
            outputs = net_d_s(preprocess_disc_s(data_p.unsqueeze(1), data_d.unsqueeze(1), g_s_data_p.unsqueeze(1)))

            loss_total = 0
            loss = criterion(outputs, g_s_labels.unsqueeze(1).expand(-1, outputs.shape[1]))
            loss_total = loss
            imitation_loss.append(loss.item())

            loss = contrastive_loss(fakes_s, data_p, gaussian_filter_sigma)
            loss_total += loss * c_loss_factor
            contrastive_loss_s.append(loss.item())

            if is_train:
                optimizer_g_s.zero_grad()
                loss_total.backward()
                optimizer_g_s.step()


        train_disc_loss = []
        train_contrastive_loss_t = []
        train_gen_loss = []
        train_contrastive_loss_s = []
        train_imitation_loss = []
        for batch_idx, (data_p, data_d, labels) in enumerate(train_loader):
            if batch_idx % 10 == 0:
                print(f"train {batch_idx}/{len(train_loader)}")
            work(data_p, data_d, labels, True, train_disc_loss, train_contrastive_loss_t, train_gen_loss, train_contrastive_loss_s, train_imitation_loss)
            
        train_disc_loss = np.mean(train_disc_loss)
        train_contrastive_loss_t = np.mean(train_contrastive_loss_t)
        train_gen_loss = np.mean(train_gen_loss)
        train_contrastive_loss_s = np.mean(train_contrastive_loss_s)
        train_imitation_loss = np.mean(train_imitation_loss)
        train_loss = (train_contrastive_loss_t + train_contrastive_loss_s) * c_loss_factor + train_gen_loss + train_imitation_loss

        if USE_TEST_SET:
            test_disc_loss = []
            test_contrastive_loss_t = []
            test_gen_loss = []
            test_contrastive_loss_s = []
            test_imitation_loss = []
            for batch_idx, (data_p, data_d, labels) in enumerate(test_loader):
                if batch_idx % 10 == 0:
                    print(f"val {batch_idx}/{len(test_loader)}")
                work(data_p, data_d, labels, False, test_disc_loss, test_contrastive_loss_t, test_gen_loss, test_contrastive_loss_s, test_imitation_loss)
                
            test_disc_loss = np.mean(test_disc_loss)
            test_contrastive_loss_t = np.mean(test_contrastive_loss_t)
            test_gen_loss = np.mean(test_gen_loss)
            test_contrastive_loss_s = np.mean(test_contrastive_loss_s)
            test_imitation_loss = np.mean(test_imitation_loss)
            test_loss = (test_contrastive_loss_t + test_contrastive_loss_s) * c_loss_factor + test_gen_loss + test_imitation_loss


        if epoch % 1 == 0:
            f0_magic_log(f"Epoch: {epoch:d}")
            f0_magic_log(f"t_loss: {train_loss:.4f} t_loss_c_t: {train_contrastive_loss_t:.4f} t_loss_g: {train_gen_loss:.4f} t_loss_d: {train_disc_loss:.4f} t_loss_c_s: {train_contrastive_loss_t:.4f} t_loss_i:{train_imitation_loss:.4f}")
            if USE_TEST_SET:
                f0_magic_log(f"v_loss: {test_loss:.4f} v_loss_c_t: {test_contrastive_loss_t:.4f} v_loss_g: {test_gen_loss:.4f} v_loss_d: {test_disc_loss:.4f} v_loss_i:{test_imitation_loss:.4f}")
            checkpoint = { 
                'epoch': epoch,
                'net_g_t': net_g_t.state_dict(),
                'net_d_t': net_d_t.state_dict(),
                'net_g_s': net_g_s.state_dict(),
                'net_d_s': net_d_s.state_dict(),
                'optimizer_g_t': optimizer_g_t.state_dict(),
                'optimizer_d_t': optimizer_d_t.state_dict(),
                'optimizer_g_s': optimizer_g_s.state_dict(),
                'optimizer_d_s': optimizer_d_s.state_dict()}
            while True:
                try:
                    torch.save(net_g_s.state_dict(), MODEL_FILE) 
                    break
                except:
                    pass
            torch.save(checkpoint, TMP_FILE)
            if os.path.isfile(CHECKPOINT_FILE):
                while True:
                    try:
                        os.remove(CHECKPOINT_FILE)
                        break
                    except:
                        pass
            while True:
                try:
                    os.rename(TMP_FILE, CHECKPOINT_FILE)
                    break
                except:
                    pass
            try:
                #            np.save(FAKE_DATA_FILE, fakes)
                pass
            except:
                pass
            print(f"Data saved.")
        if True:#(USE_TEST_SET and test_loss < best_loss) or ((not USE_TEST_SET) and train_loss < best_loss):
            #            best_loss = test_loss if USE_TEST_SET else train_loss 
            BAK_FILE = name + " " + str(epoch // EPOCH_PER_BAK * EPOCH_PER_BAK) + ".pt"
            BAK_FILE_T = name + " t " + str(epoch // EPOCH_PER_BAK * EPOCH_PER_BAK) + ".pt"
            while True:
                try:
                    torch.save(net_g_s.state_dict(), BAK_FILE) 
                    torch.save(net_g_t.state_dict(), BAK_FILE_T) 
                    break
                except:
                    pass
            print(f"Model backed up to '{BAK_FILE:s}'")


if __name__ == "__main__":
    random.seed(42)

    train_target_data, train_others_data, test_target_data, test_others_data = load_data()
    train_model("model", train_target_data, train_others_data, test_target_data, test_others_data)
