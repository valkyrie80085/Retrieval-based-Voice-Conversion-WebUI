import torch

import numpy as np

from f0_magic_new_diff import (
    compute_f0_inference,
    compute_d,
    resize,
    pitch_invert_mel,
    pitch_shift_mel,
    resize_with_zeros,
)
from f0_magic_new_diff import postprocess, preprocess, padding_size
from f0_magic_gen_diff import PitchContourGenerator, segment_size
from f0_magic_new_diff import snap
from f0_magic_new_diff import num_timesteps, sample, get_noise, sample_new

import random
import os
import json

from infer.lib.audio import pitch_blur

torch.manual_seed(42)
random.seed(42)
eps = 1e-3

with open("f0_test_config.json", "r") as openfile:
    data = json.load(openfile)
    model_path = data["model_path"]
    index_file = data["index_file"]
    audio_file = data["audio_file"]
    pitch_shift = float(data["pitch_shift"])
    try:
        invert_axis = data["invert_axis"]
    except:
        invert_axis = ""
    try:
        feature_file = data["feature_file"]
    except:
        feature_file = ""
    try:
        starter_file = data["starter_file"]
        if starter_file == "":
            starter_file = audio_file
    except:
        starter_file = audio_file
    if feature_file == "":
        feature_file = audio_file
    try:
        snap_sensitivity = float(data["snap_sensitivity"])
    except:
        snap_sensitivity = None
    if not invert_axis:
        invert_axis = None
    try:
        noise_level = int(data["noise_level"])
    except:
        noise_level = num_timesteps
if not model_path.endswith(".pt"):
    model_path += ".pt"

model = PitchContourGenerator().to("cuda")
model.load_state_dict(torch.load(model_path))
model.eval()
print(f"Model loaded from '{model_path:s}'")

input_file_p = os.path.splitext(audio_file)[0] + " p.npy"
if not os.path.isfile(input_file_p):
    np.save(input_file_p, compute_f0_inference(audio_file, index_file=index_file))
starter_file_p = os.path.splitext(starter_file)[0] + " p.npy"
if not os.path.isfile(starter_file_p):
    np.save(starter_file_p, compute_f0_inference(starter_file))

input_file_d = os.path.splitext(feature_file)[0] + " d.npy"
if not os.path.isfile(input_file_d):
    np.save(input_file_d, np.pad(compute_d(feature_file), (150, 150)))

output_file = os.path.splitext(audio_file)[0] + " out.npy"
# input_contour = np.load("input.npy")
input_contour = np.load(input_file_p)
starter_contour = resize_with_zeros(np.load(starter_file_p), len(input_contour))
starter_contour[input_contour < eps] = input_contour[input_contour < eps]
starter_contour[starter_contour < eps] = input_contour[starter_contour < eps]
input_phone_diff = resize(np.load(input_file_d), len(input_contour))
input_contour_mel = 1127 * np.log(1 + input_contour / 700)
starter_contour_mel = 1127 * np.log(1 + starter_contour / 700)
# input_contour_mel = np.round(input_contour_mel / 10) * 10
# length = len(input_contour_mel)
# input_contour_mel = resize_with_zeros(input_contour_mel, length // 3)
# input_contour_mel = resize_with_zeros(input_contour_mel, length)
if invert_axis is not None:
    input_contour_mel = pitch_invert_mel(input_contour_mel, invert_axis)
input_contour_mel = pitch_shift_mel(input_contour_mel, pitch_shift)
input_contour_mel = np.pad(input_contour_mel, (padding_size, padding_size))
starter_contour_mel = np.pad(starter_contour_mel, (padding_size, padding_size))
input_phone_diff_pad = np.pad(input_phone_diff, (padding_size, padding_size))
extra = segment_size - ((len(input_contour_mel) - 1) % segment_size + 1)
input_contour_mel = np.pad(input_contour_mel, (extra, 0))
starter_contour_mel = np.pad(starter_contour_mel, (extra, 0))
input_phone_diff_pad = np.pad(input_phone_diff_pad, (extra, 0))
input_contour_mel_tensor = torch.tensor(
    input_contour_mel, dtype=torch.float32, device="cuda"
)
starter_contour_mel_tensor = torch.tensor(
    starter_contour_mel, dtype=torch.float32, device="cuda"
)
input_phone_diff_tensor = torch.tensor(
    input_phone_diff_pad, dtype=torch.float32, device="cuda"
)
# input_contour_mel_tensor += torch.randn_like(input_contour_mel_tensor) * noise_amp

if False:
    if noise_level < num_timesteps:
        modified_contour_mel_tensor = get_noise(
            starter_contour_mel_tensor,
            torch.tensor(noise_level, device=starter_contour_mel_tensor.device).reshape(1),
            unnormalize=False,
        )
    else:
        modified_contour_mel_tensor = torch.randn_like(starter_contour_mel_tensor)
    input_contour_mel_tensor_clone = input_contour_mel_tensor.clone()
    input_contour_mel_tensor = torch.zeros_like(input_contour_mel_tensor)
    for t in reversed(range(noise_level)):
        #    last = modified_contour_mel_tensor.clone()
        t_tensor = torch.tensor(t, device=modified_contour_mel_tensor.device).reshape(1)
        input_contour_mel_tensor, modified_contour_mel_tensor = sample_new(
                model,
                modified_contour_mel_tensor.unsqueeze(0).unsqueeze(0),
                input_phone_diff_tensor.unsqueeze(0).unsqueeze(0),
                input_contour_mel_tensor.unsqueeze(0).unsqueeze(0),
                input_contour_mel_tensor_clone.unsqueeze(0).unsqueeze(0),
                t_tensor,
            )
        input_contour_mel_tensor = input_contour_mel_tensor.detach().squeeze(0).squeeze(0)
        modified_contour_mel_tensor = modified_contour_mel_tensor.detach().squeeze(0).squeeze(0)
        from torch.nn import functional as F

        #    print(t, F.mse_loss(postprocess(modified_contour_mel_tensor), postprocess(last)))
        print(
            t,
            F.mse_loss(postprocess(modified_contour_mel_tensor), input_contour_mel_tensor_clone),
        )

if snap_sensitivity is not None:
    input_contour_mel_tensor = snap(input_contour_mel_tensor, snap_sensitivity)
if noise_level < num_timesteps:
    modified_contour_mel_tensor = get_noise(
        starter_contour_mel_tensor,
        torch.tensor(noise_level, device=starter_contour_mel_tensor.device).reshape(1),
        unnormalize=False,
    )
else:
    modified_contour_mel_tensor = torch.randn_like(starter_contour_mel_tensor)
for t in reversed(range(noise_level)):
    #    last = modified_contour_mel_tensor.clone()
    t_tensor = torch.tensor(t, device=modified_contour_mel_tensor.device).reshape(1)
    modified_contour_mel_tensor = (
        sample(
            model,
            modified_contour_mel_tensor.unsqueeze(0).unsqueeze(0),
            input_phone_diff_tensor.unsqueeze(0).unsqueeze(0),
            input_contour_mel_tensor.unsqueeze(0).unsqueeze(0),
            t_tensor,
        )
        .detach()
        .squeeze(0)
        .squeeze(0)
    )
    from torch.nn import functional as F

    #    print(t, F.mse_loss(postprocess(modified_contour_mel_tensor), postprocess(last)))
    print(
        t,
        F.mse_loss(postprocess(modified_contour_mel_tensor), input_contour_mel_tensor),
    )
modified_contour_mel_tensor = postprocess(modified_contour_mel_tensor).detach()
modified_contour_mel = modified_contour_mel_tensor.detach().cpu().numpy()
modified_contour_mel = modified_contour_mel[extra:]
modified_contour_mel = modified_contour_mel[padding_size:-padding_size]
# modified_contour_mel = median_filter(modified_contour_mel, size=17)
# modified_contour_mel = modify_contour_mel(model, modified_contour_mel, threshold=threshold)
# modified_contour_mel = pitch_shift_mel(modified_contour_mel, 0)


modified_contour = (np.exp(modified_contour_mel / 1127) - 1) * 700
modified_contour[input_contour < eps] = 0
# modified_contour = pitch_blur(modified_contour, 1, 1, 1)
np.save(output_file, modified_contour)
