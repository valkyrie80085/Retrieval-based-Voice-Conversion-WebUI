import torch

import numpy as np

from f0_magic import compute_f0_inference, compute_d, resize, pitch_invert_mel, pitch_shift_mel
from f0_magic import postprocess, preprocess_s, preprocess_t, padding_size
from f0_magic_gen import PitchContourGenerator, segment_size
from f0_magic import snap

import random
import os
import json

from infer.lib.audio import pitch_blur

torch.manual_seed(42)
random.seed(42)
eps = 1e-3

with open('f0_test_config.json', 'r') as openfile:
    data = json.load(openfile)
    model_path = data["model_path"]
    index_file = data["index_file"]
    audio_file = data["audio_file"]
    pitch_shift = float(data["pitch_shift"])
    try:
        if data["model_type"].lower().startswith("t"):
            preprocess = preprocess_t
        else:
            preprocess = preprocess_s
    except:
        preprocess = preprocess_s
    try:
        invert_axis = data["invert_axis"]
    except:
        invert_axis = ""
    try:
        feature_file = data["feature_file"]
    except:
        feature_file = ""
    if feature_file == "":
        feature_file = audio_file
    try:
        snap_sensitivity = float(data["snap_sensitivity"])
    except:
        snap_sensitivity = None
    if not invert_axis:
        invert_axis = None
if not model_path.endswith(".pt"):
    model_path += ".pt"

model = PitchContourGenerator().to("cuda")
model.eval()
model.load_state_dict(torch.load(model_path)) 
print(f"Model loaded from '{model_path:s}'")

input_file_p = os.path.splitext(audio_file)[0] + " p.npy"
if not os.path.isfile(input_file_p):
    np.save(input_file_p, compute_f0_inference(audio_file, index_file=index_file))

input_file_d = os.path.splitext(feature_file)[0] + " d.npy"
if not os.path.isfile(input_file_d):
    np.save(input_file_d, np.pad(compute_d(feature_file), (150, 150)))

output_file = os.path.splitext(audio_file)[0] + " out.npy"
#input_contour = np.load("input.npy")
input_contour = np.load(input_file_p)
input_phone_diff = resize(np.load(input_file_d), len(input_contour))
input_contour_mel = 1127 * np.log(1 + input_contour / 700) 
#input_contour_mel = np.round(input_contour_mel / 10) * 10
#length = len(input_contour_mel)
#input_contour_mel = resize_with_zeros(input_contour_mel, length // 3)
#input_contour_mel = resize_with_zeros(input_contour_mel, length)
if invert_axis is not None:
    input_contour_mel = pitch_invert_mel(input_contour_mel, invert_axis) 
modified_contour_mel = pitch_shift_mel(input_contour_mel, pitch_shift)
modified_contour_mel = np.pad(modified_contour_mel, (padding_size, padding_size))
input_phone_diff_pad = np.pad(input_phone_diff, (padding_size, padding_size))
extra = segment_size - ((len(modified_contour_mel) - 1) % segment_size + 1)
modified_contour_mel = np.pad(modified_contour_mel, (extra, 0))
input_phone_diff_pad = np.pad(input_phone_diff_pad, (extra, 0))
modified_contour_mel_tensor = torch.tensor(modified_contour_mel, dtype=torch.float32, device="cuda")
input_phone_diff_tensor = torch.tensor(input_phone_diff_pad, dtype=torch.float32, device="cuda")
#modified_contour_mel_tensor += torch.randn_like(modified_contour_mel_tensor) * noise_amp
if snap_sensitivity is not None:
    modified_contour_mel_tensor = snap(modified_contour_mel_tensor, snap_sensitivity)
modified_contour_mel_tensor = postprocess(model(preprocess(modified_contour_mel_tensor.unsqueeze(0).unsqueeze(0), input_phone_diff_tensor.unsqueeze(0).unsqueeze(0)))).squeeze(0).squeeze(0)
modified_contour_mel = modified_contour_mel_tensor.detach().cpu().numpy()
modified_contour_mel = modified_contour_mel[extra:]
modified_contour_mel = modified_contour_mel[padding_size:-padding_size]
#modified_contour_mel = median_filter(modified_contour_mel, size=17)
#modified_contour_mel = modify_contour_mel(model, modified_contour_mel, threshold=threshold)
#modified_contour_mel = pitch_shift_mel(modified_contour_mel, 0)


modified_contour = (np.exp(modified_contour_mel / 1127) - 1) * 700
modified_contour[input_contour < eps] = 0
#modified_contour = pitch_blur(modified_contour, 1, 1, 1)
np.save(output_file, modified_contour)
