import torch

import numpy as np
from scipy.ndimage import median_filter

from f0_magic import compute_f0_inference, pitch_invert_mel, pitch_shift_mel#, noise_amp
from f0_magic import postprocess, preprocess, padding_size
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
        invert_axis = data["invert_axis"]
    except:
        invert_axis = ""
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

input_file = os.path.splitext(audio_file)[0] + ".npy"
if not os.path.isfile(input_file):
    np.save(input_file, compute_f0_inference(audio_file, index_file=index_file))

output_file = os.path.splitext(input_file)[0] + " out.npy"
#input_contour = np.load("input.npy")
input_contour = np.load(input_file)
input_contour_mel = 1127 * np.log(1 + input_contour / 700) 
#input_contour_mel = np.round(input_contour_mel / 10) * 10
#length = len(input_contour_mel)
#input_contour_mel = resize_with_zeros(input_contour_mel, length // 3)
#input_contour_mel = resize_with_zeros(input_contour_mel, length)
if invert_axis is not None:
    input_contour_mel = pitch_invert_mel(input_contour_mel, invert_axis) 
modified_contour_mel = pitch_shift_mel(input_contour_mel, pitch_shift)
modified_contour_mel = np.pad(modified_contour_mel, (padding_size, padding_size))
extra = segment_size - ((len(modified_contour_mel) - 1) % segment_size + 1)
modified_contour_mel = np.pad(modified_contour_mel, (extra, 0))
modified_contour_mel_tensor = torch.tensor(modified_contour_mel, dtype=torch.float32, device="cuda")
#modified_contour_mel_tensor += torch.randn_like(modified_contour_mel_tensor) * noise_amp
#if snap_sensitivity is not None:
    #    modified_contour_mel_tensor = snap(modified_contour_mel_tensor, snap_sensitivity)
modified_contour_mel_tensor = postprocess(model(preprocess(modified_contour_mel_tensor.unsqueeze(0).unsqueeze(0)))).squeeze(0).squeeze(0)
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
