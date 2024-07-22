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
    try:
        if data["model_type"].lower().startswith("t"):
            preprocess = preprocess_t
        else:
            preprocess = preprocess_s
    except:
        preprocess = preprocess_s
    try:
        snap_sensitivity = float(data["snap_sensitivity"])
    except:
        snap_sensitivity = None
if not model_path.endswith(".pt"):
    model_path += ".pt"

model = PitchContourGenerator().to("cuda")
model.eval()
model.load_state_dict(torch.load(model_path)) 
print(f"Model loaded from '{model_path:s}'")

def walk(path):
   return sum(([os.path.join(dirpath, file_name) for file_name in filenames] for (dirpath, dirnames, filenames) in os.walk(path)), [])

WORK_PATH = "C:/datasets/singing_ai/f0_magic/work" 
for filename in walk(WORK_PATH):
    if os.path.splitext(filename)[1] == ".wav":
        print(filename)
        audio_file, feature_file = filename, filename
        input_file_p = os.path.splitext(audio_file)[0] + " p.npy"
        input_file_d = os.path.splitext(feature_file)[0] + " d.npy"

        output_file = os.path.splitext(audio_file)[0] + " p.npy"
        input_contour_mel = np.load(input_file_p)
        input_phone_diff = resize(np.load(input_file_d), len(input_contour_mel))
        modified_contour_mel = np.pad(input_contour_mel, (padding_size, padding_size))
        input_phone_diff_pad = np.pad(input_phone_diff, (padding_size, padding_size))
        extra = segment_size - ((len(modified_contour_mel) - 1) % segment_size + 1)
        modified_contour_mel = np.pad(modified_contour_mel, (extra, 0))
        input_phone_diff_pad = np.pad(input_phone_diff_pad, (extra, 0))
        modified_contour_mel_tensor = torch.tensor(modified_contour_mel, dtype=torch.float32, device="cuda")
        input_phone_diff_tensor = torch.tensor(input_phone_diff_pad, dtype=torch.float32, device="cuda")
        if snap_sensitivity is not None:
            modified_contour_mel_tensor = snap(modified_contour_mel_tensor, snap_sensitivity)
        modified_contour_mel_tensor = postprocess(model(preprocess(modified_contour_mel_tensor.unsqueeze(0).unsqueeze(0), input_phone_diff_tensor.unsqueeze(0).unsqueeze(0)))).squeeze(0).squeeze(0)
        modified_contour_mel = modified_contour_mel_tensor.detach().cpu().numpy()
        modified_contour_mel = modified_contour_mel[extra:]
        modified_contour_mel = modified_contour_mel[padding_size:-padding_size]
        #modified_contour_mel = median_filter(modified_contour_mel, size=17)
        #modified_contour_mel = modify_contour_mel(model, modified_contour_mel, threshold=threshold)
        #modified_contour_mel = pitch_shift_mel(modified_contour_mel, 0)


        modified_contour_mel[input_contour_mel < eps] = 0
        #modified_contour = pitch_blur(modified_contour, 1, 1, 1)
        np.save(output_file, modified_contour_mel)
