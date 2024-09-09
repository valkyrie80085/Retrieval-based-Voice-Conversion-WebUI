import torch

import numpy as np

from f0_magic import compute_f0_inference, compute_d, resize, pitch_invert_mel, pitch_shift_mel
from f0_magic import postprocess, preprocess_s, preprocess_t, padding_size
from f0_magic_gen_legacy import PitchContourGenerator, segment_size
from f0_magic import snap
from f0_magic_new_diff import adjust_to_match
from configs.config import Config
from infer.modules.vc.utils import load_hubert
from infer.lib.audio import load_audio, extract_features_simple
config = Config()

import librosa
import random
import os
import json

from infer.lib.audio import pitch_blur

from torch.nn import functional as F
def adjust_to_match_(contour_x, contour_y):
    def get_max(contour):
        return F.max_pool1d(contour, kernel_size=25, stride=1, padding=12)


    def get_min(contour):
        contour_clone = (-contour).clone()
        contour_clone[contour < eps] = float("-inf")
        return -F.max_pool1d(contour_clone, kernel_size=25, stride=1, padding=12)


    def get_avg(contour):
        ret = (get_max(contour) + get_min(contour)) / 2
        ret[contour < eps] = 0
        return ret


    contour = contour_x.clone()
    for i in range(1):
        delta = get_avg(contour_y) - get_avg(contour)
        delta[contour < eps] = 0
        contour = contour + delta
        print(i, torch.mean((contour - contour_x) ** 2))
    return contour

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

audio = load_audio(audio_file, 44100)
audio = librosa.resample(
        audio, orig_sr=44100, target_sr=16000
        )

index_file = "D:/matthew99/AI/singing_ai/Retrieval-based-Voice-Conversion-WebUI/logs/ipa/added_IVF521_Flat_nprobe_1_ipa_v2.index" 
import faiss
index = faiss.read_index(index_file)
# big_npy = np.load(file_big_npy)
big_npy = index.reconstruct_n(0, index.ntotal)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hubert_model = load_hubert(config)

feats = extract_features_simple(audio, model=hubert_model, version="v2", device=device, is_half=config.is_half)
npy = feats[0].cpu().numpy()
feats_diff_pre = np.pad(np.linalg.norm(npy[:-1] - npy[1:], axis=1), (1, 0))

score, ix = index.search(npy, k=8)
weight = np.square(1 / score)
weight /= weight.sum(axis=1, keepdims=True)
npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
feats_diff_post = np.pad(np.linalg.norm(npy[:-1] - npy[1:], axis=1), (1, 0))

for i in range(len(feats_diff_pre)):
    print(i, feats_diff_pre[i], feats_diff_post[i])
