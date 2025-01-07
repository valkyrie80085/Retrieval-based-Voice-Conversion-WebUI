import os
import sys
import traceback
import random
import shutil

from dotenv import load_dotenv

load_dotenv()
load_dotenv("sha256.env")

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

device = "cuda"
exp_dir = "D:/matthew99/rvc/Retrieval-based-Voice-Conversion-WebUI/logs/mi-test"
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

now_dir = os.getcwd()
sys.path.append(now_dir)
from infer.lib.audio import load_audio
from infer.lib.audio import extract_features_simple_segment

from infer.modules.vc.modules import VC
from configs.config import Config
from scipy.io import wavfile

from scipy.interpolate import CubicSpline

eps = 1e-3

config = Config()
vc = VC(config)

from f0_magic import resize_with_zeros
from infer.lib.audio import pitch_blur

version = "v2"

from infer.modules.vc.utils import load_hubert

model = load_hubert(config=config, version=version)


model_rmvpe = None


def compute_f0(path):
    print("computing f0 for: " + path)
    x = load_audio(path, 16000)

    global model_rmvpe
    if model_rmvpe is None:
        from infer.lib.rmvpe import RMVPE

        print("Loading rmvpe model")
        model_rmvpe = RMVPE("assets/rmvpe/rmvpe.pt", is_half=False, device="cuda")
    f0 = model_rmvpe.infer_from_audio(x, thred=0.03)
    return f0


def add_noise(contour, amp=5, scale=1):
    zeros = contour < 0.001
    contour = 1127 * np.log(1 + contour / 700)
    length = int(contour.shape[0] / scale) + 1
    noise = np.random.normal(0, amp, length)
    if len(noise) != len(contour):
        noise = CubicSpline(np.arange(0, len(noise)), noise)(
            np.arange(0, len(noise) * len(contour), len(noise)) / len(contour)
        )
    contour_with_noise = contour + noise
    contour_with_noise = (np.exp(contour_with_noise / 1127) - 1) * 700
    contour_with_noise[zeros] = 0
    return contour_with_noise


model_path = "assets/hubert/hubert_base.pt"

modelsPath = "D:/matthew99/rvc/downloaded"
wavPath = "%s/0_gt_wavs" % exp_dir
outPath = "%s/3_feature768" % exp_dir
os.makedirs(outPath, exist_ok=True)

todo = sorted(list(os.listdir(wavPath)))
vc_todos = []
total = 0
for idx, file in enumerate(todo):
    if file.endswith(".wav"):
        total += 1
        vc_todos.append(file)

idx = 0
n = max(1, total // 100)


def perturb_waveform(waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    from nansy import change_gender_smart, random_eq, random_formant_f0

    perturbed_waveform = random_formant_f0(waveform, sr)
    perturbed_waveform = random_eq(perturbed_waveform, sr)
    return np.clip(perturbed_waveform, -1.0, 1.0)


for file in vc_todos:
    try:
        idx += 1
        wav_path = "%s/%s" % (wavPath, file)
        out_path = "%s/%s" % (outPath, file.replace("wav", "npy"))
        f0_npy_path = "%s/%s" % (outPath, file + ".npy")
        try:
            np.load(out_path)
        except:
            audio = load_audio(wav_path, 16000)
            if random.randint(0, 1) == 0:
                audio = perturb_waveform(audio)
            feats = extract_features_simple_segment(
                audio, model=model, version=version, device=device
            )

            feats = feats.squeeze(0).float().cpu().numpy()

            if np.isnan(feats).sum() == 0:
                np.save(out_path, feats, allow_pickle=False)
            else:
                print("%s-contains nan" % file)
            if idx % n == 0:
                print("now-%s,all-%s,%s,%s" % (idx, total, file, feats.shape))
    except:
        print(traceback.format_exc())
