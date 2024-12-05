import os
import sys
import traceback

from dotenv import load_dotenv
load_dotenv()
load_dotenv("sha256.env")

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

device = "cuda"
exp_dir = "D:/matthew99/rvc/Retrieval-based-Voice-Conversion-WebUI/logs/mi-test"
import fairseq
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

now_dir = os.getcwd()
sys.path.append(now_dir)
from infer.lib.audio import load_audio
from infer.lib.audio import extract_features_new

from infer.modules.vc.modules import VC
from configs.config import Config

config = Config()
vc = VC(config)
vc.enc_q = True
vc.get_vc("D:/matthew99/rvc/Retrieval-based-Voice-Conversion-WebUI/enc_q.pth")

from f0_magic import resize_with_zeros


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
    f0_mel = 1127 * np.log(1 + f0 / 700)
    return (f0_mel - 550) / 120


print(" ".join(sys.argv))
model_path = "assets/hubert/hubert_base.pt"

print("exp_dir: " + exp_dir)
wavPath = "%s/0_gt_wavs" % exp_dir
outPath = "%s/3_feature193" % exp_dir
os.makedirs(outPath, exist_ok=True)

todo = sorted(list(os.listdir(wavPath)))
n = max(1, len(todo) // 50)
if len(todo) == 0:
    print("no-feature-todo")
else:
    print("all-feature-%s" % len(todo))
    for idx, file in enumerate(todo):
        try:
            if file.endswith(".wav"):
                wav_path = "%s/%s" % (wavPath, file)
                out_path = "%s/%s" % (outPath, file.replace("wav", "npy"))

                feats = vc.get_hidden_features(0, wav_path)
                f0 = resize_with_zeros(compute_f0(wav_path), feats.shape[0])
                feats = np.concatenate((feats, f0[:, np.newaxis]), axis=1)
                feats[:, -1] *= 192**0.5

                if np.isnan(feats).sum() == 0:
                    np.save(out_path, feats, allow_pickle=False)
                else:
                    print("%s-contains nan" % file)
                if idx % n == 0:
                    print("now-%s,all-%s,%s,%s" % (len(todo), idx, file, feats.shape))
        except:
            print(traceback.format_exc())
    print("all-feature-done")
