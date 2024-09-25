import os
import sys
import traceback

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

device = "cuda"
exp_dir = "D:/matthew99/AI/singing_ai/Retrieval-based-Voice-Conversion-WebUI/logs/mi-test"
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
vc.get_vc("D:/matthew99/AI/singing_ai/Retrieval-based-Voice-Conversion-WebUI/enc_q.pth")

from f0_magic import resize_with_zeros

if "privateuseone" not in device:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
else:
    import torch_directml

    device = torch_directml.device(torch_directml.default_device())

    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml

f = open("%s/extract_f0_feature.log" % exp_dir, "a+")


def printt(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()


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


printt(" ".join(sys.argv))
model_path = "assets/hubert/hubert_base.pt"

printt("exp_dir: " + exp_dir)
wavPath = "%s/0_gt_wavs" % exp_dir
outPath = (
    "%s/3_feature193" % exp_dir
)
os.makedirs(outPath, exist_ok=True)

todo = sorted(list(os.listdir(wavPath)))
n = max(1, len(todo) // 50)
if len(todo) == 0:
    printt("no-feature-todo")
else:
    printt("all-feature-%s" % len(todo))
    for idx, file in enumerate(todo):
        try:
            if file.endswith(".wav"):
                wav_path = "%s/%s" % (wavPath, file)
                out_path = "%s/%s" % (outPath, file.replace("wav", "npy"))

#                feats = vc.get_hidden_features(0, wav_path)
#                f0 = resize_with_zeros(compute_f0(wav_path), feats.shape[0])
#                feats = np.concatenate(
#                    (feats, f0[:, np.newaxis]), axis=1
#                )
                feats = np.load(out_path)
                feats[:, -1] *= 192 ** 0.5

                if np.isnan(feats).sum() == 0:
                    np.save(out_path, feats, allow_pickle=False)
                else:
                    printt("%s-contains nan" % file)
                if idx % n == 0:
                    printt("now-%s,all-%s,%s,%s" % (len(todo), idx, file, feats.shape))
        except:
            printt(traceback.format_exc())
    printt("all-feature-done")
