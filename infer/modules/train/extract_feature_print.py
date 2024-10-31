import os
import sys
import traceback
import random

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

device = sys.argv[1]
n_part = int(sys.argv[2])
i_part = int(sys.argv[3])
if len(sys.argv) == 7:
    exp_dir = sys.argv[4]
    version = sys.argv[5]
    is_half = sys.argv[6].lower() == "true"
else:
    i_gpu = sys.argv[4]
    exp_dir = sys.argv[5]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
    version = sys.argv[6]
    is_half = sys.argv[7].lower() == "true"
import fairseq
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

now_dir = os.getcwd()
sys.path.append(now_dir)
from infer.lib.audio import load_audio
from infer.lib.audio import extract_features_new

GET_EXTENDED_FEATURES = False

index_file = None  # "D:/matthew99/rvc/Retrieval-based-Voice-Conversion-WebUI/logs/ipa/added_IVF521_Flat_nprobe_1_ipa_v2.index"
if index_file is not None:
    import faiss

    index = faiss.read_index(index_file)
    # big_npy = np.load(file_big_npy)
    big_npy = index.reconstruct_n(0, index.ntotal)

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


printt(" ".join(sys.argv))
model_path = "assets/hubert/hubert_base.pt"

printt("exp_dir: " + exp_dir)
wavPath = "%s/1_16k_wavs" % exp_dir
outPath = (
    "%s/3_feature256" % exp_dir if version == "v1" else "%s/3_feature768" % exp_dir
)
os.makedirs(outPath, exist_ok=True)


# wave must be 16k, hop_size=320
def readwave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats


# HuBERT model
printt("load model(s) from {}".format(model_path))
# if hubert model is exist
if os.access(model_path, os.F_OK) == False:
    printt(
        "Error: Extracting is shut down because %s does not exist, you may download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
        % model_path
    )
    exit(0)
models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    [model_path],
    suffix="",
)
model = models[0]
model = model.to(device)
printt("move model to %s" % device)
if is_half:
    if device not in ["mps", "cpu"]:
        model = model.half()
model.eval()

from infer.lib.rmvpe import RMVPE

model_rmvpe = RMVPE(
    "%s/rmvpe.pt" % os.environ["rmvpe_root"],
    is_half=False,
    device=device,
)

todo = sorted(list(os.listdir(wavPath)))[i_part::n_part]
n = max(1, len(todo) // 10)  # 最多打印十条
if len(todo) == 0:
    printt("no-feature-todo")
else:
    printt("all-feature-%s" % len(todo))
    for idx, file in enumerate(todo):
        try:
            if file.endswith(".wav"):
                wav_path = "%s/%s" % (wavPath, file)
                out_path = "%s/%s" % (outPath, file.replace("wav", "npy"))
                out_path_extended = "%s/%s_extended" % (
                    outPath,
                    file.replace("wav", "npy"),
                )

                if os.path.exists(out_path):
                    continue

                if "(" in file:
                    var_path = "%s_shifted/%s" % (wavPath, file)
                    audio_shifted = load_audio(var_path, 16000)
                else:
                    audio_shifted = None

                audio = load_audio(wav_path, 16000)
                feats = extract_features_new(
                    audio, audio_shifted, model=model, version=version, device=device
                )

                feats = feats.squeeze(0).float().cpu().numpy()

                if index_file is not None and random.randint(0, 1) == 0:
                    score, ix = index.search(feats, k=8)
                    weight = np.square(1 / score)
                    weight /= weight.sum(axis=1, keepdims=True)
                    feats = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

                if np.isnan(feats).sum() == 0:
                    np.save(out_path, feats, allow_pickle=False)
                    if GET_EXTENDED_FEATURES and "(" not in file:
                        f0 = model_rmvpe.infer_from_audio(audio, thred=0.03)
                        pd = np.ones_like(f0)
                        pd[f0 < 0.001] = 0
                        pd = np.interp(
                            np.arange(0, len(pd) * feats.shape[0], len(pd))
                            / feats.shape[0],
                            np.arange(0, len(pd)),
                            pd,
                        )
                        safe = np.abs(pd - 0.5) > 0.4999
                        safe = np.pad(
                            np.logical_and(
                                np.logical_and(safe[:-2], safe[1:-1]), safe[2:]
                            ),
                            (1, 1),
                        )
                        zeros = pd < 0.5
                        pd = np.ones_like(pd)
                        pd[zeros] = 0
                        feats_extended = np.concatenate(
                            (feats, pd[:, np.newaxis]), axis=1
                        )
                        np.save(
                            out_path_extended, feats_extended[safe], allow_pickle=False
                        )
                else:
                    printt("%s-contains nan" % file)
                if idx % n == 0:
                    printt("now-%s,all-%s,%s,%s" % (len(todo), idx, file, feats.shape))
        except:
            printt(traceback.format_exc())
    printt("all-feature-done")
