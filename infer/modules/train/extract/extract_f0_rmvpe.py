import os
import sys
import traceback

import parselmouth

now_dir = os.getcwd()
sys.path.append(now_dir)
import logging

import numpy as np
import pyworld
import torch, torchcrepe

from infer.lib.audio import load_audio

logging.getLogger("numba").setLevel(logging.WARNING)

n_part = int(sys.argv[1])
i_part = int(sys.argv[2])
i_gpu = sys.argv[3]
os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
exp_dir = sys.argv[4]
is_half = sys.argv[5]
f = open("%s/extract_f0_feature.log" % exp_dir, "a+")


def printt(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()


class FeatureInput(object):
    def __init__(self, samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0(self, path, f0_method):
        x = load_audio(path, self.fs)
        # p_len = x.shape[0] // self.hop
        if f0_method == "rmvpe":
            if hasattr(self, "model_rmvpe") == False:
                from infer.lib.rmvpe import RMVPE

                print("Loading rmvpe model")
                self.model_rmvpe = RMVPE(
                    "assets/rmvpe/rmvpe.pt", is_half=is_half, device="cuda"
                )
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)

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
                self.fs,
                160,
                self.f0_min,
                self.f0_max,
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

            source = f0_crepe
            source[source < 0.001] = np.nan
            target = np.interp(
                np.arange(0, len(source) * target_len, len(source)) / target_len,
                np.arange(0, len(source)),
                source
            )
            f0_crepe = np.nan_to_num(target)

            f0_rmvpe_mel = np.log(1 + f0 / 700)
            f0_crepe_mel = np.log(1 + f0_crepe / 700)
            f0 = np.where(np.logical_and(f0_rmvpe_mel > 0.001, f0_crepe_mel - f0_rmvpe_mel > 0.05), f0_crepe, f0)

        return f0

    def coarse_f0(self, f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        # use 0 or 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse

    def go(self, paths, f0_method):
        if len(paths) == 0:
            printt("no-f0-todo")
        else:
            printt("todo-f0-%s" % len(paths))
            n = max(len(paths) // 5, 1)  # 每个进程最多打印5条
            for idx, (inp_path, opt_root1, opt_root2, variations) in enumerate(paths):
                try:
                    if idx % n == 0:
                        printt("f0ing,now-%s,all-%s,-%s" % (idx, len(paths), inp_path))
                    featur_pit = self.compute_f0(inp_path, f0_method)
                    coarse_pit = self.coarse_f0(featur_pit)
                    for name in variations:
                        opt_path1 = "%s/%s" % (opt_root1, name)
                        opt_path2 = "%s/%s" % (opt_root2, name)
                        np.save(
                            opt_path2,
                            featur_pit,
                            allow_pickle=False,
                        )  # nsf
                        np.save(
                            opt_path1,
                            coarse_pit,
                            allow_pickle=False,
                        )  # ori
                except:
                    printt("f0fail-%s-%s-%s" % (idx, inp_path, traceback.format_exc()))


if __name__ == "__main__":
    # exp_dir=r"E:\codes\py39\dataset\mi-test"
    # n_p=16
    # f = open("%s/log_extract_f0.log"%exp_dir, "w")
    printt(sys.argv)
    featureInput = FeatureInput()
    paths = []
    inp_root = "%s/1_16k_wavs" % (exp_dir)
    opt_root1 = "%s/2a_f0" % (exp_dir)
    opt_root2 = "%s/2b-f0nsf" % (exp_dir)

    os.makedirs(opt_root1, exist_ok=True)
    os.makedirs(opt_root2, exist_ok=True)
    variations = dict()
    for name in sorted(list(os.listdir(inp_root))):
        if "(" not in name:
            if name not in variations:
                variations[name] = []
            variations[name].append(name)
            inp_path = "%s/%s" % (inp_root, name)
            if "spec" in inp_path:
                continue
            paths.append([inp_path, opt_root1, opt_root2, variations[name]])
        else:
            name_root = name[:name.find("(") - 1] + name[name.find("."):]
            if name_root not in variations:
                variations[name_root] = []
            variations[name_root].append(name)    
    try:
        featureInput.go(paths[i_part::n_part], "rmvpe")
    except:
        printt("f0_all_fail-%s" % (traceback.format_exc()))
    # ps = []
    # for i in range(n_p):
    #     p = Process(
    #         target=featureInput.go,
    #         args=(
    #             paths[i::n_p],
    #             f0method,
    #         ),
    #     )
    #     ps.append(p)
    #     p.start()
    # for i in range(n_p):
    #     ps[i].join()
