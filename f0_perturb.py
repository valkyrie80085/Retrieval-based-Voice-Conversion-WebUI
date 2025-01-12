import os
import librosa
import soundfile as sf
from infer.lib.audio import load_audio
from scipy.io import wavfile
import random
import numpy as np

f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)

def coarse_f0(f0):
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (
        f0_bin - 2
    ) / (f0_mel_max - f0_mel_min) + 1

    # use 0 or 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = np.rint(f0_mel).astype(int)
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
        f0_coarse.max(),
        f0_coarse.min(),
    )
    return f0_coarse

folder_f0 = "D:/matthew99/rvc/Retrieval-based-Voice-Conversion-WebUI/logs/mi-test/2b-f0nsf"
folder_coarse = "D:/matthew99/rvc/Retrieval-based-Voice-Conversion-WebUI/logs/mi-test/2a_f0"

for file_name in os.listdir(folder_f0):
    try:
        path = os.path.join(folder_f0, file_name)
        path_coarse = os.path.join(folder_coarse, file_name)

        limit = 0 if random.randint(0, 1) == 0 else random.randint(1, 10)

        f0 = np.load(path)
        old_f0 = f0.copy()
        last = 0
        for i in range(len(f0) + 1):
            if i == len(f0) or f0[i] > 0.001:
                if last < i:
                    left, right = None, None
                    if last > 0:
                        left = f0[last - 1]
                    if i < len(f0):
                        right = f0[i]
                    if left is None:
                        assert right is not None
                        left = right
                    elif right is None:
                        right = left
                    for j in range(last, i):
                        left_gap = limit + 1 if last == 0 else j - last + 1
                        right_gap = limit + 1 if i == len(f0) else i - j
                        if min(left_gap, right_gap) <= limit:
                            f0[j] = ((j - last) * right + (i - j) * left) / (i - last)
                last = i + 1
        f0_coarse = np.load(path_coarse)
        f0_coarse = coarse_f0(f0)
        
        np.save(path, f0, allow_pickle=False)
        np.save(path_coarse, f0_coarse, allow_pickle=False)
    except:
        pass
