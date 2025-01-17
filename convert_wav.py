import os
import librosa
import soundfile as sf
from infer.lib.audio import load_audio
from scipy.io import wavfile
import numpy as np

folder_a = (
    "D:/matthew99/rvc/Retrieval-based-Voice-Conversion-WebUI/logs/mi-test/0_gt_wavs"
)
folder_b = (
    "D:/matthew99/rvc/Retrieval-based-Voice-Conversion-WebUI/logs/mi-test/1_16k_wavs"
)
folder_c = (
    "D:/matthew99/rvc/Retrieval-based-Voice-Conversion-WebUI/logs/mi-test/0_gt_wavs_"
)

os.makedirs(folder_c, exist_ok=True)
os.makedirs(folder_b, exist_ok=True)

for file_name in os.listdir(folder_a):
    if file_name.lower().endswith(".wav"):
        input_path = os.path.join(folder_a, file_name)
        output_path = os.path.join(folder_b, file_name)
        output_path2 = os.path.join(folder_c, file_name)

        # Load audio (force sample rate = 40000 to ensure correct input rate)
        audio = load_audio(input_path, 40000)

        # Resample to 16000
        audio_16k = librosa.resample(audio, orig_sr=40000, target_sr=16000)

        # Save the resampled audio
        wavfile.write(output_path2, 40000, audio.astype(np.float32))
        wavfile.write(output_path, 16000, audio_16k.astype(np.float32))
