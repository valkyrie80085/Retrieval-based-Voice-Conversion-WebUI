import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import numpy as np
import os

import librosa
import ffmpeg

from infer.lib.audio import load_audio, pitch_blur_mel, extract_features_simple, trim_sides_mel
import torchcrepe
import random

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.interpolate import CubicSpline
from scipy.signal import medfilt 

from configs.config import Config
from infer.modules.vc.utils import load_hubert
config = Config()

eps = 0.001

def visualize_contour(contour):
    plt.figure(figsize=(10, 4))  # Adjust figure size as needed 
    plt.plot(contour)
    plt.xlabel("Time (samples)")
    plt.ylabel("Pitch (Hz)")
    plt.title("Pitch Contour")
    plt.grid(True) 
    plt.show()


sr = 16000
window_length = 160
frames_per_sec = sr // window_length
def resize_with_zeros(contour, target_len):
    a = contour.copy()
    a[a < eps] = np.nan
    a = np.interp(
        np.arange(0, len(a) * target_len, len(a)) / target_len,
        np.arange(0, len(a)),
        a
    )
    a = np.nan_to_num(a)
    return a


hubert_model = None
def trim_f0(f0, audio, index_file, version="v2"):
    global hubert_model

    if not os.path.isfile(index_file):
        return f0
    import faiss
    try:
        index = faiss.read_index(index_file)
        # big_npy = np.load(file_big_npy)
        big_npy = index.reconstruct_n(0, index.ntotal)
    except:
        print("Failed to read index file: \"{index_file:s}\"")
        return f0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hubert_model is None:
        hubert_model = load_hubert(config)

    feats = extract_features_simple(audio, model=hubert_model, version=version, device=device, is_half=config.is_half)
    npy = feats[0].cpu().numpy()
    npy = np.concatenate((npy, np.full((npy.shape[0], 1), 0.5)), axis=1)

    score, ix = index.search(npy, k=8)
    weight = np.square(1 / score)
    weight /= weight.sum(axis=1, keepdims=True)
    npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

    pd = npy[:, -1]
    pd = np.interp(
        np.arange(0, len(pd) * len(f0), len(pd)) / len(f0),
        np.arange(0, len(pd)),
        pd
    )

    threshold = 0.5
    for it in (range(len(f0)), reversed(range(len(f0)))):
        keep = False
        for i in it:
            if f0[i] > eps:
                if pd[i] > threshold:
                    keep = True
                if not keep:
                    f0[i] = 0
            else:
                keep = False

    return f0


model_rmvpe = None
def compute_f0_inference(path, index_file=""):
    print("computing f0 for: " + path)
    x = load_audio(path, 44100)
    x = librosa.resample(
        x, orig_sr=44100, target_sr=sr
    )

    global model_rmvpe
    if model_rmvpe is None:
        from infer.lib.rmvpe import RMVPE
        print("Loading rmvpe model")
        model_rmvpe = RMVPE(
            "assets/rmvpe/rmvpe.pt", is_half=False, device="cuda")
    f0 = model_rmvpe.infer_from_audio(x, thred=0.03)

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
        16000,
        160,
        50,
        1100,
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
    f0_crepe = resize_with_zeros(f0_crepe, target_len)

    f0_rmvpe_mel = np.log(1 + f0 / 700)
    f0_crepe_mel = np.log(1 + f0_crepe / 700)
    f0 = np.where(np.logical_and(f0_rmvpe_mel > eps, f0_crepe_mel - f0_rmvpe_mel > 0.05), f0_crepe, f0)

    f0_mel = 1127 * np.log(1 + f0 / 700)

    target_len = x.shape[0] // window_length
    f0_mel = resize_with_zeros(f0_mel, target_len)

    if index_file != "":
        f0_mel = trim_f0(f0_mel, x, index_file)

    f0_mel = trim_sides_mel(f0_mel, frames_per_sec)

    f0 = (np.exp(f0_mel / 1127) - 1) * 700 
    f0 = np.pad(f0, (300, 300))
    return f0


model_rmvpe = None
def compute_f0(path):
    print("computing f0 for: " + path)
    x = load_audio(path, 44100)
    x = librosa.resample(
        x, orig_sr=44100, target_sr=sr
    )

    global model_rmvpe
    if model_rmvpe is None:
        from infer.lib.rmvpe import RMVPE
        print("Loading rmvpe model")
        model_rmvpe = RMVPE(
            "assets/rmvpe/rmvpe.pt", is_half=False, device="cuda")
    f0 = model_rmvpe.infer_from_audio(x, thred=0.03)

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
        16000,
        160,
        50,
        1100,
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
    f0_crepe = resize_with_zeros(f0_crepe, target_len)

    f0_rmvpe_mel = np.log(1 + f0 / 700)
    f0_crepe_mel = np.log(1 + f0_crepe / 700)
    f0 = np.where(np.logical_and(f0_rmvpe_mel > eps, f0_crepe_mel - f0_rmvpe_mel > 0.05), f0_crepe, f0)

    f0_mel = 1127 * np.log(1 + f0 / 700)

    target_len = x.shape[0] // window_length
    f0_mel = resize_with_zeros(f0_mel, target_len)
    return f0_mel



TARGET_PATH = "f0_magic/target"
OTHERS_PATH = "f0_magic/others"

def walk(path):
   return sum(([os.path.join(dirpath, file_name) for file_name in filenames] for (dirpath, dirnames, filenames) in os.walk(path)), [])


def prepare_data():
    filenames = []
    for filename in walk(TARGET_PATH):
        if filename.endswith(".wav"): 
            filenames.append(filename)
    for filename in walk(OTHERS_PATH):
        if filename.endswith(".wav"):
            filenames.append(filename)
    for filename in filenames:
        npy_file = os.path.splitext(filename)[0] + ".npy"
        if not os.path.isfile(npy_file):
            try:
                np.save(npy_file, compute_f0(filename))
            except:
                os.remove(filename)


segment_size = 1782

def pitch_shift_mel(contour, semitones):
    contour = (np.exp(contour / 1127) - 1) * 700
    contour *= 2 ** (semitones / 12)
    contour = 1127 * np.log(1 + contour / 700)
    contour[contour < eps] = 0
    return contour


def pitch_invert_mel(contour, note):
    contour = (np.exp(contour / 1127) - 1) * 700
    contour[contour > 0] = (librosa.note_to_hz(note) ** 2) / contour[contour > 0]
    contour = 1127 * np.log(1 + contour / 700)
    contour[contour < eps] = 0
    return contour


def add_noise(contour, amp=5, scale=1):
    zeros = contour < eps
    length = int(contour.shape[0] / scale) + 1
    noise = np.random.normal(0, amp, length)
    if len(noise) != len(contour):
        noise = CubicSpline(np.arange(0, len(noise)), noise)(np.arange(0, len(noise) * len(contour), len(noise)) / len(contour))
    contour_with_noise = contour + noise
    contour_with_noise[zeros] = 0
    return contour_with_noise


def get_average(contour):
    try:
        return np.average(contour[contour > eps])
    except ZeroDivisionError:
        return 0


def change_vibrato(contour, factor):
    blurred = pitch_blur_mel(contour, frames_per_sec)
    modified_contour = blurred + factor * (contour - blurred)
    modified_contour[modified_contour < eps] = 0
    return modified_contour


def modify_ends(contour):
    from scipy.ndimage import gaussian_filter1d
    contour_pad = np.concatenate(([0], contour))
    contour_segments = np.split(contour_pad, np.where(contour_pad < eps)[0])
    border_length = random.randint(4, 24)
    amount = random.uniform(30, 60) * random.choice((-1, 1))
    t = random.randint(0, 1)
    mask = np.hanning(border_length * 2)
    if t == 0:
        mask = mask[border_length:]
    else:
        mask = mask[:border_length]
    mask *= amount
    modified_segments = []
    for segment in contour_segments:
        if segment.shape[0] > 0:
            if len(segment) > border_length:
                if t == 0:
                    segment[1:border_length + 1] += mask
                else:
                    segment[-border_length:] += mask
            modified_segments.append(segment)
    modified_contour = np.concatenate(modified_segments)[1:]
    return modified_contour


def load_data():
    multiplicity_target = 40
    multiplicity_others = 40
    min_ratio = 0.25
    noise_rate = 1.0

    prepare_data()
    train_target_data = []
    train_others_data = []
    test_target_data = []
    test_others_data = []
    test_set = set()
    for filename in walk(TARGET_PATH) + walk(OTHERS_PATH):
        if filename.endswith(".npy"): 
            if random.uniform(0, 1) < 0.2:
                test_set.add(filename)
    for filename in walk(TARGET_PATH):
        if filename.endswith(".npy"): 
            if filename in test_set:
                target_data, others_data = test_target_data, test_others_data
            else:
                target_data, others_data = train_target_data, train_others_data
            contour = np.load(filename)
            if contour.shape[0] < segment_size:
                contour = np.pad(contour, (0, segment_size - contour.shape[0]))
            for i in range(int(multiplicity_target * (contour.shape[0] - segment_size) / segment_size) + 1):
                start = random.randint(0, contour.shape[0] - segment_size)
                if np.sum(contour[start:start + segment_size] > eps) > segment_size * min_ratio:
                    target_data.append(torch.tensor(contour[start:start + segment_size], dtype=torch.float32))
                    if random.uniform(0, 1) < noise_rate:
                        others_data.append(torch.tensor(add_noise(contour[start:start + segment_size]), dtype=torch.float32))
#                    others_data.append(torch.tensor(pitch_blur_mel(contour[start:start + segment_size], frames_per_sec), dtype=torch.float32))
#                    others_data.append(torch.tensor(change_vibrato(contour[start:start + segment_size], 5), dtype=torch.float32))
#                    others_data.append(torch.tensor(modify_ends(contour[start:start + segment_size]), dtype=torch.float32))

    if multiplicity_others > 0:
        min_segment_size = int(segment_size * 0.8)
        max_segment_size = int(segment_size * 1.25)
        for filename in walk(OTHERS_PATH):
            if filename.endswith(".npy"):
                if filename in test_set:
                    target_data, others_data = test_target_data, test_others_data
                else:
                    target_data, others_data = train_target_data, train_others_data
                contour = np.load(filename)
                to_val = random.uniform(0, 1) < 0.2
                if contour.shape[0] < max_segment_size:
                    contour = np.pad(contour, (0, max_segment_size - contour.shape[0]))
                for i in range(int(multiplicity_others * (contour.shape[0] - segment_size) / segment_size) + 1):
                    start = random.randint(0, contour.shape[0] - segment_size)
                    use_original = random.randint(0, 4) == 0
                    if use_original:
                        actual_length = segment_size
                    else:
                        actual_length = random.randint(min_segment_size, max_segment_size)
                        actual_length = min(actual_length, contour.shape[0])
                    start = random.randint(0, contour.shape[0] - actual_length)
                    contour_sliced = contour[start:start + actual_length].copy()
                    if actual_length != segment_size:
                        contour_sliced = resize_with_zeros(contour_sliced, segment_size)
                    if np.sum(contour_sliced > eps) > segment_size * min_ratio:
                        if use_original:
                            shift_real = 0
                        else:
                            contour_sliced = change_vibrato(contour_sliced, random.uniform(0, 2))
                            shift = random.uniform(0, 1)
                            average = get_average(contour_sliced) / 1127
                            LOW, HIGH = librosa.note_to_hz("Ab3"), librosa.note_to_hz("Bb5")
                            LOW, HIGH = math.log(1 + LOW / 700), math.log(1 + HIGH / 700)
                            average_goal = (HIGH - LOW) * shift + LOW
                            average = (math.exp(average) - 1) * 700
                            average_goal = (math.exp(average_goal) - 1) * 700
                            shift_real = math.log(average_goal / average) / math.log(2) * 12
                        contour_final = pitch_shift_mel(contour_sliced, shift_real)
                        if False and random.uniform(0, 1) < 0.1:
                            if random.randint(0, 4) == 0:
                                amp = 5
                                scale = 1
                            else:
                                amp = random.uniform(5, 30)
                                scale = random.uniform(1, 2 ** random.randint(0, 10))
                            contour_final = add_noise(contour_final, amp=amp, scale=scale)
                        others_data.append(torch.tensor(contour_final, dtype=torch.float32))

    print("Train target data count:", len(train_target_data))
    print("Train others data count:", len(train_others_data))
    print("Test target data count:", len(test_target_data))
    print("Test others data count:", len(test_others_data))
    return map(torch.stack, (train_target_data, train_others_data, test_target_data, test_others_data))


#channels = [3, 7, 7, 11, 11]
channels = [64, 128, 256, 512, 256, 256]
class PitchContourClassifier(nn.Module):
    def __init__(self):
        super(PitchContourClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=channels[0], kernel_size=8)
        self.pool1 = nn.LPPool1d(norm_type=2, kernel_size=4) 
        self.conv2 = nn.Conv1d(in_channels=channels[0], out_channels=channels[1], kernel_size=10)
        self.pool2 = nn.LPPool1d(norm_type=2, kernel_size=5) 
        self.conv3 = nn.Conv1d(in_channels=channels[1], out_channels=channels[2], kernel_size=8)
        self.pool3 = nn.LPPool1d(norm_type=2, kernel_size=4) 
        self.conv4 = nn.Conv1d(in_channels=channels[2], out_channels=channels[3], kernel_size=6)
        self.pool4 = nn.LPPool1d(norm_type=2, kernel_size=3)
        self.fc1 = nn.Linear(channels[-3] * 4, channels[-2])
        self.fc2 = nn.Linear(channels[-2], channels[-1])
        self.fc3 = nn.Linear(channels[-1], 1)


    def forward(self, x):
        x = self.pool1(F.elu(self.conv1(x)))
        x = self.pool2(F.elu(self.conv2(x)))
        x = self.pool3(F.elu(self.conv3(x)))
        x = self.pool4(F.elu(self.conv4(x)))
        x = x.view(-1, channels[-3] * 4)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


USE_TEST_SET = True
def train_model(name, train_target_data, train_others_data, test_target_data, test_others_data, include_fakes=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PitchContourClassifier().to(device)
    lr = 1e-6
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epoch = 0

    MODEL_FILE = name + ".pt"
    CHECKPOINT_FILE = name + " checkpoint.pt"
    TMP_FILE = name + "_tmp.pt"
    BAK_FILE = name + "_bak.pt"
#    FAKE_DATA_FILE = name + "_fakes.npy"
    if os.path.isfile(TMP_FILE):
        if not os.path.isfile(CHECKPOINT_FILE):
            os.rename(TMP_FILE, CHECKPOINT_FILE)
        else:
            os.remove(TMP_FILE)
    try:
        if os.path.isfile(CHECKPOINT_FILE):
            checkpoint = torch.load(CHECKPOINT_FILE)
            epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Data loaded from '{CHECKPOINT_FILE:s}'")
        else:
            print("Model initialized with random weights")
    except:
        epoch = 0
        model = PitchContourClassifier().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        print("Model initialized with random weights")

    criterion = nn.BCELoss()

#    lr_fakes = 0.01
#    lr_epoch = 10
#    fake_count = len(others_data) // lr_epoch
#    reset_prob = 1 / 1000 * lr_epoch
#    others_data_npy = others_data.cpu().numpy()
#    fakes_loaded = False
#    if os.path.isfile(FAKE_DATA_FILE):
#                fakes = np.load(FAKE_DATA_FILE)
#        if len(fakes) == fake_count:
            #            fakes_loaded = True
#        else:
#                        fake_count = len(fakes)
#            fakes_loaded = True
#    if not fakes_loaded:
        #        fakes = np.array([others_data_npy[random.randint(0, len(others_data_npy) - 1)] for i in range(fake_count)])

    train_dataset = torch.utils.data.TensorDataset(train_target_data, torch.full((len(train_target_data),), 1.0))
    train_dataset += torch.utils.data.TensorDataset(train_others_data, torch.full((len(train_others_data),), 0.0))
    if USE_TEST_SET:
        test_dataset = torch.utils.data.TensorDataset(test_target_data, torch.full((len(test_target_data),), 1.0))
        test_dataset += torch.utils.data.TensorDataset(test_others_data, torch.full((len(test_others_data),), 0.0))
    else:
        train_dataset = torch.utils.data.TensorDataset(test_target_data, torch.full((len(test_target_data),), 1.0))
        train_dataset += torch.utils.data.TensorDataset(test_others_data, torch.full((len(test_others_data),), 0.0))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    if USE_TEST_SET:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=True)
    best_loss = float("inf")
    while True:
#        train_dataset = torch.utils.data.TensorDataset(target_data, torch.full((len(target_data),), 1.0))
#        train_dataset += torch.utils.data.TensorDataset(others_data, torch.full((len(others_data),), 0.0))
#        fakes_dataset = torch.utils.data.TensorDataset(torch.stack([torch.tensor(data, dtype=torch.float32) for data in fakes]), torch.full((fake_count,), 0.0))
#        train_dataset += fakes_dataset

#        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

        epoch += 1

        train_loss = 0.0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(data.unsqueeze(1))
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()  
        train_loss /= len(train_loader)

        if USE_TEST_SET:
            test_loss = 0.0
            for batch_idx, (data, labels) in enumerate(test_loader):
                data, labels = data.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(data.unsqueeze(1))
                loss = criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()

                test_loss += loss.item()  
            test_loss /= len(test_loader)

#        fakes_loader = torch.utils.data.DataLoader(fakes_dataset, batch_size=256, shuffle=True)
#        fakes_similarity = 0
#        if False:
#            fakes_new = torch.empty((0, segment_size), dtype=torch.float32).to(device)
#            for batch_idx, (data, labels) in enumerate(fakes_loader):
#                data, labels = data.to(device), labels.to(device)
#                data.requires_grad_(True)
#                fakes_optimizer = optim.Adam([data], lr=lr_fakes) 
#                for i in range(lr_epoch):
#                    fakes_optimizer.zero_grad()
#                    fakes_outputs = model(data.unsqueeze(1))
#                    fakes_loss = torch.sum(1 - fakes_outputs)
#                    current_similarity = torch.sum(fakes_outputs)
#                    fakes_loss.backward()
#                    fakes_optimizer.step()
#                fakes_similarity += current_similarity
#                fakes_new = torch.cat((fakes_new, data.detach()))
#            fakes = fakes_new.cpu().numpy()
#        else:
#            for batch_idx, (data, labels) in enumerate(fakes_loader):
#                data, labels = data.to(device), labels.to(device)
#                fakes_outputs = model(data.unsqueeze(1))
#                current_similarity = torch.sum(fakes_outputs)
#                fakes_similarity += current_similarity
#            fakes_similarity += current_similarity
#        fakes_similarity /= fake_count


#        for i in range(fake_count):
#            if random.uniform(0, 1) < reset_prob:
#                fakes[i] = others_data_npy[random.randint(0, len(others_data_npy) - 1)]
        # Print epoch statistics
        if epoch % 1 == 0:
            if USE_TEST_SET:
                print(f"Epoch: {epoch:d} Train Loss: {train_loss:.6f} Test Loss: {test_loss:.6f}")# Gen Similarity: {fakes_similarity:.4f}") 
            else:
                print(f"Epoch: {epoch:d} Train Loss: {train_loss:.6f}")# Gen Similarity: {fakes_similarity:.4f}") 
            if epoch > 250 and train_loss > 1.0:
                print("Gradient seems to be exploding. Emergency stopping.")
                break
            checkpoint = { 
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}
            while True:
                try:
                    torch.save(model.state_dict(), MODEL_FILE) 
                    break
                except:
                    pass
            torch.save(checkpoint, TMP_FILE)
            if os.path.isfile(CHECKPOINT_FILE):
                while True:
                    try:
                        os.remove(CHECKPOINT_FILE)
                        break
                    except:
                        pass
            if os.path.isfile(CHECKPOINT_FILE):
                while True:
                    try:
                        os.rename(TMP_FILE, CHECKPOINT_FILE)
                        break
                    except:
                        pass
            try:
                #            np.save(FAKE_DATA_FILE, fakes)
                pass
            except:
                pass
            print(f"Data saved.")
        if (USE_TEST_SET and test_loss < best_loss) or ((not USE_TEST_SET) and train_loss < best_loss):
            best_loss = test_loss if USE_TEST_SET else train_loss 
            while True:
                try:
                    torch.save(model.state_dict(), BAK_FILE) 
                    break
                except:
                    pass
            print(f"Model backed up to '{BAK_FILE:s}'")

def modify_contour(model, original_contour, threshold=0.65):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    original_contour = np.pad(original_contour, (segment_size, segment_size))
    original_contour_tensor = torch.tensor(original_contour, dtype=torch.float32, device=device)
    changes = torch.zeros(len(original_contour), dtype=torch.float32, device=device)
    changes.requires_grad_(True)

    optimizer = optim.Adam([changes], lr=0.1) 
#    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9999)
#    epoch_count = 10000

    max_lives = 100
    best_output = None
    best_changes = None
    lives = max_lives
    epoch = 0
    beta = 0.99
    output_raw = 0
    beta_decayed = 1
    while True:
        epoch += 1
#    for epoch in range(epoch_count):
        start = random.randint(1, segment_size)
        changed = original_contour_tensor + changes
#        changed[original_contour_tensor < eps] = 0
        modified_contours = []
        cur = start
        while cur + segment_size < len(changes):
            if torch.sum(original_contour_tensor[cur:cur + segment_size] > eps) > 0:
                modified_contours.append((changed[cur:cur + segment_size]).unsqueeze(0))
            cur += segment_size
        optimizer.zero_grad()
        outputs = model(torch.stack(modified_contours)).view(-1)
        cur = start
        sum_output = 0
        segment_count = 0
        norm = 0.5
        index = 0
        loss = 0
        while cur + segment_size < len(changes):
            if torch.sum(original_contour_tensor[cur:cur + segment_size] > eps) > 0:
                current_output = outputs[index]
                index += 1
                if current_output > 0:
                    loss += 1 - (current_output ** norm)
                else:
                    loss += 1 - current_output
                sum_output += current_output ** norm
                segment_count += 1
            cur += segment_size
        loss.backward()
        changes_zeroed = changes.clone()
        changes_zeroed[original_contour_tensor < eps] = 0
        mse_loss = F.mse_loss(changes_zeroed[1:], changes_zeroed[:-1])
        l1_loss = torch.mean(torch.abs(changes_zeroed))
#        loss = ((1 * l1_loss + 0 * mse_loss) * 1e-3 * segment_count)
#        loss.backward()
        if segment_count > 0: 
            output_current = (sum_output / segment_count) ** (1 / norm)
        else:
            output_current = 1
        output_raw = output_raw * beta + output_current * (1 - beta)
        beta_decayed *= beta
        output = output_raw / (1 - beta_decayed)
#        print(output)
        print("Similarity: %.4f MSE: %.4f L1: %.4f" % (output, mse_loss, l1_loss))
        if best_output is None:
            print("Initial similarity: %.4f" % output.squeeze(0).squeeze(0))
            best_output = output
            best_changes = changes.clone()
        if output > best_output:
            best_output = output
            best_changes = changes.clone()
            lives = max_lives
        else:
            lives -= 1
        if output >= threshold:# or (epoch > 1000 and lives == 0):
            break
        optimizer.step()

    print("Final similarity: %.4f" % output.squeeze(0).squeeze())
    best_contour = original_contour_tensor + changes
    final_contour = best_contour.detach().cpu().numpy() 
    final_contour = final_contour[segment_size:-segment_size]
    return final_contour


if __name__ == "__main__":
    random.seed(42)

    train_target_data, train_others_data, test_target_data, test_others_data = load_data()
    train_model("model", train_target_data, train_others_data, test_target_data, test_others_data)
