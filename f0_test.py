from f0_magic import *
from infer.lib.audio import pitch_blur
import json

model = PitchContourClassifier()
model_path = "model_bak.pt"
model.load_state_dict(torch.load(model_path)) 
print(f"Model loaded from '{model_path:s}'")

with open('f0_test_config.json', 'r') as openfile:
    data = json.load(openfile)
    audio_file = data["audio_file"]
    index_file = data["index_file"]

input_file = os.path.splitext(audio_file)[0] + ".npy"
if not os.path.isfile(input_file):
    np.save(input_file, compute_f0_inference(audio_file, index_file=index_file))

output_file = os.path.splitext(input_file)[0] + " out.npy"
#input_contour = np.load("input.npy")
input_contour = np.load(input_file)
input_contour = 1127 * np.log(1 + input_contour / 700) 
#input_contour = np.round(input_contour / 10) * 10
#length = len(input_contour)
#input_contour = resize_with_zeros(input_contour, length // 3)
#input_contour = resize_with_zeros(input_contour, length)
#input_contour = pitch_invert_mel(input_contour, "Ab4") 
input_contour = pitch_shift_mel(input_contour, 6)
modified_contour = modify_contour(model, input_contour, threshold=0.65)
#modified_contour = pitch_shift_mel(modified_contour, 0)

modified_contour = (np.exp(modified_contour / 1127) - 1) * 700
modified_contour[modified_contour < eps] = 0
#modified_contour = pitch_blur(modified_contour, 1, 1, 1)
np.save(output_file, modified_contour)
