import torch
from Utils.model_io import load_detector

model_path = "Mononito/trained_models/smd/machine-1-2/LOF_1.pth"
# An inspector only prints the model, so it has no reason to touch a GPU.
model = load_detector(model_path, device="cpu")
print('hi')
print(model)  # Print the model architecture
