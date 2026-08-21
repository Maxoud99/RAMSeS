import torch
from Utils.model_io import load_checkpoint

model_path = "Mononito/trained_models/smd/machine-1-2/LOF_1.pth"
model = load_checkpoint(model_path, map_location="cpu")
print('hi')
print(model)  # Print the model architecture
