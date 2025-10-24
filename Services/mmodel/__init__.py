import torch

model_path = "Mononito/trained_models/smd/machine-1-2/LOF_1.pth"
model = torch.load(model_path)  # Use CPU to avoid GPU requirements for inspection
print('hi')
print(model)  # Print the model architecture
