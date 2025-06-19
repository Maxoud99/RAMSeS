import torch

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a tensor and move it to the GPU if available
tensor = torch.rand(10000, 10000)
tensor = tensor.to(device)
print(f"Tensor is on {tensor.device}")

# Perform a simple operation
result = tensor @ tensor  # Matrix multiplication
print(f"Result tensor is on {result.device}")

# Print a few elements of the result
print(result[:5, :5])
