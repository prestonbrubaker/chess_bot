import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os

class ChessNN(nn.Module):
    # Define your ChessNN class as before
    # ...

def find_most_recent_model(directory):
    model_files = [f for f in os.listdir(directory) if f.endswith('.pth')]
    if not model_files:
        return None
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(directory, x)))
    return os.path.join(directory, latest_model)

def visualize_layer_output(output, layer_name, folder):
    os.makedirs(folder, exist_ok=True)
    output = output - output.min()
    output = output / output.max()
    output = output.detach().numpy()

    for i, filter in enumerate(output[0]):
        img = Image.fromarray(np.uint8(filter * 255), 'L')
        img.save(os.path.join(folder, f"{layer_name}_filter_{i}.png"))

model_directory = "models"
model_path = find_most_recent_model(model_directory)
if model_path:
    model = ChessNN()
    model.load_state_dict(torch.load(model_path))
    print(f"Loaded model from {model_path}")

    # Create a dummy input tensor
    input_tensor = torch.randn(1, 1, 16, 16)

    # Extract convolutional layers from the loaded model
    conv1 = model.conv1
    conv2 = model.conv2

    # Get the output of each layer
    output_conv1 = F.relu(conv1(input_tensor))
    output_conv2 = F.relu(conv2(output_conv1))

    # Visualize the outputs
    visualize_layer_output(output_conv1, "conv1", "filter_photos")
    visualize_layer_output(output_conv2, "conv2", "filter_photos")
else:
    print("No model found in the 'models' directory.")
