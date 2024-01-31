import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os

LOAD_SAVED_MODEL = True
MODEL_SAVE_PATH = 'chess_cnn_model.pth'

# Custom dataset class
class ChessDataset(Dataset):
    def __init__(self, board_file_path, score_file_path):
        self.board_data = []
        self.scores = []

        with open(board_file_path, 'r') as board_file, open(score_file_path, 'r') as score_file:
            board_lines = board_file.read().split('\n\n')
            score_lines = score_file.read().split('\n\n')

            for board_str, score_str in zip(board_lines, score_lines):
                if score_str.strip():  # Check for empty lines in score data
                    board_rows = board_str.strip().split('\n')
                    if len(board_rows) != 16:
                        continue  # Skip incorrect board sizes
                    board_data = [[int(cell) for cell in row.strip()] for row in board_rows]
                    board_tensor = torch.tensor(board_data, dtype=torch.float32)
                    self.board_data.append(board_tensor.unsqueeze(0))  # Add channel dimension
                    score = float(score_str.strip())
                    self.scores.append(score)

    def __getitem__(self, idx):
        return self.board_data[idx], self.scores[idx]

    def __len__(self):
        return len(self.board_data)

# Load data
dataset = ChessDataset('board_data.txt', 'score_data_per_turn.txt')

# Define data loaders
batch_size = 1000
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define CNN model
class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        # Define the CNN layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Adjusted input size for the fully connected layer
        self.fc1 = nn.Linear(in_features=64, out_features=128)  # Corrected input size
        self.fc2 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        # Debugging print
        #print(f"Input to CNN shape: {x.shape}")

        x = self.pool(F.relu(self.conv1(x)))
        # Debugging print
        #print(f"After first convolution shape: {x.shape}")
        # Apply convolutional and pooling layers
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Reshape the tensor for the fully connected layers
        x = x.view(-1, 64)  # Corrected size
        
        # Apply fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x




def load_model_if_available(model, path):
    if LOAD_SAVED_MODEL and os.path.isfile(path):
        model.load_state_dict(torch.load(path))
        print(f"Loaded model from {path}.")
    else:
        if LOAD_SAVED_MODEL:
            print(f"No saved model found at {path}. Starting with a new model.")
        else:
            print("LOAD_SAVED_MODEL is set to False. Starting with a new model.")
    return model

# Check if CUDA (GPU support) is available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = ChessCNN()
model = load_model_if_available(model, MODEL_SAVE_PATH)
model.to(device)  # Move the model to the GPU if available

# Define loss function and optimizer
criterion = torch.nn.MSELoss()  # Use mean squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)




# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        inputs = inputs.float()  # Convert input to Float data type
        labels = labels.float()  # Convert labels to Float data type
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for batch in val_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
    print(f"Epoch {epoch+1} - Validation Loss: {val_loss/len(val_loader)}")
    with open("fitness_log.txt", 'a') as file:
        file.write(f"Epoch {epoch+1} - Validation Loss: {val_loss/len(val_loader)}\n")
    if(epoch % 10 == 0):
        torch.save(model.state_dict(), 'chess_cnn_model.pth')
        print("MODEL SAVED")

# Save the trained model
torch.save(model.state_dict(), 'chess_cnn_model.pth')
