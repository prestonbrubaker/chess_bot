import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# Custom dataset class
class ChessDataset(Dataset):
    def __init__(self, board_file_path, score_file_path):
        self.board_data = []
        self.scores = []

        with open(board_file_path, 'r') as board_file, open(score_file_path, 'r') as score_file:
            board_lines = board_file.read().split('\n\n')
            score_lines = score_file.read().split('\n\n')

            for board_str, score_str in zip(board_lines, score_lines):
                # Check for empty lines in score data
                if score_str.strip():
                    # Parse board data into a tensor
                    board_data = [[int(cell) for cell in row] for row in board_str.split('\n') if row]
                    board_tensor = torch.tensor(board_data, dtype=torch.float32)
                    self.board_data.append(board_tensor.unsqueeze(0))  # Add a channel dimension

                    score = float(score_str.strip())
                    self.scores.append(score)

    def __len__(self):
        return len(self.board_data)

    def __getitem__(self, idx):
        board_str = self.board_data[idx]
        print(f"Board string (before processing): {board_str}")  # Debugging print
    
        # Process the board string
        board_data = [[int(cell) for cell in row[:16]] for row in board_str.split('\n') if row]
        board_tensor = torch.tensor(board_data, dtype=torch.float32).unsqueeze(0)
    
        print(f"Processed board shape: {board_tensor.shape}, Sample score: {self.scores[idx]}")  # Debugging print
        return board_tensor, self.scores[idx]

# Load data
dataset = ChessDataset('board_data.txt', 'score_data.txt')

# Define data loaders
batch_size = 2000
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define CNN model
class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        # Define the CNN layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Adjusted input size for the fully connected layer
        self.fc1 = nn.Linear(in_features=64, out_features=128)  # Corrected input size
        self.fc2 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        # Apply convolutional and pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Reshape the tensor for the fully connected layers
        x = x.view(-1, 64)  # Corrected size
        
        # Apply fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Check if CUDA (GPU support) is available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = ChessCNN()

model.to(device)  # Move the model to the GPU if available

# Define loss function and optimizer
criterion = torch.nn.MSELoss()  # Use mean squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        inputs = inputs.float()  # Convert input to Float data type
        labels = labels.float()  # Convert labels to Float data type

        # Debugging print statement
        if batch_idx == 0:  # Print for the first batch of each epoch
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Input shape: {inputs.shape}, Labels shape: {labels.shape}", flush=True)
        
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
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {val_loss/len(val_loader)}")
    with open("fitness_log.txt", 'a') as file:
        file.write(f"Epoch {epoch+1} - Validation Loss: {val_loss/len(val_loader)}\n")
    if(epoch % 1 == 0):
        torch.save(model.state_dict(), 'chess_cnn_model.pth')
        print("MODEL SAVED")

# Save the trained model
torch.save(model.state_dict(), 'chess_cnn_model.pth')
