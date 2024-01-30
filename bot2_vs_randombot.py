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
                    board_tensor = torch.tensor(board_data, dtype=torch.float32) / 255.0  # Normalize to [0, 1]
                    self.board_data.append(board_tensor.unsqueeze(0))  # Add a channel dimension

                    score = float(score_str.strip())
                    # Ensure that scores have the correct size
                    self.scores.append([score])  # Wrap score in a list

    def __len__(self):
        return len(self.board_data)

    def __getitem__(self, idx):
        return self.board_data[idx], self.scores[idx]

# Load data
dataset = ChessDataset('board_data.txt', 'score_data_per_turn.txt')

# Define data loaders
batch_size = 200
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

model = ChessCNN()

# Define loss function and optimizer
criterion = torch.nn.MSELoss()  # Use mean squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs, labels = batch
        inputs = inputs.float()  # Convert input to Float data type
        
        # Convert labels to Float data type, assuming labels are single float values
        labels = [score[0] for score in labels]  # Extract the float value
        labels = torch.tensor(labels, dtype=torch.float32)  # Convert to tensor
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Ensure the target size matches the input size by resizing labels
        labels = labels.view_as(outputs)  # Reshape labels to match the size of outputs
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for batch in val_loader:
            inputs, labels = batch
            inputs = inputs.float()
            
            # Convert labels to Float data type for validation as well
            labels = [score[0] for score in labels]  # Extract the float value
            labels = torch.tensor(labels, dtype=torch.float32)  # Convert to tensor
            
            outputs = model(inputs)
            
            # Resize labels for validation as well
            labels = labels.view_as(outputs)
            
            val_loss += criterion(outputs, labels).item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {val_loss/len(val_loader)}")
    if epoch % 10 == 0:
        torch.save(model.state_dict(), 'chess_cnn_model.pth')
        print("Model Saved")

# Save the trained model
torch.save(model.state_dict(), 'chess_cnn_model.pth')
