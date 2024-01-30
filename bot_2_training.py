import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Custom dataset class
class ChessDataset(Dataset):
    def __init__(self, board_file_path, score_file_path):
        self.board_data = []
        self.scores = []

        with open(board_file_path, 'r') as board_file, open(score_file_path, 'r') as score_file:
            board_lines = board_file.read().split('\n\n')
            score_lines = score_file.read().split('\n\n')

            for board_str, score_str in zip(board_lines, score_lines):
                # Parse board data and convert to tensors
                board_data = [list(row) for row in board_str.split('\n')]
                board_tensor = torch.tensor(board_data, dtype=torch.float32)
                self.board_data.append(board_tensor)

                # Parse scores and convert to numerical values
                score = float(score_str.strip())
                self.scores.append(score)

    def __len__(self):
        return len(self.board_data)

    def __getitem__(self, idx):
        return self.board_data[idx], self.scores[idx]

# Load data
dataset = ChessDataset('board_data.txt', 'score_data.txt')

# Define data loaders
batch_size = 32
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define CNN model
class ChessCNN(torch.nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        # Define your CNN architecture here

    def forward(self, x):
        # Implement the forward pass of your CNN
        pass

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
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {val_loss/len(val_loader)}")

# Save the trained model
torch.save(model.state_dict(), 'chess_cnn_model.pth')

