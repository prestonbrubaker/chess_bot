import chess
import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn

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
                    self.scores.append(score)

    def __len__(self):
        return len(self.board_data)

    def __getitem__(self, idx):
        return self.board_data[idx], self.scores[idx]

# Load data
dataset = ChessDataset('board_data.txt', 'score_data_per_turn.txt')

# Define data loaders
batch_size = 32
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define CNN model
class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        # Define the CNN layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=32 * 4 * 4, out_features=128)  # Adjust the input size
        self.fc2 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        # Apply convolutional and pooling layers
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        
        # Reshape the tensor for fully connected layers
        x = x.view(-1, 32 * 4 * 4)  # Adjust the size here
        
        # Apply fully connected layers with ReLU activation
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Load the trained CNN model
model = ChessCNN()
model.load_state_dict(torch.load('chess_cnn_model.pth'))
model.eval()

def encode_board(board):
    piece_mapping = {
        None: '0000',  # Empty square
        chess.Piece(chess.PAWN, chess.WHITE): '0001',
        chess.Piece(chess.ROOK, chess.WHITE): '0010',
        chess.Piece(chess.KNIGHT, chess.WHITE): '0011',
        chess.Piece(chess.BISHOP, chess.WHITE): '0100',
        chess.Piece(chess.QUEEN, chess.WHITE): '0101',
        chess.Piece(chess.KING, chess.WHITE): '0110',
        chess.Piece(chess.PAWN, chess.BLACK): '1001',
        chess.Piece(chess.ROOK, chess.BLACK): '1010',
        chess.Piece(chess.KNIGHT, chess.BLACK): '1011',
        chess.Piece(chess.BISHOP, chess.BLACK): '1100',
        chess.Piece(chess.QUEEN, chess.BLACK): '1101',
        chess.Piece(chess.KING, chess.BLACK): '1110',
    }

    encoded_board = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        encoded_piece = piece_mapping.get(piece, '0111')  # Default to reserved
        encoded_board.append(encoded_piece)

    return encoded_board

def create_2d_board(encoded_board):
    rows = []
    for i in range(0, len(encoded_board), 8):
        row = ''
        for j in range(8):
            piece_code = encoded_board[i + j]
            row += piece_code
        rows.append(row)
        rows.append(row)  # Duplicate the row for the 16x16 grid
    return rows

def predict_move(board):
    encoded_board = encode_board(board)
    two_d_board = create_2d_board(encoded_board)
    
    # Prepare the input tensor for the model
    input_board = torch.tensor([[list(map(int, row))] for row in two_d_board], dtype=torch.float32)
    
    # Use the model to predict the best move
    with torch.no_grad():
        predicted_score = model(input_board.unsqueeze(0))  # Add batch dimension
    return predicted_score.item()

def main():
    wins = 0
    draws = 0
    losses = 0
    
    for _ in range(10):  # Play 10 games
        board = chess.Board()
        
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                # White's turn, use CNN to predict the move
                best_move = None
                best_score = -float('inf')
                for move in board.legal_moves:
                    board.push(move)
                    score = predict_move(board)
                    board.pop()
                    if score > best_score:
                        best_move = move
                        best_score = score
                board.push(best_move)
            else:
                # Black's turn, random move
                random_move = random.choice(list(board.legal_moves))
                board.push(random_move)

        # Determine the game outcome
        if board.result() == "1-0":
            wins += 1
            print("CNN wins")
        elif board.result() == "0-1":
            losses += 1
            print("Random bot wins")
        else:
            draws += 1
            print("Draw")

    print(f"Wins: {wins}, Draws: {draws}, Losses: {losses}")

if __name__ == "__main__":
    main()
