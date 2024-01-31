import chess
import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import numpy

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
                    self.board_data.append(board_tensor)  # No need to add a channel dimension

                    score = float(score_str.strip())
                    self.scores.append(score)

    def __len__(self):
        return len(self.board_data)

    def __getitem__(self, idx):
        return self.board_data[idx], self.scores[idx]


# Define CNN model
class ChessCNN(nn.Module):
    def __init__(self, input_size=(16, 16)):
        super(ChessCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate size after convolutions and pooling
        self.final_conv_size = self._get_conv_output(input_size)

        # Fully connected layers
        self.fc1 = nn.Linear(self.final_conv_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.autograd.Variable(torch.rand(1, *shape))
            output = self.pool(F.relu(self.conv1(input)))
            output = self.pool(F.relu(self.conv2(output)))
            output = self.pool(F.relu(self.conv3(output)))
            return int(numpy.prod(output.size()))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.final_conv_size)  # Flatten the tensor
        x = F.relu(self.fc1(x))
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

def predict_moves(board):
    legal_moves = list(board.legal_moves)
    move_scores = {}
    
    for move in legal_moves:
        board.push(move)
        encoded_board = encode_board(board)
        two_d_board = create_2d_board(encoded_board)
    
        # Prepare the input tensor for the model
        input_board = torch.tensor([list(map(int, row)) for row in two_d_board], dtype=torch.float32)
    
        # Reshape the input to [batch_size, channels, height, width]
        input_board = input_board.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
        # Use the model to predict the evaluation score
        with torch.no_grad():
            predicted_scores = model(input_board)
        
        if predicted_scores.size(0) == 1:
            predicted_score = predicted_scores.item()
            move_scores[move] = predicted_score  # Extract the scalar value correctly
        else:
            # Handle cases where the model returns more than one score
            # You can choose how to handle this based on your model's architecture
            
            # For example, you can take the average of all scores
            # predicted_score = predicted_scores.mean().item()
            
            # Or choose the score that corresponds to the desired move if it's in the list
            # predicted_score = predicted_scores[legal_moves.index(move)].item()
            
            # You should adapt this part based on your model's output
            
            # For simplicity, let's just take the first score
            predicted_score = predicted_scores[0].item()
            move_scores[move] = predicted_score
        
        board.pop()
    
    # Rank moves based on scores
    ranked_moves = sorted(move_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Select the move with the highest score
    best_move, _ = ranked_moves[0]
    
    return best_move

def main():
    wins = 0
    draws = 0
    losses = 0
    
    for _ in range(1000):  # Play 1000 games
        board = chess.Board()
        
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                # White's turn, use CNN to predict the best move
                best_move = predict_moves(board)
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
