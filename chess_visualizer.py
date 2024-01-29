import chess
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

class ChessNN(nn.Module):
    def __init__(self):
        super(ChessNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)  # Output: 3x16x16
        self.conv2 = nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1)  # Output: 2x16x16

        # Pooling to reduce to 2x8x8
        self.pool = nn.MaxPool2d(2, 2)  # Output: 2x8x8


        self.fc1 = nn.Linear(2 * 8 * 8 + 13, 64)
        self.fc2 = nn.Linear(64, 1)

        # Xavier initialization
        for layer in [self.fc1, self.fc2, self.conv1, self.conv2]:
            if hasattr(layer, 'weight'):
                init.xavier_uniform_(layer.weight)

    def forward(self, x):
        board = x[:, :256].view(-1, 1, 16, 16)
        additional_bits = x[:, 256:]

        board = torch.sigmoid(self.conv1(board))
        board = torch.sigmoid(self.conv2(board))
        board = self.pool(board)

        board = board.view(-1, 2 * 8 * 8)
        combined = torch.cat((board, additional_bits), dim=1)

        combined = torch.sigmoid(self.fc1(combined))
        output = torch.sigmoid(self.fc2(combined))

        return output



def find_most_recent_model(directory):
    model_files = [f for f in os.listdir(directory) if f.endswith('.pth') and 'best_model_gen_' in f]

    # Filter out files that do not match the expected pattern and extract generation numbers
    valid_files = []
    for f in model_files:
        parts = f.split('_')
        if len(parts) >= 6 and parts[5].split('.')[0].isdigit():
            gen_number = int(parts[5].split('.')[0])
            valid_files.append((f, gen_number))

    # Check if there are any files left after filtering
    if not valid_files:
        return None
    
    # Find the latest model based on the generation number
    latest_model = max(valid_files, key=lambda x: x[1])[0]
    return os.path.join(directory, latest_model)


# Create an instance of the ChessNN
model = ChessNN()

MOVE = 0
GAME = 0

def evaluate_position(encoded_state):
    # Convert the encoded state to a PyTorch tensor
    input_tensor = torch.FloatTensor(encoded_state).unsqueeze(0)  # Add batch dimension

    # Feed the tensor into the neural network
    output = model(input_tensor)

    return output.item()  # Return the single output value as a Python number


def print_board(board):
    """Prints the chess board in a simple text format."""
    print("  +------------------------+")
    for i in range(8, 0, -1):
        print(f"{i} |", end=" ")
        for j in range(8):
            piece = board.piece_at(chess.square(j, i-1))
            symbol = piece.symbol() if piece else "."
            print(symbol, end=" ")
        print("|")
    print("  +------------------------+")
    print("    a b c d e f g h")




def encode_game_state(board, move, is_white_turn):
    def piece_to_binary(piece):
        """Converts a chess piece to a 4-bit binary representation."""
        if piece is None:
            return [0, 0, 0, 0]
        binary = [int(bit) for bit in "{0:03b}".format(piece.piece_type)]
        binary.append(1 if piece.color == chess.BLACK else 0)
        return binary

    def move_to_binary(move):
        """Converts a chess move to a 12-bit binary representation."""
        from_square = [int(bit) for bit in "{0:06b}".format(move.from_square)]
        to_square = [int(bit) for bit in "{0:06b}".format(move.to_square)]
        return from_square + to_square

    # Encode the board state
    board_state = [bit for i in range(64) for bit in piece_to_binary(board.piece_at(i))]

    # Encode the move
    encoded_move = move_to_binary(move)

    # Encode the turn
    turn = [0] if is_white_turn else [1]

    # Combine all encodings into one list
    return board_state + turn + encoded_move


def choose_index_by_evaluation(evaluation_scores):
    total = sum(evaluation_scores)
    if total == 0:
        # If all scores are zero, choose randomly
        return random.randint(0, len(evaluation_scores) - 1)

    # Normalize scores to sum to 1 (if not already)
    normalized_scores = [score / total for score in evaluation_scores]

    # Choose an index based on these normalized weights
    chosen_index = random.choices(range(len(evaluation_scores)), weights=normalized_scores, k=1)[0]
    return chosen_index




# White is the neural network bot, and black is the human
def play_human_game(model):
    global MOVE

    board = chess.Board()
    MOVE = 0

    while not board.is_game_over():
        print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("~~~~~~~~~~~~~             MOVE " + str(MOVE) + "                 ~~~~~~~~~~~~~~~~")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

        print("\nCURRENT STATE OF THE BOARD\n")
        print_board(board)
        print("\nWHITE'S TURN" if board.turn else "\nBLACK'S TURN")

        legal_moves = list(board.legal_moves)
        print("\nPOSSIBLE MOVES:", ", ".join(map(str, legal_moves)))
        
        if board.turn == chess.WHITE:
            # AI's turn (White)
            is_white_turn = True
            evaluation_scores = []
            for possible_move in legal_moves:
                encoded_state = encode_game_state(board, possible_move, is_white_turn)
                evaluation = evaluate_position(encoded_state)
                evaluation_scores.append(evaluation)
            
            chosen_index = choose_index_by_evaluation(evaluation_scores)
            chosen_move = legal_moves[chosen_index]
            print("\nAI'S EVALUATION OF EACH MOVE: ", str(evaluation_scores))
            print("\nAI's MOVE:", chosen_move)
        else:
            # Human's turn (Black)
            move_uci = input("Enter your move: ")
            try:
                chosen_move = chess.Move.from_uci(move_uci)
                if chosen_move not in legal_moves:
                    print("Illegal move. Try again.")
                    continue
            except ValueError:
                print("Invalid input. Please enter a move in UCI format.")
                continue

        board.push(chosen_move)
        print("\nBOARD AFTER MOVE:")
        print_board(board)
        MOVE += 1

    # Determine the winner
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
    elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        winner = "Draw"
    else:
        winner = "Game not finished"

    print("\nGame over. Winner:", winner)

if __name__ == "__main__":
    model_directory = "models"
    model_path = find_most_recent_model(model_directory)

    if model_path:
        print(f"Loading model from {model_path}")
        model = ChessNN()
        model.load_state_dict(torch.load(model_path))
    else:
        print("No existing model found. Starting with a new model.")
        model = ChessNN()

    # Play against the model
    play_human_game(model)
