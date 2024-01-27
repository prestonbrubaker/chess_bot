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
        self.fc1 = nn.Linear(269, 269)  # Input layer
        self.fc2 = nn.Linear(269, 269)  # Hidden layer 1
        self.fc3 = nn.Linear(269, 1)    # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Sigmoid activation for output
        return x



def find_most_recent_model(directory):
    model_files = [f for f in os.listdir(directory) if f.endswith('.pth')]
    if not model_files:
        return None
    
    # Assuming the file format is 'model_gen_X.pth', where X is the generation number
    latest_model = max(model_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
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
