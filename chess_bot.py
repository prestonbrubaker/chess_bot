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

def get_random_move(board):
    """Returns a random legal move for the current player."""
    return random.choice(list(board.legal_moves))

def run_multiple_games(number_of_games):
    white_wins = 0
    black_wins = 0
    draws = 0

    for game in range(number_of_games):
        winner = play_random_game()
        print("\nGAME", game, "OVER. Winner:", winner)

        if winner == "White":
            white_wins += 1
        elif winner == "Black":
            black_wins += 1
        elif winner == "Draw":
            draws += 1

    print("\nFinal Results after", number_of_games, "games:")
    print("White wins:", white_wins)
    print("Black wins:", black_wins)
    print("Draws:", draws)



def evaluate_fitness(model, number_of_games):
    white_wins = 0
    black_wins = 0

    for _ in range(number_of_games):
        winner = play_random_game(model)
        if winner == "White":
            white_wins += 1
        elif winner == "Black":
            black_wins += 1
    fitness = (white_wins - black_wins) / number_of_games
    return fitness




# White is the neural network bot, and black is the random bot
def play_random_game(model):
    global MOVE

    board = chess.Board()
    MOVE = 0

    while not board.is_game_over():
        #print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        #print("~~~~~~~~~~~~~             MOVE " + str(MOVE) + "                 ~~~~~~~~~~~~~~~~")
        #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

        #print("\nCURRENT STATE OF THE BOARD\n")
        #print_board(board)
        #print("\nWHITE'S TURN" if board.turn else "\nBLACK'S TURN")

        legal_moves = list(board.legal_moves)
        #print("\nPOSSIBLE MOVES:", ", ".join(map(str, legal_moves)))
        
        if board.turn == chess.WHITE:
            is_white_turn = True
            evaluation_scores = []
            for possible_move in legal_moves:
                encoded_state = encode_game_state(board, possible_move, is_white_turn)
                #print("\n\nENCODED STATE: " + str(encoded_state))

                evaluation = evaluate_position(encoded_state)
                evaluation_scores.append(evaluation)
                #print("\nEvaluation of the position:", evaluation)
            
            chosen_index = choose_index_by_evaluation(evaluation_scores)
            chosen_move = legal_moves[chosen_index]
        else:
            chosen_move = get_random_move(board)

        #print("\nCHOSEN MOVE:", chosen_move)

        board.push(chosen_move)
        #print("\nBOARD AFTER MOVE:")
        #print_board(board)
        #print("\n\n\n")
        MOVE += 1

    # Determine the winner
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
    elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        winner = "Draw"
    else:
        winner = "Game not finished"

    return winner






def mutate_model(model):
    # Create a new model instance
    new_model = ChessNN()
    # Copy weights from the old model
    new_model.load_state_dict(model.state_dict())

    # Choose mutation strategy
    mutation_strategy = random.choice(["original", "single_weight", "probabilistic"])

    magnitude = 10 ** random.randint(-6,0)
    total_weights = sum(p.numel() for p in model.parameters())

    with torch.no_grad():
        if mutation_strategy == "original":
            for param in new_model.parameters():
                # Apply a small random change to each weight
                param.add_(torch.randn(param.size()) * 0.1 * magnitude)

        elif mutation_strategy == "single_weight":
            # Choose a single random weight to mutate
            chosen_param = random.choice([p for p in new_model.parameters() if p.numel() > 0])
            random_idx = tuple(random.randint(0, dim - 1) for dim in chosen_param.shape)
            chosen_param[random_idx] += torch.randn(1) * 0.1 * magnitude

        elif mutation_strategy == "probabilistic":
            for param in new_model.parameters():
                change_probability = 1 / total_weights
                for idx in np.ndindex(param.shape):
                    if random.random() < change_probability:
                        param[idx] += torch.randn(1) * 0.1 * magnitude

    return new_model







def evolve_models(initial_model, generations, number_of_games):
    best_model = initial_model
    best_fitness = -float('inf')

    for gen in range(generations):
        print(f"Generation {gen + 1}/{generations}")

        # Evaluate the current best model
        fitness = evaluate_fitness(best_model, number_of_games)
        print(f"Current best fitness: {fitness}")
        with open("fitness_log.txt", 'a') as file:
            file.write("gen " + str(gen) + " fitness " + str(fitness) + "\n")

        # If the new model is better, update the best model
        if fitness > best_fitness:
            best_fitness = fitness
            save_model(best_model, gen)
        
        # Create a new model by mutating the best model
        best_model = mutate_model(best_model)
    
        # Save the model
        save_model(best_model, gen + 1)

def save_model(model, generation):
    directory = "models"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f"{directory}/model_gen_{generation}.pth"
    torch.save(model.state_dict(), filename)



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

    generations = 100000  # Number of generations to evolve
    number_of_games = 1000    # Number of games played to evaluate fitness against random bot
    evolve_models(model, generations, number_of_games)

