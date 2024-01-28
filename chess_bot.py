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

def initialize_population(size):
    return [ChessNN() for _ in range(size)]


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
    total_score = 0
    wins = 0
    losses = 0

    for _ in range(number_of_games):
        winner, white_score, black_score = play_random_game(model)
        score_difference = white_score - black_score
        if winner == "White":
            total_score += score_difference + 20
            wins += 1
        elif winner == "Black":
            total_score += score_difference
            losses += 1

    fitness = total_score / number_of_games
    win_fraction = wins / (wins + losses) if wins + losses > 0 else 0
    return fitness, win_fraction



def piece_value(piece):
    """Returns the value of a chess piece."""
    if piece is None:
        return 0
    value_dict = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    return value_dict[piece.piece_type]

def calculate_material_score(board):
    """Calculates and returns the material score for White and Black."""
    white_score = 0
    black_score = 0
    for i in range(64):
        piece = board.piece_at(i)
        if piece is not None:
            if piece.color == chess.WHITE:
                white_score += piece_value(piece)
            else:
                black_score += piece_value(piece)
    return white_score, black_score


# White is the neural network bot, and black is the random bot
def play_random_game(model, max_turns=10):
    global MOVE

    board = chess.Board()
    MOVE = 0

    while not board.is_game_over() and MOVE < max_turns:
        legal_moves = list(board.legal_moves)
        
        if board.turn == chess.WHITE:
            is_white_turn = True
            evaluation_scores = []
            for possible_move in legal_moves:
                encoded_state = encode_game_state(board, possible_move, is_white_turn)
                evaluation = evaluate_position(encoded_state)
                evaluation_scores.append(evaluation)
            
            chosen_index = choose_index_by_evaluation(evaluation_scores)
            chosen_move = legal_moves[chosen_index]
        else:
            chosen_move = get_random_move(board)

        board.push(chosen_move)
        MOVE += 1

    white_score, black_score = calculate_material_score(board)

    if MOVE >= max_turns:
        winner = "Draw"
    elif board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
    elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        winner = "Draw"
    else:
        winner = "Game not finished"
    return winner, white_score, black_score







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
            random_change = torch.randn(1).item() * 0.1 * magnitude  # Convert to scalar
            chosen_param[random_idx] += random_change


        elif mutation_strategy == "probabilistic":
            for param in new_model.parameters():
                change_probability = 1 / total_weights
                for idx in np.ndindex(param.shape):
                    if random.random() < change_probability:
                        random_change = torch.randn(1).item() * 0.1 * magnitude  # Convert to scalar
                        param[idx] += random_change


    return new_model



def select_top_models(population, fitness_scores, top_n):
    sorted_models = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
    return [model for model, _ in sorted_models[:top_n]]

def repopulate(selected_models, total_size):
    new_population = selected_models.copy()
    while len(new_population) < total_size:
        for model in selected_models:
            mutated_model = mutate_model(model)
            new_population.append(mutated_model)
            if len(new_population) >= total_size:
                break
    return new_population





# Global variables to track the best model and its fitness
global_best_model = None
global_best_fitness = -float('inf')

def evolve_models(generations, number_of_games):
    global global_best_model, global_best_fitness
    population_size = 100
    top_n = 10
    population = initialize_population(population_size)

    for gen in range(generations):
        fitness_scores = []
        win_fractions = []
        for model in population:
            fitness, win_fraction = evaluate_fitness(model, number_of_games)
            fitness_scores.append(fitness)
            win_fractions.append(win_fraction)

        std_dev_fitness = np.std(fitness_scores)
        avg_win_fraction = np.mean(win_fractions)

        gen_best_fitness = max(fitness_scores)
        if gen_best_fitness > global_best_fitness:
            global_best_fitness = gen_best_fitness
            global_best_model = population[fitness_scores.index(gen_best_fitness)]
            save_model(global_best_model, f"best_model_gen_{gen + 1}")

        top_models = select_top_models(population, fitness_scores, top_n)
        population = repopulate(top_models, population_size)

        print(f"Generation {gen + 1}/{generations}, Best Fitness of Generation: {gen_best_fitness}, Global Best Fitness: {global_best_fitness}, Std Dev: {std_dev_fitness}, Avg Win Fraction: {avg_win_fraction}")
        with open("fitness_log.txt", 'a') as file:
            file.write(f"Generation {gen + 1}, Best Fitness of Generation: {gen_best_fitness}, Global Best Fitness: {global_best_fitness}, Std Dev: {std_dev_fitness}, Avg Win Fraction: {avg_win_fraction}\n")



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
        print("Starting with the loaded model.")
    else:
        print("No existing model found in the 'models' directory. Starting with a new model.")
        model = ChessNN()

    generations = 100000
    number_of_games = 10
    evolve_models(generations, number_of_games)
