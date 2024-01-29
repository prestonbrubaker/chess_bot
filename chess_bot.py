import chess
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torch.nn.init as init


class ChessNN(nn.Module):
    def __init__(self):
        super(ChessNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)  # Output: 8x16x16
        self.conv2 = nn.Conv2d(8, 2, kernel_size=3, stride=1, padding=1)  # Output: 2x16x16

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


model = ChessNN()

def initialize_population(size):
    return [ChessNN() for _ in range(size)]


MOVE = 0
GAME = 0

def evaluate_position(encoded_state):
    # Convert the encoded state to a PyTorch tensor
    input_tensor = torch.FloatTensor(encoded_state).unsqueeze(0)  # [1, 269] shape for a single encoded state
    #print("Input tensor shape in evaluate_position:", input_tensor.shape)  # [1, 269]

    output = model(input_tensor)
    return output.item()


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

    normalized_scores = [score / total for score in evaluation_scores]

    # Choose an index based on these normalized weights
    #chosen_index = random.choices(range(len(evaluation_scores)), weights=normalized_scores, k=1)[0]
    max_value = max(evaluation_scores)
    max_index = evaluation_scores.index(max_value)
    chosen_index = max_index
    return chosen_index

def get_random_move(board):
    """Returns a random legal move for the current player."""
    return random.choice(list(board.legal_moves))


def evaluate_fitness(model, number_of_games):
    total_score = 0
    wins = 0
    losses = 0

    for _ in range(number_of_games):
        winner, white_score, black_score = play_random_game(model)
        #score_difference = white_score - black_score
        score_difference = 8 - black_score
        total_score += score_difference
        if winner == "White":
            #total_score += score_difference
            wins += 1
        elif winner == "Black":
            #total_score += score_difference
            losses += 1

    fitness = total_score
    #print(fitness)
    win_fraction = wins / (wins + losses) if wins + losses != 0 else 0
    return fitness, win_fraction



def piece_value(piece):
    """Returns the value of a chess piece."""
    if piece is None:
        return 0
    value_dict = {chess.PAWN: 1, chess.KNIGHT: 0, chess.BISHOP: 0, chess.ROOK: 0, chess.QUEEN: 0, chess.KING: 0}
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
def play_random_game(model, max_turns=12):
    global MOVE

    board = chess.Board()
    MOVE = 0

    while not board.is_game_over() and MOVE < max_turns:
        legal_moves = list(board.legal_moves)
        white_score, black_score = calculate_material_score(board)
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
                param.add_(torch.randn(param.size()) * 1 * magnitude)

        elif mutation_strategy == "single_weight":
            # Choose a single random weight to mutate
            chosen_param = random.choice([p for p in new_model.parameters() if p.numel() > 0])
            random_idx = tuple(random.randint(0, dim - 1) for dim in chosen_param.shape)
            random_change = torch.randn(1).item() * 1 * magnitude  # Convert to scalar
            chosen_param[random_idx] += random_change


        elif mutation_strategy == "probabilistic":
            for param in new_model.parameters():
                change_probability = 1 / total_weights
                for idx in np.ndindex(param.shape):
                    if random.random() < change_probability:
                        random_change = torch.randn(1).item() * 1 * magnitude  # Convert to scalar
                        param[idx] += random_change


    return new_model



def select_top_models(population, fitness_scores, top_n):
    sorted_models = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
    return [model for model, _ in sorted_models[:top_n]]

def repopulate(selected_models, total_size):
    new_population = selected_models.copy()
    while len(new_population) < total_size:
        for model in selected_models:
            if(random.uniform(0, 1) < .2):
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
    population_size = 300
    top_n = 200
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
        avg_fitness = np.mean(fitness_scores)

        gen_best_fitness = max(fitness_scores)
        if gen_best_fitness > global_best_fitness:
            global_best_fitness = gen_best_fitness
            global_best_model = population[fitness_scores.index(gen_best_fitness)]
            save_model(global_best_model, f"best_model_gen_{gen + 1}")

        top_models = select_top_models(population, fitness_scores, top_n)
        population = repopulate(top_models, population_size)

        # Print generation info including best fitness ever, standard deviation, average win fraction, and average fitness
        print(f"Generation {gen + 1}, Best Fitness of Generation: {gen_best_fitness}, Global Best Fitness: {global_best_fitness}, Std Dev: {std_dev_fitness}, Avg Win Fraction: {avg_win_fraction}, Avg Fitness: {avg_fitness}")
        with open("fitness_log.txt", 'a') as file:
            file.write(f"Gen {gen + 1}, Best Fitness of Gen: {gen_best_fitness}, Global Best Fitn: {global_best_fitness}, Std Dev: {std_dev_fitness}, Avg Win Frac: {avg_win_fraction}, Avg Fitn: {avg_fitness}\n")


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
    number_of_games = 100
    evolve_models(generations, number_of_games)
