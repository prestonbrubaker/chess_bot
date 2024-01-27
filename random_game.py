import chess
import random
import time

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

def get_random_move(board):
    """Returns a random legal move for the current player."""
    return random.choice(list(board.legal_moves))

def play_random_game():
    board = chess.Board()

    while not board.is_game_over():
        print_board(board)
        print("White's turn" if board.turn else "Black's turn")

        legal_moves = list(board.legal_moves)
        print("Possible moves:", ", ".join(map(str, legal_moves)))

        chosen_move = get_random_move(board)
        print("Chosen move:", chosen_move)

        board.push(chosen_move)
        print("Board after move:")
        print_board(board)
        print("\n")

        if board.is_checkmate():
            print("Checkmate!")
        elif board.is_stalemate():
            print("Stalemate!")
        time.sleep(5)

if __name__ == "__main__":
    play_random_game()
