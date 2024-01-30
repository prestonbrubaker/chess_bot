import chess
import random


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
        row = []
        for j in range(8):
            piece_code = encoded_board[i + j]
            row.extend([piece_code[0:2], piece_code[2:4]])
        rows.extend([row, row])
    return rows


def get_random_move(board):
    """Returns a random legal move for the current player."""
    return random.choice(list(board.legal_moves))

def save_board_state_to_file(encoded_board, file):
    with open(file, "a") as f:
        f.write("".join(encoded_board) + "\n\n")

def play_random_game():
    board = chess.Board()
    game_count = 0

    while not board.is_game_over():
        if board.turn:
            print("White's turn")
        else:
            print("Black's turn")

        legal_moves = list(board.legal_moves)
        chosen_move = get_random_move(board)
        board.push(chosen_move)

        encoded_board = encode_board(board)
        create_2d = create_2d_board(encoded_board)

        with open("board_data.txt", "a") as f:
            f.write("Game {}:\n".format(game_count + 1))
            for row in create_2d:
                f.write("".join(row) + "\n")
            f.write("\n")
        
        game_count += 1

if __name__ == "__main__":
    with open("board_data.txt", "w") as f:
        f.write("")  # Clear the file if it exists
    play_random_game()
