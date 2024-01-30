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


def get_random_move(board):
  """Returns a random legal move for the current player."""
  return random.choice(list(board.legal_moves))

def play_random_game():
    board = chess.Board()

    while not board.is_game_over():
        legal_moves = list(board.legal_moves)
        chosen_move = get_random_move(board)
        board.push(chosen_move)

if __name__ == "__main__":
    play_random_game()
