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
        print(encoded_board)
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
    return random.choice(list(board.legal_moves))

def main():
    board = chess.Board()
    with open("board_data.txt", "w") as file:
        while not board.is_game_over():
            encoded_board = encode_board(board)
            two_d_board = create_2d_board(encoded_board)
            for row in two_d_board:
                file.write(" ".join(row) + "\n")
            file.write("\n")

            chosen_move = get_random_move(board)
            board.push(chosen_move)

if __name__ == "__main__":
    main()
