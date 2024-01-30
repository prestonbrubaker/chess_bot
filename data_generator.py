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
        row = ''
        for j in range(8):
            piece_code = encoded_board[i + j]
            row += piece_code
        rows.append(row)
        rows.append(row)  # Duplicate the row for the 16x16 grid
    return rows

def calculate_white_score(board):
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    score = 0
    for piece_type in piece_values:
        # Count the difference in pieces from the starting count
        score += piece_values[piece_type] * (board.pieces_count(piece_type, chess.BLACK) - chess.BaseBoard().pieces_count(piece_type, chess.BLACK))
    return score

def main():
    for _ in range(100):  # Play 100 games
        board = chess.Board()
        with open("board_data.txt", "a") as file:
            while not board.is_game_over():
                encoded_board = encode_board(board)
                two_d_board = create_2d_board(encoded_board)
                for row in two_d_board:
                    file.write(row + "\n")
                file.write(" ")  # Single space between moves

                chosen_move = get_random_move(board)
                board.push(chosen_move)

                # Optionally, write the score after each move
                white_score = calculate_white_score(board)
                file.write(f"White's score: {white_score}\n")

            file.write("\n\n")  # Two empty lines between games

if __name__ == "__main__":
    main()
