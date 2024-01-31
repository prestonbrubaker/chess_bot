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



def get_random_move(board):
    return random.choice(list(board.legal_moves))

def calculate_white_score(board):
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    # Initial counts of black pieces for a standard chess game
    initial_counts = {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 2, chess.QUEEN: 1}

    # Count the black pieces currently on the board
    current_counts = {chess.PAWN: 0, chess.KNIGHT: 0, chess.BISHOP: 0, chess.ROOK: 0, chess.QUEEN: 0}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == chess.BLACK and piece.piece_type in current_counts:
            current_counts[piece.piece_type] += 1

    # Calculate the score based on the pieces white has captured
    score = sum(piece_values[piece_type] * (initial_counts[piece_type] - current_counts[piece_type]) for piece_type in piece_values)
    return score

def main():
    for _ in range(1):  # Play 1 games
        board = chess.Board()
        with open("board_data.txt", "a") as board_file, open("score_data.txt", "a") as score_file:
            while not board.is_game_over():
                encoded_board = encode_board(board)
                two_d_board = create_2d_board(encoded_board)
                for row in two_d_board:
                    board_file.write(row + "\n")
                board_file.write("\n")  # Single line between moves

                chosen_move = get_random_move(board)
                board.push(chosen_move)

                # Write the score to the score file
                white_score = calculate_white_score(board)
                score_file.write(f"{white_score}\n\n")

            board_file.write("\n")  # Additional empty line between games in board file
            score_file.write("\n")  # Additional empty line between games in score file

if __name__ == "__main__":
    main()
