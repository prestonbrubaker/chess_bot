import chess
import random

def encode_board(board):
    piece_mapping = {
        None: ['00', '00'],  # Empty square
        chess.Piece(chess.PAWN, chess.WHITE): ['00', '01'],
        chess.Piece(chess.ROOK, chess.WHITE): ['00', '10'],
        chess.Piece(chess.KNIGHT, chess.WHITE): ['00', '11'],
        chess.Piece(chess.BISHOP, chess.WHITE): ['01', '00'],
        chess.Piece(chess.QUEEN, chess.WHITE): ['01', '01'],
        chess.Piece(chess.KING, chess.WHITE): ['01', '10'],
        chess.Piece(chess.PAWN, chess.BLACK): ['10', '01'],
        chess.Piece(chess.ROOK, chess.BLACK): ['10', '10'],
        chess.Piece(chess.KNIGHT, chess.BLACK): ['10', '11'],
        chess.Piece(chess.BISHOP, chess.BLACK): ['11', '00'],
        chess.Piece(chess.QUEEN, chess.BLACK): ['11', '01'],
        chess.Piece(chess.KING, chess.BLACK): ['11', '10'],
    }

    encoded_board = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        encoded_piece = piece_mapping.get(piece, ['11', '11'])  # Default to reserved
        encoded_board.append(encoded_piece)

    return encoded_board

def create_2d_board(encoded_board):
    rows = []
    for i in range(0, len(encoded_board), 8):
        upper_row = ''
        lower_row = ''
        for j in range(8):
            piece_code = encoded_board[i + j]
            upper_row += piece_code[0]
            lower_row += piece_code[1]
        rows.append(upper_row)
        rows.append(lower_row)
    return rows

def get_random_move(board):
    return random.choice(list(board.legal_moves))

def calculate_white_score(board):
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    initial_counts = {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 2, chess.QUEEN: 1}
    current_counts = {chess.PAWN: 0, chess.KNIGHT: 0, chess.BISHOP: 0, chess.ROOK: 0, chess.QUEEN: 0}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == chess.BLACK and piece.piece_type in current_counts:
            current_counts[piece.piece_type] += 1
    score = sum(piece_values[piece_type] * (initial_counts[piece_type] - current_counts[piece_type]) for piece_type in piece_values)
    return score

def main():
    for _ in range(1000):  # Play 1000 games
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

                white_score = calculate_white_score(board)
                score_file.write(f"{white_score}\n\n")

            board_file.write("\n")  # Additional empty line between games
            score_file.write("\n")  # Additional empty line between games

if __name__ == "__main__":
    main()
