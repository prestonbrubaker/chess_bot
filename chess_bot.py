import chess
import random
import time

MOVE = 0

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




def get_random_move(board):
    """Returns a random legal move for the current player."""
    return random.choice(list(board.legal_moves))


# White is the neural network bot, and black is the random bot
def play_random_game():
    global MOVE
    board = chess.Board()
    

    while not board.is_game_over():
        print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("~~~~~~~~~~~~~             MOVE " + str(MOVE) +"                 ~~~~~~~~~~~~~~~~")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

        print("\nCURRENT STATE OF THE BOARD\n")
        print_board(board)
        print("\nWHITE'S TURN" if board.turn else "\nBLACK'S TURN")

        legal_moves = list(board.legal_moves)
        print("\nPOSSIBLE MOVES:", ", ".join(map(str, legal_moves)))

        if(board.turn == chess.WHITE):
            is_white_turn = True
            for i in range(len(legal_moves)):
                possible_move = legal_moves[i]
                encoded_state = encode_game_state(board, possible_move, is_white_turn)
                print("\n\nENCODED STATE: " + str(encoded_state))
                chosen_move = get_random_move(board)
        else:
            chosen_move = get_random_move(board)
        print("\nCHOSEN MOVE:", chosen_move)

        

        board.push(chosen_move)
        print("\nBOARD AFTER MOVE:")
        print_board(board)
        print("\n\n\n")

        if board.is_checkmate():
            print("\nCHECKMATE!")
        elif board.is_stalemate():
            print("\nSTALEMATE!")
        
        time.sleep(5)
        MOVE += 1


if __name__ == "__main__":
    play_random_game()
