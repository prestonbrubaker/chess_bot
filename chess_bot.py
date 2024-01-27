import chess

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




def main():
    board = chess.Board()
    print_board(board)

    while not board.is_game_over():
        print("Enter your move in UCI format (e.g., e2e4):")
        try:
            move_uci = input().strip()
            move = chess.Move.from_uci(move_uci)

            is_white_turn = board.turn == chess.WHITE
            encoded_state = encode_game_state(board, move, is_white_turn)
            print("ENCODED STATE: " + str(encoded_state))

            
            if move not in board.legal_moves:
                print("Illegal move. Try again.")
                continue

            board.push(move)
            print_board(board)

        except ValueError:
            print("Invalid input. Please enter a move in UCI format.")
        
        except KeyboardInterrupt:
            print("\nGame interrupted.")
            break

if __name__ == "__main__":
    main()
