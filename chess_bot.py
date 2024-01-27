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

def main():
    board = chess.Board()
    print_board(board)

    while not board.is_game_over():
        print("Enter your move in UCI format (e.g., e2e4):")
        try:
            move_uci = input().strip()
            move = chess.Move.from_uci(move_uci)

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
