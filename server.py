import socket
import threading
import chess

class ChessServer:
    def __init__(self, host='73.166.159.150', port=65432):
        self.board = chess.Board()
        self.sockets = []
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(2)
        print(f"Chess Server started on {host}:{port}")

    def handle_client(self, client_socket, player_color):
        while not self.board.is_game_over():
            if self.board.turn == player_color:
                client_socket.sendall(str(self.board).encode())
                move = client_socket.recv(1024).decode().strip()
                try:
                    chess_move = chess.Move.from_uci(move)
                    if chess_move in self.board.legal_moves:
                        self.board.push(chess_move)
                        self.broadcast_board()
                    else:
                        client_socket.sendall(b"Illegal move")
                except:
                    client_socket.sendall(b"Invalid move format")

    def broadcast_board(self):
        for s in self.sockets:
            s.sendall(str(self.board).encode())

    def start(self):
        while len(self.sockets) < 2:
            client_socket, addr = self.server_socket.accept()
            print(f"Connection from {addr} established.")
            self.sockets.append(client_socket)
            player_color = chess.WHITE if len(self.sockets) == 1 else chess.BLACK
            threading.Thread(target=self.handle_client, args=(client_socket, player_color)).start()

if __name__ == "__main__":
    ChessServer().start()
