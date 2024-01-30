def process_files(board_file, score_file, threshold, board_output, score_output):
    with open(board_file, 'r') as bf, open(score_file, 'r') as sf, \
         open(board_output, 'w') as b_out, open(score_output, 'w') as s_out:

        board_game_data, score_game_data = [], []
        total_score, turn_count = 0, 0
        write_game = False

        while True:
            b_line = bf.readline()
            s_line = sf.readline()

            if not b_line or not s_line:  # End of file
                break

            if b_line.strip() and s_line.strip():  # Non-empty lines (moves and scores)
                board_game_data.append(b_line)
                score_game_data.append(s_line)
                total_score += float(s_line.strip())
                turn_count += 1
            elif b_line.strip() == '' and s_line.strip() == '':  # End of a game
                if turn_count > 0 and (total_score / turn_count) >= threshold:
                    write_game = True
                if write_game:
                    for bd, sd in zip(board_game_data, score_game_data):
                        b_out.write(bd + "\n")
                        s_out.write(sd + "\n")
                    b_out.write("\n")  # Extra line for game separation
                    s_out.write("\n")  # Extra line for game separation
                # Reset for next game
                board_game_data, score_game_data = [], []
                total_score, turn_count = 0, 0
                write_game = False

def main():
    threshold = 0.14  # Set your threshold
    process_files("board_data.txt", "score_data_per_turn.txt", threshold,
                  "board_data_reformed.txt", "score_data_per_turn_reformed.txt")

if __name__ == "__main__":
    main()
