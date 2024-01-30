def main():
    threshold = 0.1
    with open('board_data.txt', 'r') as board_file, \
         open('score_data_per_turn.txt', 'r') as score_file, \
         open('board_data_reformed.txt', 'w') as board_output, \
         open('score_data_per_turn_reformed.txt', 'w') as score_output:

        board_game, score_game = [], []
        total_score, move_count = 0, 0
        game_above_threshold = False

        while True:
            board_line = board_file.readline()
            score_line = score_file.readline()

            # Check if end of file
            if not board_line or not score_line:
                break

            if board_line.strip() and score_line.strip():
                # Accumulate board states and scores for a game
                board_game.append(board_line)
                score_game.append(score_line)
                total_score += float(score_line.strip())
                move_count += 1
            elif board_line.strip() == '' and score_line.strip() == '':
                # End of a game
                if move_count > 0:
                    average_score = total_score / move_count
                    game_above_threshold = average_score >= threshold

                if game_above_threshold:
                    # Write the game data to output files
                    for b_line, s_line in zip(board_game, score_game):
                        board_output.write(b_line)
                        score_output.write(s_line)
                    board_output.write("\n")  # Two empty lines between games
                    score_output.write("\n")
                
                # Reset for the next game
                board_game, score_game = [], []
                total_score, move_count = 0, 0
                game_above_threshold = False

            # To handle the extra empty line between games
            elif board_line.strip() == '' and move_count == 0:
                board_output.write("\n")
                score_output.write("\n")

if __name__ == "__main__":
    main()
