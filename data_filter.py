def main():
    threshold = 0.1 # Set your threshold
    with open('board_data.txt', 'r') as board_file, open('score_data_per_turn.txt', 'r') as score_file, \
         open('board_data_reformed.txt', 'w') as board_output, open('score_data_per_turn_reformed.txt', 'w') as score_output:

        board_game, score_game = [], []
        total_score, turn_count = 0, 0
        for board_line, score_line in zip(board_file, score_file):
            if board_line.strip() and score_line.strip():
                # Collect data for a game
                board_game.append(board_line)
                score_game.append(score_line)
                total_score += float(score_line.strip())
                turn_count += 1
            elif not board_line.strip() and not score_line.strip() and turn_count > 0:
                # End of a game, check if it meets the threshold
                average_score = total_score / turn_count
                if average_score >= threshold:
                    # Write game data to the output files
                    for line in board_game:
                        board_output.write(line)
                    board_output.write("\n\n")  # Double empty line between games
                    for line in score_game:
                        score_output.write(line)
                    score_output.write("\n\n")  # Double empty line between games
                # Reset for the next game
                board_game, score_game = [], []
                total_score, turn_count = 0, 0

if __name__ == "__main__":
    main()
