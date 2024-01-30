def main():
    threshold = 0.2  # Set your threshold
    with open('board_data.txt', 'r') as board_file, open('score_data_per_turn.txt', 'r') as score_file, \
         open('board_data_reformed.txt', 'w') as board_output, open('score_data_per_turn_reformed.txt', 'w') as score_output:

        in_game = False  # To keep track of whether we're inside a game
        game_average = 0.0  # To calculate average score per turn for the current game
        game_turns = 0  # To count the number of turns in the current game

        for board_line, score_line in zip(board_file, score_file):
            if board_line.strip() and score_line.strip():
                in_game = True  # We are inside a game
                game_average += float(score_line.strip())
                game_turns += 1
                board_output.write(board_line)
                score_output.write(score_line)
                score_output.write('\n')  # Add a single empty line between moves
            elif not board_line.strip() and not score_line.strip() and in_game:
                in_game = False  # We have reached the end of a game
                average_score_per_turn = game_average / game_turns
                if average_score_per_turn >= threshold:
                    # Double empty line between games
                    board_output.write('\n\n')
                    score_output.write('\n\n')
                else:
                    # Remove the game data if it doesn't meet the threshold
                    board_output.seek(board_output.tell() - len(board_game))  # Go back to the start of the game data
                    board_output.truncate()  # Remove the game data
                    score_output.seek(score_output.tell() - len(score_game))  # Go back to the start of the game data
                    score_output.truncate()  # Remove the game data
                game_average = 0.0  # Reset for the next game
                game_turns = 0

if __name__ == "__main__":
    main()
