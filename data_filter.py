def read_average_scores(file_name, threshold):
    # ... [No changes in this function]

def reformat_files(board_file, score_file, scores, board_output, score_output):
    with open(board_file, 'r') as bf, open(score_file, 'r') as sf, \
         open(board_output, 'w') as b_out, open(score_output, 'w') as s_out:

        game_index = 0
        include_game = scores[game_index]  # Determine if the current game is to be included
        game_ended = False

        # Process board data
        for line in bf:
            if line.strip():
                if include_game:
                    b_out.write(line + "\n")
            elif not game_ended:  # First empty line of a game end
                game_ended = True
                if include_game:
                    b_out.write("\n")
            else:  # Second empty line confirming the end of a game
                game_ended = False
                if scores[game_index] == 'GameEnd':
                    game_index += 1
                    include_game = game_index < len(scores) and scores[game_index]

        # Reset game index for score data
        game_index = 0
        include_game = scores[game_index]
        game_ended = False

        # Process score data
        for line in sf:
            if line.strip():
                if include_game:
                    s_out.write(line + "\n")
            elif not game_ended:  # First empty line of a game end
                game_ended = True
                if include_game:
                    s_out.write("\n")
            else:  # Second empty line confirming the end of a game
                game_ended = False
                if scores[game_index] == 'GameEnd':
                    game_index += 1
                    include_game = game_index < len(scores) and scores[game_index]

def main():
    # ... [No changes in this function]

if __name__ == "__main__":
    main()
