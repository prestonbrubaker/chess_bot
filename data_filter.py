def read_average_scores(file_name, threshold):
    with open(file_name, 'r') as file:
        scores = []
        for line in file:
            if line.strip():  # Each score line
                score = float(line.strip())
                if score >= threshold:
                    scores.append(True)  # Mark this game to be included
                else:
                    scores.append(False)  # Mark this game to be excluded
            else:
                if not line.strip() and file.readline().strip() == '':
                    scores.append('GameEnd')  # Mark the end of a game
        return scores

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
    threshold = 0.2
    average_scores = read_average_scores("score_data_per_turn.txt", threshold)

    # Calculate the percentage of games that meet or exceed the threshold
    total_games = average_scores.count('GameEnd')
    passing_games = average_scores.count(True)
    percentage = (passing_games / total_games) * 100 if total_games > 0 else 0
    print(f"Percentage of games meeting the threshold: {percentage:.3f}%")

    reformat_files("board_data.txt", "score_data_per_turn.txt", average_scores,
                   "board_data_reformed.txt", "score_data_per_turn_reformed.txt")


if __name__ == "__main__":
    main()
