def read_average_scores(file_name, threshold):
    with open(file_name, 'r') as file:
        scores = []
        current_score = ''
        for line in file:
            if line.strip():
                current_score = float(line.strip())
            else:
                if current_score and current_score >= threshold:
                    scores.append(current_score)
                current_score = ''
        return scores

def reformat_files(board_file, score_file, scores, board_output, score_output):
    with open(board_file, 'r') as bf, open(score_file, 'r') as sf, \
         open(board_output, 'w') as b_out, open(score_output, 'w') as s_out:

        # Trackers for current game index and whether to include current game data
        game_index = 0
        include_game = True

        # Process board data
        for line in bf:
            if line.strip():
                if include_game:
                    b_out.write(line)
            else:
                if bf.readline().strip() == '':
                    if game_index < len(scores):
                        include_game = True
                        b_out.write("\n")
                    else:
                        include_game = False
                    game_index += 1

        # Reset trackers
        game_index = 0
        include_game = True

        # Process score data
        for line in sf:
            if line.strip():
                if include_game:
                    s_out.write(line)
            else:
                if sf.readline().strip() == '':
                    if game_index < len(scores):
                        include_game = True
                        s_out.write("\n")
                    else:
                        include_game = False
                    game_index += 1

def main():
    threshold = 0.2
    average_scores = read_average_scores("score_data_per_turn.txt", threshold)

    # Calculate the percentage of games that meet or exceed the threshold
    total_games = 0
    with open("score_data_per_turn.txt", 'r') as file:
        for line in file:
            if line.strip() == '':
                total_games += 1

    percentage = (len(average_scores) / total_games) * 100 if total_games > 0 else 0
    print(f"Percentage of games meeting the threshold: {percentage:.2f}%")

    reformat_files("board_data.txt", "score_data_per_turn.txt", average_scores,
                   "board_data_reformed.txt", "score_data_per_turn_reformed.txt")

if __name__ == "__main__":
    main()
