def read_average_scores(file_name, threshold):
    with open(file_name, 'r') as file:
        scores = []
        include_game = False
        for line in file:
            if line.strip():  # Each score line
                score = float(line.strip())
                include_game = score >= threshold
            else:
                # At the end of a game
                scores.append(include_game)
        return scores

def reformat_files(board_file, score_file, scores, board_output, score_output):
    with open(board_file, 'r') as bf, open(score_file, 'r') as sf, \
         open(board_output, 'w') as b_out, open(score_output, 'w') as s_out:

        game_index = 0
        for b_line, s_line in zip(bf, sf):
            if b_line.strip() and s_line.strip():
                if scores[game_index]:
                    b_out.write(b_line)
                    s_out.write(s_line)
            else:
                if scores[game_index]:
                    b_out.write("\n")
                    s_out.write("\n")
                if not b_line.strip() and not bf.readline().strip():
                    game_index += 1
                    if game_index < len(scores) and scores[game_index]:
                        b_out.write("\n")
                        s_out.write("\n")

def main():
    threshold = 0.2
    average_scores = read_average_scores("score_data_per_turn.txt", threshold)

    # Calculate the percentage of games that meet or exceed the threshold
    passing_games = sum(1 for score in average_scores if score)
    total_games = len(average_scores)
    percentage = (passing_games / total_games) * 100 if total_games > 0 else 0
    print(f"Percentage of games meeting the threshold: {percentage:.3f}%")

    reformat_files("board_data.txt", "score_data_per_turn.txt", average_scores,
                   "board_data_reformed.txt", "score_data_per_turn_reformed.txt")

if __name__ == "__main__":
    main()
