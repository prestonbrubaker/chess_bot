def read_and_filter_games(board_file, score_file, threshold):
    with open(board_file, 'r') as bf, open(score_file, 'r') as sf, \
         open("board_data_reformed.txt", 'w') as b_out, open("score_data_per_turn_reformed.txt", 'w') as s_out:

        board_game_data = []
        score_game_data = []
        total_games = 0
        games_meeting_threshold = 0

        for b_line, s_line in zip(bf, sf):
            if b_line.strip():  # Non-empty line in board file
                board_game_data.append(b_line)
                score_game_data.append(s_line)
            elif not b_line.strip() and not next(bf, '').strip():  # End of a game
                total_games += 1
                average_score = float(score_game_data[-1].strip())
                if average_score >= threshold:
                    games_meeting_threshold += 1
                    for data in board_game_data:
                        b_out.write(data)
                    for data in score_game_data:
                        s_out.write(data)
                    b_out.write("\n")
                    s_out.write("\n")
                # Clear data for the next game
                board_game_data = []
                score_game_data = []

        return games_meeting_threshold, total_games

def main():
    threshold = 0.2
    games_meeting_threshold, total_games = read_and_filter_games("board_data.txt", "score_data_per_turn.txt", threshold)
    percentage = (games_meeting_threshold / total_games) * 100 if total_games > 0 else 0
    print(f"Percentage of games meeting the threshold: {percentage:.3f}%")

if __name__ == "__main__":
    main()
