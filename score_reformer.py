def calculate_average_score_per_game(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        total_score = 0
        turn_count = 0
        for line in infile:
            if line.strip():  # If the line is not empty, it's a score entry
                total_score += int(line.strip())
                turn_count += 1
            else:
                # Check if it's the end of a game
                if infile.readline().strip() == '':
                    if turn_count > 0:
                        average_score = total_score / turn_count
                        outfile.write(f"{average_score}\n\n")  # Write average score
                    # Reset for next game
                    total_score = 0
                    turn_count = 0

def main():
    calculate_average_score_per_game("score_data.txt", "score_data_per_turn.txt")

if __name__ == "__main__":
    main()
