def process_score_data(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        total_score = 0
        turn_count = 0
        for line in infile:
            if line.strip():  # Score line
                total_score += int(line.strip())
                turn_count += 1
            else:
                # Check if the next line is also empty, which indicates the end of a game
                next_line = infile.readline()
                if not next_line.strip():
                    if turn_count > 0:
                        # Calculate average score per turn
                        average_score_per_turn = total_score / turn_count
                        outfile.write(f"{average_score_per_turn}\n\n")  # Write to output file
                    # Reset counters for the next game
                    total_score = 0
                    turn_count = 0

def main():
    process_score_data("score_data.txt", "score_data_per_turn.txt")

if __name__ == "__main__":
    main()
