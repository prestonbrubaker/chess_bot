def process_score_data(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        total_score = 0
        turn_count = 0
        game_ended = False

        for line in infile:
            if line.strip():  # Score line
                total_score += int(line.strip())
                turn_count += 1
                game_ended = False  # Resetting the game end flag
            elif not game_ended:  # First empty line of a game end
                game_ended = True  # Set the flag indicating a potential game end
            else:  # Second empty line confirming the end of a game
                if turn_count > 0:
                    # Calculate and write average score per turn
                    average_score_per_turn = total_score / turn_count
                    outfile.write(f"{average_score_per_turn}\n\n")
                # Reset for the next game
                total_score = 0
                turn_count = 0
                game_ended = False  # Reset the game end flag for the next game

def main():
    process_score_data("score_data.txt", "score_data_per_turn.txt")

if __name__ == "__main__":
    main()
