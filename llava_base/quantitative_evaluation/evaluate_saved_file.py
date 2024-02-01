import openai
import os
import argparse
import json
import ast
from multiprocessing.pool import Pool


def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--saved_file", required=True, help="The path to file containing prediction.")
    args = parser.parse_args()
    return args


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()

    # Write combined content to a json file
    with open(args.saved_file, "r") as json_file:
        combined_contents = json.load(json_file)

    # Calculate average score
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    print (len(combined_contents))
    for key, result in combined_contents.items():
        try:
            count += 1
            score_match = result[0]['score']
            score = int(score_match)
            score_sum += score
            if 'pred' in result[0]:
                pred = result[0]['pred']
                if "yes" in pred.lower():
                    yes_count += 1
                elif "no" in pred.lower():
                    no_count += 1
        except:
            print (key)
            continue
    average_score = score_sum / count

    print("Average score for correctness:", average_score)
 
    if no_count != 0:
        accuracy = yes_count / (yes_count + no_count)
        print("Accuracy:", accuracy)


if __name__ == "__main__":
    main()

