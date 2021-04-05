import argparse
import logging
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO)


def get_sexism_proba(text: str):
    return 0.31


def parse_args(cli_arguments):
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Configure parser for config mode
    parser.add_argument("text", type=str)

    return parser.parse_args(cli_arguments)


if __name__ == "__main__":
    # Parse arguments from CLI
    arguments = parse_args(sys.argv[1:])

    # Make prediction
    start_time = datetime.now()
    sexism_proba = get_sexism_proba(arguments.text)
    end_time = datetime.now()

    time_spent = end_time - start_time

    # Print results
    print(
        f"Probability of racism/sexism in message: {sexism_proba}\nTime spent: {time_spent}"
    )
    logging.info(
        f"\nProbability of racism/sexism in message: {sexism_proba}\nTime spent: {time_spent}"
    )
