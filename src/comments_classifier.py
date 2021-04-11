import argparse
import logging
import sys
from datetime import datetime

import yaml
from model import RoBERTaClassification
from scipy.special import softmax
from utils import filter_text, get_project_root

logging.basicConfig(level=logging.INFO)
PROJECT_ROOT = get_project_root()


def load_model(model_path):
    model = RoBERTaClassification.load_from_checkpoint(model_path)
    model.eval()
    return model


def get_hate_proba(params, text: str):
    text = filter_text(text)

    logging.info("Model loading...")
    model = load_model(PROJECT_ROOT / params["model"]["model_path"])

    prediction = softmax(model(**model.encode(text)).tolist()[0])
    print(prediction)
    hate_probability = round(prediction[1], 2)

    return hate_probability


def parse_args(cli_arguments):
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Configure parser for config mode
    parser.add_argument("text", type=str)

    return parser.parse_args(cli_arguments)


if __name__ == "__main__":
    # Load general parameters
    with open(PROJECT_ROOT / "config.yml", "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    # Parse arguments from CLI
    arguments = parse_args(sys.argv[1:])
    logging.info(f"Text received: {arguments.text}")

    # Make prediction
    start_time = datetime.now()
    hate_proba = get_hate_proba(params, arguments.text)
    end_time = datetime.now()

    time_spent = end_time - start_time

    # Print results
    logging.info(f"Probability of hate speech in message: {hate_proba}")
    logging.info(f"Time spent: {time_spent}")
