import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Get general parameters
with open("config.yml") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

# Set for reproducability
np.random.seed(params["general"]["seed"])
random.seed(params["general"]["seed"])

# Retrieve original dataset
raw_data = load_dataset("tweets_hate_speech_detection")["train"]

# Generate all possible indexes
indexes = list(range(raw_data.num_rows))

# Splitting indexes into train/test
train_indexes, test_indexes = train_test_split(
    indexes, train_size=params["training"]["train_size"]
)
test_indexes, val_indexes = train_test_split(
    test_indexes,
    train_size=params["training"]["test_size"]
    / params["training"]["validation_size"]
    / 2,
)

# Split dataset by indexes
train_data = pd.DataFrame(raw_data[train_indexes])
valid_data = pd.DataFrame(raw_data[val_indexes])
test_data = pd.DataFrame(raw_data[test_indexes])

# Save datasets to path for further training
train_data.to_csv(
    Path(params["data"]["path_to_data"]) / params["data"]["train_filename"], index=False
)
valid_data.to_csv(
    Path(params["data"]["path_to_data"]) / params["data"]["validation_filename"],
    index=False,
)
test_data.to_csv(
    Path(params["data"]["path_to_data"]) / params["data"]["test_filename"], index=False
)
