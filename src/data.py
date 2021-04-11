from pathlib import Path
from typing import List, Mapping, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from utils import filter_text, set_global_seed


class TextClassificationDataset(Dataset):
    """
    Wrapper around Torch Dataset to perform text classification
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[str] = None,
        label_dict: Mapping[str, int] = None,
        max_seq_length: int = 512,
        model_name: str = "roberta-base",
    ):
        self.texts = list(map(filter_text, texts))
        self.labels = labels
        self.label_dict = label_dict
        self.max_seq_length = max_seq_length

        if self.label_dict is None and labels is not None:
            self.label_dict = dict(zip(sorted(set(labels)), range(len(set(labels)))))

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.texts)

    def __getitem__(self, index) -> Mapping[str, torch.Tensor]:
        """Gets element of the dataset

        Args:
            index (int): index of the element in the dataset
        Returns:
            Single element by index
        """

        # encoding the text
        x = self.texts[index]

        # a dictionary with `input_ids` and `attention_mask` as keys
        output_dict = self.tokenizer.encode_plus(
            x,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt",
            truncation=True,
            return_attention_mask=True,
        )

        # for Catalyst, there needs to be a key called features
        output_dict["features"] = output_dict["input_ids"].squeeze(0)
        del output_dict["input_ids"]

        # encoding target
        if self.labels is not None:
            y = self.labels[index]
            y_encoded = torch.Tensor([self.label_dict.get(y, -1)]).long().squeeze(0)
            output_dict["targets"] = y_encoded

        return output_dict


def read_data(params: dict) -> Tuple[dict, dict]:
    """
    A custom function that reads data from CSV files, creates PyTorch datasets and
    data loaders. The output is provided to be easily used with Catalyst

    :param params: a dictionary read from the config.yml file
    :return: a tuple with 2 dictionaries
    """
    # reading CSV files to Pandas dataframes
    train_df = pd.read_csv(
        Path(params["data"]["path_to_data"]) / params["data"]["train_filename"]
    )
    valid_df = pd.read_csv(
        Path(params["data"]["path_to_data"]) / params["data"]["validation_filename"]
    )
    test_df = pd.read_csv(
        Path(params["data"]["path_to_data"]) / params["data"]["test_filename"]
    )

    # creating PyTorch Datasets
    train_dataset = TextClassificationDataset(
        texts=train_df[params["data"]["text_field_name"]].values.tolist(),
        labels=train_df[params["data"]["label_field_name"]].values,
        max_seq_length=params["model"]["max_seq_length"],
        model_name=params["model"]["model_name"],
    )

    valid_dataset = TextClassificationDataset(
        texts=valid_df[params["data"]["text_field_name"]].values.tolist(),
        labels=valid_df[params["data"]["label_field_name"]].values,
        max_seq_length=params["model"]["max_seq_length"],
        model_name=params["model"]["model_name"],
    )

    test_dataset = TextClassificationDataset(
        texts=test_df[params["data"]["text_field_name"]].values.tolist(),
        labels=test_df[params["data"]["label_field_name"]].values,
        max_seq_length=params["model"]["max_seq_length"],
        model_name=params["model"]["model_name"],
    )

    set_global_seed(params["general"]["seed"])

    # creating PyTorch data loaders and placing them in dictionaries (for Catalyst)
    train_val_loaders = {
        "train": DataLoader(
            dataset=train_dataset,
            batch_size=params["training"]["batch_size"],
            shuffle=True,
        ),
        "valid": DataLoader(
            dataset=valid_dataset,
            batch_size=params["training"]["batch_size"],
            shuffle=False,
        ),
    }

    test_loaders = {
        "test": DataLoader(
            dataset=test_dataset,
            batch_size=params["training"]["batch_size"],
            shuffle=False,
        )
    }

    return train_val_loaders, test_loaders


def generate_datasets(params: dict):
    # Set reptoducability
    set_global_seed(params["general"]["seed"])

    # Retrieve original dataset
    raw_data = load_dataset("tweets_hate_speech_detection")["train"]

    # Split postitive / negative indexes
    positive_idxs = np.where(np.array(raw_data["label"]) == 1)[0]
    negative_idxs = np.where(np.array(raw_data["label"]) == 0)[0]

    if params["data"]["is_balanced"]:
        min_idxs = min(len(positive_idxs), len(negative_idxs))
        positive_idxs = np.random.choice(positive_idxs, min_idxs, replace=False)
        negative_idxs = np.random.choice(negative_idxs, min_idxs, replace=False)

    valid_idxs = np.concatenate([positive_idxs, negative_idxs])

    # Splitting indexes into train/test
    train_indexes, test_indexes = train_test_split(
        valid_idxs, train_size=params["training"]["train_size"]
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

    # Find path to right directory
    if params["data"]["is_balanced"]:
        data_dir_path = params["data"]["path_to_balanced_data"]
    else:
        data_dir_path = params["data"]["path_to_imbalanced_data"]

    # Save datasets to path for further training
    train_data.to_csv(
        Path(data_dir_path) / params["data"]["train_filename"],
        index=False,
    )
    valid_data.to_csv(
        Path(data_dir_path) / params["data"]["validation_filename"],
        index=False,
    )
    test_data.to_csv(
        Path(data_dir_path) / params["data"]["test_filename"],
        index=False,
    )
