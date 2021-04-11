from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from utils import filter_text

from data import TextClassificationDataset


class RoBERTaClassification(pl.LightningModule):
    def __init__(
        self,
        hyper_params,
        pretrained_model_name: str = "roberta-base",
        num_classes: int = 2,
        max_seq_length: int = 512,
    ):
        super().__init__()
        self.params = hyper_params

        config = AutoConfig.from_pretrained(
            pretrained_model_name, num_labels=num_classes
        )
        self.max_seq_length = max_seq_length

        self.model = AutoModel.from_pretrained(pretrained_model_name, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

        for param in self.model.parameters():
            param.requires_grad = False

    def encode(self, text: str) -> dict:
        data = self.tokenizer.encode_plus(
            filter_text(text),
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt",
            truncation=True,
            return_attention_mask=True,
        ).data

        data["features"] = data["input_ids"]
        del data["input_ids"]

        return data

    def decode(self, predictions: Tensor):
        return predictions

    def loss_function(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def forward(self, features, attention_mask=None, head_mask=None):
        # Get hidden states of RoBERTa model
        roberta_output = self.model(
            input_ids=features, attention_mask=attention_mask, head_mask=head_mask
        )
        seq_output = roberta_output[0]  # (bs, seq_len, dim)

        pooled_output = seq_output.mean(axis=1)  # (bs, dim)
        predictions = self.classifier(pooled_output)  # (bs, num_classes)

        return self.decode(predictions)

    def training_step(self, batch, batch_idx):
        features = batch["features"]
        attention_mask = batch["attention_mask"]
        y = batch["targets"]

        y_hat = self(features, attention_mask)
        return {"loss": self.loss_function(y_hat, y)}

    def validation_step(self, batch, batch_idx):
        features = batch["features"]
        attention_mask = batch["attention_mask"]
        y = batch["targets"]

        y_hat = self(features, attention_mask)
        return {"val_loss": self.loss_function(y_hat, y)}

    def test_step(self, batch, batch_idx):
        features = batch["features"]
        attention_mask = batch["attention_mask"]
        y = batch["targets"]

        y_hat = self(features, attention_mask)
        return {"test_loss": self.loss_function(y_hat, y)}

    def get_dataset(self, filename):
        # Find path to right directory
        if self.params["data"]["is_balanced"]:
            data_dir_path = self.params["data"]["path_to_balanced_data"]
        else:
            data_dir_path = self.params["data"]["path_to_imbalanced_data"]

        df = pd.read_csv(Path(data_dir_path) / filename)
        dataset = TextClassificationDataset(
            texts=df[self.params["data"]["text_field_name"]].values.tolist(),
            labels=df[self.params["data"]["label_field_name"]].values,
            max_seq_length=self.params["model"]["max_seq_length"],
            model_name=self.params["model"]["model_name"],
        )

        return dataset

    def train_dataloader(self):
        train_dataset = self.get_dataset(self.params["data"]["train_filename"])

        return DataLoader(
            dataset=train_dataset,
            batch_size=self.params["training"]["batch_size"],
            shuffle=True,
            num_workers=6,
        )

    def val_dataloader(self):
        val_dataset = self.get_dataset(self.params["data"]["validation_filename"])

        return DataLoader(
            dataset=val_dataset,
            batch_size=self.params["training"]["batch_size"],
            shuffle=False,
            num_workers=6,
        )

    def test_dataloader(self):
        test_dataset = self.get_dataset(self.params["data"]["test_filename"])

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.params["training"]["batch_size"],
            shuffle=False,
            num_workers=6,
        )

        return test_loader

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=float(self.params["training"]["learn_rate"])
        )
