import torch.nn as nn
from transformers import AutoConfig, AutoModel


class RoBERTaClassification(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str = "roberta-base",
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        config = AutoConfig.from_pretrained(
            pretrained_model_name, num_labels=num_classes
        )

        self.model = AutoModel.from_pretrained(pretrained_model_name, config=config)
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, attention_mask=None, head_mask=None):
        assert attention_mask is not None, "attention mask is none"

        bert_output = self.model(
            input_ids=features, attention_mask=attention_mask, head_mask=head_mask
        )

        # we only need the hidden state here and don't need
        # transformer output, so index 0
        seq_output = bert_output[0]  # (bs, seq_len, dim)
        # mean pooling, i.e. getting average representation of all tokens
        pooled_output = seq_output.mean(axis=1)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        scores = self.classifier(pooled_output)  # (bs, num_classes)

        return scores


if __name__ == "__main__":
    classifier = RoBERTaClassification()
