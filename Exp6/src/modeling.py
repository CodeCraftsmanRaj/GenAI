import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


class CustomBertClassifier(nn.Module):
    def __init__(
        self,
        model_name,
        num_labels,
        head="linear",
        dropout=0.1,
        hidden_dim=256,
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        hidden = self.bert.config.hidden_size
        self.head_name = head

        if head == "linear":
            self.classifier = nn.Linear(hidden, num_labels)

        elif head == "dropout":
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden, num_labels),
            )

        elif head == "dense_dropout":
            self.classifier = nn.Sequential(
                nn.Linear(hidden, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_labels),
            )

        else:
            raise ValueError(f"Unknown head: {head}")

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        token_type_ids=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        cls = outputs.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)

        logits = self.classifier(cls)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
        }


def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)


def load_baseline_model(model_name, num_labels):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )


def class_weights(labels, num_labels):
    counts = np.bincount(
        np.asarray(labels),
        minlength=num_labels,
    ).astype(float)

    weights = len(labels) / (
        num_labels * np.maximum(counts, 1.0)
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )