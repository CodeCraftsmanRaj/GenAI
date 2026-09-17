import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from .metrics import classification_metrics
from .modeling import class_weights


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx], dtype=torch.long) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


@dataclass
class TrainOutput:
    model: torch.nn.Module
    history: list
    metrics: dict
    predictions: np.ndarray
    labels: np.ndarray
    elapsed_seconds: float


def _logits(model, batch, device):
    inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
    labels = batch["labels"].to(device)
    out = model(**inputs)
    return out["logits"], labels


def _evaluate(model, loader, device, weighted_loss=None):
    model.eval()
    losses, preds, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits, y = _logits(model, batch, device)
            loss = torch.nn.functional.cross_entropy(logits, y, weight=weighted_loss)
            losses.append(float(loss.item()))
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            labels.extend(y.cpu().numpy())
    m = classification_metrics(labels, preds)
    m["loss"] = float(np.mean(losses)) if losses else 0.0
    return m, np.asarray(preds), np.asarray(labels)


def train_model(
    model,
    train_df,
    val_df,
    tokenizer,
    max_length,
    lr,
    batch_size,
    epochs,
    weight_decay,
    device,
    num_labels,
    use_class_weights=False,
    warmup_ratio=0.1,
):
    train_ds = TextDataset(train_df["text"].squeeze().tolist(), train_df["label"].squeeze().tolist(), tokenizer, max_length)
    val_ds = TextDataset(val_df["text"].squeeze().tolist(), val_df["label"].squeeze().tolist(), tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, len(train_loader) * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    weights = None
    if use_class_weights:
        weights = class_weights(train_df["label"].values, num_labels).to(device)

    history = []
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(out["logits"], labels, weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.item()))

        val_metrics, _, _ = _evaluate(model, val_loader, device, weights)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": val_metrics["loss"],
            "validation_accuracy": val_metrics["accuracy"],
            "validation_precision": val_metrics["precision"],
            "validation_recall": val_metrics["recall"],
            "validation_f1": val_metrics["f1"],
            "validation_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{epochs} | train_loss={row['train_loss']:.4f} "
            f"| val_loss={row['validation_loss']:.4f} | val_f1={row['validation_f1']:.4f}"
        )

    elapsed = time.perf_counter() - start
    return TrainOutput(model, history, {}, np.array([]), np.array([]), elapsed)


def evaluate_model(model, df, tokenizer, max_length, batch_size, device):
    ds = TextDataset(df["text"].squeeze().tolist(), df["label"].squeeze().tolist(), tokenizer, max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    metrics, preds, labels = _evaluate(model, loader, device)
    return metrics, preds, labels