import re
from collections import Counter

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


TEXT_CANDIDATES = ["text", "sms", "message", "sentence", "review", "content"]
LABEL_CANDIDATES = ["label", "labels", "target", "class", "category"]


def _pick_column(columns, candidates, explicit=None):
    if explicit:
        if explicit not in columns:
            raise ValueError(f"Configured column '{explicit}' not found. Available: {columns}")
        return explicit
    lower = {c.lower(): c for c in columns}
    for name in candidates:
        if name in lower:
            return lower[name]
    raise ValueError(f"Could not infer column from {columns}. Set it explicitly in config.yaml.")


def load_sms_dataset(cfg):
    name = cfg["data"]["hf_dataset"]
    try:
        raw = load_dataset(name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download Hugging Face dataset '{name}'. "
            "Check internet/Hugging Face access and the dataset name in config.yaml."
        ) from exc

    # Accept DatasetDict, Dataset, or datasets with a predefined train split.
    if hasattr(raw, "keys"):
        if "train" in raw:
            df = raw["train"].to_pandas()
        else:
            first = next(iter(raw.keys()))
            df = raw[first].to_pandas()
    else:
        df = raw.to_pandas()

    text_col = _pick_column(df.columns, TEXT_CANDIDATES, cfg["data"].get("text_column"))
    label_col = _pick_column(df.columns, LABEL_CANDIDATES, cfg["data"].get("label_column"))

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df["text"] = df["text"].astype(str)
    df = df.dropna(subset=["text", "label"]).copy()
    return df.reset_index(drop=True)


def encode_labels(df):
    # Make labels deterministic 0..N-1 while retaining original names.
    labels = list(pd.unique(df["label"]))
    if set(labels) == {0, 1} or set(labels) == {"0", "1"}:
        mapping = {labels[0]: int(labels[0])} if False else None
    else:
        # Common SMS dataset: ham/spam -> 0/1.
        lower_map = {str(x).lower(): x for x in labels}
        if "ham" in lower_map and "spam" in lower_map:
            mapping = {lower_map["ham"]: 0, lower_map["spam"]: 1}
        else:
            mapping = {old: i for i, old in enumerate(sorted(labels, key=str))}
    if mapping is None:
        mapping = {0: 0, 1: 1, "0": 0, "1": 1}
    df = df.copy()
    df["label"] = df["label"].map(mapping).astype(int)
    return df, {str(k): int(v) for k, v in mapping.items()}


def minimal_clean(text):
    # BERT should receive the original language signal; only remove accidental whitespace.
    return re.sub(r"\s+", " ", str(text)).strip()


def normalized_clean(text):
    text = minimal_clean(text)
    text = re.sub(r"http\S+|www\.\S+", " URL ", text, flags=re.I)
    text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", " EMAIL ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_versions(df):
    out = df.copy()
    out["text_minimal"] = out["text"].map(minimal_clean)
    out["text_normalized"] = out["text"].map(normalized_clean)
    return out


def split_data(df, seed, test_size, validation_size):
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )
    val_fraction_of_train_val = validation_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction_of_train_val,
        random_state=seed,
        stratify=train_val["label"],
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def dataset_report(df):
    return {
        "records": int(len(df)),
        "missing_text": int(df["text"].isna().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_distribution": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "average_chars": float(df["text"].str.len().mean()),
    }