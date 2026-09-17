from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from .data import (
    dataset_report,
    encode_labels,
    load_sms_dataset,
    prepare_versions,
    split_data,
)
from .experiments import run_experiment
from .modeling import load_tokenizer
from .plots import generate_plots
from .utils import device_report, ensure_dirs, load_config, save_json, set_seed


def tokenization_analysis(cfg, sample_texts):
    tokenizer = load_tokenizer(cfg["tokenization"]["model_name"])
    report = {}
    for length in cfg["tokenization"]["sequence_lengths"]:
        items = []
        for text in sample_texts[:5]:
            enc = tokenizer(text, padding="max_length", truncation=True, max_length=length)
            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
            items.append({
                "text": text,
                "tokens": tokens,
                "token_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "non_padding_tokens": int(sum(enc["attention_mask"])),
            })
        report[str(length)] = items
    return report


def _subset(df, n):
    return df if n is None else df.head(int(n)).copy()


def run_all(config_path, quick=False):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    if quick:
        cfg["data"]["max_train_samples"] = 500
        cfg["data"]["max_eval_samples"] = 150
        cfg["data"]["max_test_samples"] = 150
        cfg["training"]["baseline"]["epochs"] = 1
        cfg["training"]["selected_hyperparameters"] = [
            {"learning_rate": 2e-5, "batch_size": 8, "epochs": 1, "weight_decay": 0.01}
        ]
        cfg["experiments"]["run_preprocessing_comparison"] = False
        cfg["experiments"]["run_sequence_length_comparison"] = False
        cfg["experiments"]["run_hyperparameter_comparison"] = False
        cfg["experiments"]["run_classifier_heads"] = False
        cfg["experiments"]["run_imbalance_comparison"] = False

    out = Path(cfg["project"]["output_dir"])
    ensure_dirs(out, cfg["project"]["model_dir"])

    print("Device:", device_report())
    df = load_sms_dataset(cfg)
    before = dataset_report(df)
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    df, label_mapping = encode_labels(df)
    df = prepare_versions(df)

    save_json(
        {"before_cleaning": before, "after_cleaning": dataset_report(df), "label_mapping": label_mapping},
        out / "dataset_report.json",
    )

    train, val, test = split_data(
        df[["text", "label", "text_minimal", "text_normalized"]],
        cfg["seed"],
        cfg["data"]["test_size"],
        cfg["data"]["validation_size"],
    )
    train = _subset(train, cfg["data"].get("max_train_samples"))
    val = _subset(val, cfg["data"].get("max_eval_samples"))
    test = _subset(test, cfg["data"].get("max_test_samples"))

    splits = {"train": len(train), "validation": len(val), "test": len(test)}
    save_json(splits, out / "split_sizes.json")

    tok_report = tokenization_analysis(cfg, train["text"].tolist())
    save_json(tok_report, out / "tokenization_report.json")

    base = cfg["training"]["baseline"]
    results = []

    # E1: baseline BERT, minimal preprocessing, 32 tokens.
    r, _, _ = run_experiment(
        "E1_baseline_bert",
        train[["text_minimal", "label"]].rename(columns={"text_minimal": "text"}),
        val[["text_minimal", "label"]].rename(columns={"text_minimal": "text"}),
        test[["text_minimal", "label"]].rename(columns={"text_minimal": "text"}),
        cfg, "cuda" if torch.cuda.is_available() else "cpu", 32,
        base["learning_rate"], base["batch_size"], base["epochs"], base["weight_decay"],
        head="baseline",
    )
    results.append(r)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg["experiments"].get("run_preprocessing_comparison"):
        r, _, _ = run_experiment(
            "E2_preprocessing_normalized", train[["text_normalized", "label"]].rename(columns={"text_normalized": "text"}),
            val[["text_normalized", "label"]].rename(columns={"text_normalized": "text"}),
            test[["text_normalized", "label"]].rename(columns={"text_normalized": "text"}),
            cfg, device, 32, base["learning_rate"], base["batch_size"], base["epochs"], base["weight_decay"],
            head="baseline",
        )
        results.append(r)

    if cfg["experiments"].get("run_sequence_length_comparison"):
        for length in [128]:
            r, _, _ = run_experiment(
                f"E3_sequence_{length}", train[["text_minimal", "label"]].rename(columns={"text_minimal": "text"}),
                val[["text_minimal", "label"]].rename(columns={"text_minimal": "text"}),
                test[["text_minimal", "label"]].rename(columns={"text_minimal": "text"}),
                cfg, device, length, base["learning_rate"], base["batch_size"], base["epochs"], base["weight_decay"],
                head="baseline",
            )
            results.append(r)

    if cfg["experiments"].get("run_hyperparameter_comparison"):
        for i, hp in enumerate(cfg["training"]["selected_hyperparameters"], start=1):
            r, _, _ = run_experiment(
                f"E4_hyperparam_{i}", train[["text_minimal", "label"]].rename(columns={"text_minimal": "text"}),
                val[["text_minimal", "label"]].rename(columns={"text_minimal": "text"}),
                test[["text_minimal", "label"]].rename(columns={"text_minimal": "text"}),
                cfg, device, 32, hp["learning_rate"], hp["batch_size"], hp["epochs"], hp["weight_decay"],
                head="baseline",
            )
            results.append(r)

    if cfg["experiments"].get("run_classifier_heads"):
        for head in ["linear", "dropout", "dense_dropout"]:
            r, _, _ = run_experiment(
                f"E5_head_{head}", train[["text_minimal", "label"]].rename(columns={"text_minimal": "text"}),
                val[["text_minimal", "label"]].rename(columns={"text_minimal": "text"}),
                test[["text_minimal", "label"]].rename(columns={"text_minimal": "text"}),
                cfg, device, 32, base["learning_rate"], base["batch_size"], base["epochs"], base["weight_decay"],
                head=head,
            )
            results.append(r)

    if cfg["experiments"].get("run_imbalance_comparison"):
        # Standard loss vs class-weighted loss under identical conditions.
        r, _, _ = run_experiment(
            "E6_imbalance_standard", train[["text_minimal", "label"]].rename(columns={"text_minimal": "text"}),
            val[["text_minimal", "label"]].rename(columns={"text_minimal": "text"}),
            test[["text_minimal", "label"]].rename(columns={"text_minimal": "text"}),
            cfg, device, 32, base["learning_rate"], base["batch_size"], base["epochs"], base["weight_decay"],
            head="linear", use_class_weights=False,
        )
        results.append(r)
        r, _, _ = run_experiment(
            "E6_imbalance_weighted", train[["text_minimal", "label"]].rename(columns={"text_minimal": "text"}),
            val[["text_minimal", "label"]].rename(columns={"text_minimal": "text"}),
            test[["text_minimal", "label"]].rename(columns={"text_minimal": "text"}),
            cfg, device, 32, base["learning_rate"], base["batch_size"], base["epochs"], base["weight_decay"],
            head="linear", use_class_weights=True,
        )
        results.append(r)

    if cfg["experiments"].get("run_robustness"):
        # Use the final saved linear-head model if present; otherwise train a compact final model.
        final = next((x for x in reversed(results) if x["head"] in ["linear", "baseline"]), results[-1])
        model_name = cfg["tokenization"]["model_name"]
        tokenizer = load_tokenizer(model_name)
        from .modeling import CustomBertClassifier
        model = CustomBertClassifier(
            model_name, 2, head="linear",
            dropout=cfg["training"]["classifier"]["dropout"],
            hidden_dim=cfg["training"]["classifier"]["hidden_dim"],
        )
        model_path = Path(cfg["project"]["model_dir"]) / final["experiment"] / "pytorch_model.bin"
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            # This branch is unlikely during normal execution.
            raise FileNotFoundError(f"Final model not found: {model_path}")
        model.to(device).eval()

        from .trainer import TextDataset
        from torch.utils.data import DataLoader
        examples = pd.DataFrame(cfg["robustness"]["examples"])
        ds = TextDataset(examples["text"], examples["label"], tokenizer, 128)
        loader = DataLoader(ds, batch_size=16, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                preds.extend(torch.argmax(model(**inputs)["logits"], dim=-1).cpu().tolist())
        examples["prediction"] = preds
        examples["correct"] = examples["prediction"] == examples["label"]
        save_json(examples.to_dict(orient="records"), out / "robustness.json")

    summary_rows = []
    for r in results:
        summary_rows.append({
            "experiment": r["experiment"],
            "sequence_length": r["sequence_length"],
            "learning_rate": r["learning_rate"],
            "batch_size": r["batch_size"],
            "epochs": r["epochs"],
            "weight_decay": r["weight_decay"],
            "head": r["head"],
            "class_weighted": r["class_weighted"],
            "training_time_seconds": r["training_time_seconds"],
            "validation_loss": r["final_validation"]["validation_loss"],
            "accuracy": r["test"]["accuracy"],
            "precision": r["test"]["precision"],
            "recall": r["test"]["recall"],
            "f1": r["test"]["f1"],
            "macro_f1": r["test"]["macro_f1"],
        })
    pd.DataFrame(summary_rows).to_csv(out / "comparison.csv", index=False)
    save_json({"device": device_report(), "splits": splits, "experiments": summary_rows}, out / "summary.json")

    if cfg["experiments"].get("generate_plots"):
        generate_plots(out)

    print("\nCompleted. Results are in ./results")