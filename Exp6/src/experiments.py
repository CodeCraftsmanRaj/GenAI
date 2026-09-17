from pathlib import Path
import copy

import pandas as pd
import torch

from .modeling import CustomBertClassifier, load_baseline_model, load_tokenizer
from .trainer import train_model, evaluate_model
from .utils import save_json


def run_experiment(
    name,
    train_df,
    val_df,
    test_df,
    cfg,
    device,
    max_length,
    lr,
    batch_size,
    epochs,
    weight_decay,
    head="linear",
    use_class_weights=False,
):
    model_name = cfg["tokenization"]["model_name"]
    tokenizer = load_tokenizer(model_name)
    num_labels = int(train_df["label"].nunique())

    if head == "baseline":
        model = load_baseline_model(model_name, num_labels)
    else:
        model = CustomBertClassifier(
            model_name,
            num_labels,
            head=head,
            dropout=cfg["training"]["classifier"]["dropout"],
            hidden_dim=cfg["training"]["classifier"]["hidden_dim"],
        )

    print(f"\n=== {name} | length={max_length} | head={head} | weighted={use_class_weights} ===")
    trained = train_model(
        model, train_df, val_df, tokenizer, max_length, lr, batch_size,
        epochs, weight_decay, device, num_labels, use_class_weights,
        cfg["training"]["baseline"]["warmup_ratio"],
    )
    test_metrics, preds, labels = evaluate_model(
        trained.model, test_df, tokenizer, max_length, batch_size, device
    )
    result = {
        "experiment": name,
        "sequence_length": max_length,
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "weight_decay": weight_decay,
        "head": head,
        "class_weighted": use_class_weights,
        "training_time_seconds": trained.elapsed_seconds,
        "final_validation": trained.history[-1],
        "test": test_metrics,
        "history": trained.history,
    }

    out = Path(cfg["project"]["output_dir"])
    save_json(result, out / f"{name}.json")
    if cfg["experiments"].get("save_models", True):
        model_path = Path(cfg["project"]["model_dir"]) / name
        model_path.mkdir(parents=True, exist_ok=True)
        torch.save(trained.model.state_dict(), model_path / "pytorch_model.bin")
        save_json({"model_name": model_name, "head": head, "num_labels": num_labels}, model_path / "meta.json")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result, preds, labels