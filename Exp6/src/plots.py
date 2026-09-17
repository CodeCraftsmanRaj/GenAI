from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def load_results(results_dir):
    rows = []

    excluded = {
        "dataset_report.json",
        "tokenization_report.json",
        "robustness.json",
        "summary.json",
    }

    for p in sorted(Path(results_dir).glob("*.json")):
        if p.name in excluded:
            continue

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            rows.append(data)

    return rows


def get_experiment_name(row):
    return row.get(
        "experiment",
        row.get(
            "name",
            row.get("id", "unknown"),
        ),
    )


def get_metric(row, metric):
    """
    Supports both possible result formats:

    1. {
        "test": {
            "accuracy": ...,
            "f1": ...
        }
    }

    2. {
        "accuracy": ...,
        "f1": ...
    }
    """
    test = row.get("test")

    if isinstance(test, dict):
        value = test.get(metric)
        if value is not None:
            return float(value)

    value = row.get(metric)
    if value is not None:
        return float(value)

    # Also support common alternate naming.
    if metric == "f1":
        value = row.get("test_f1", row.get("val_f1"))
    elif metric == "accuracy":
        value = row.get("test_accuracy", row.get("val_accuracy"))
    else:
        value = None

    return float(value) if value is not None else 0.0


def get_confusion_matrix(row):
    """
    Supports confusion matrices stored either inside
    a test dictionary or directly in the result.
    """
    test = row.get("test")

    if isinstance(test, dict):
        cm = test.get("confusion_matrix")
        if cm is not None:
            return np.array(cm)

    cm = row.get("confusion_matrix")

    if cm is not None:
        return np.array(cm)

    return None


def generate_plots(results_dir):
    results_dir = Path(results_dir)
    plot_dir = results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = load_results(results_dir)

    if not rows:
        return

    # ============================================================
    # Plot 1: Training Loss vs Epoch
    # ============================================================

    plt.figure(figsize=(9, 6))

    plotted = False

    for r in rows:
        history = r.get("history", [])

        if not isinstance(history, list) or not history:
            continue

        epochs = []
        losses = []

        for item in history:
            if not isinstance(item, dict):
                continue

            epoch = item.get("epoch")
            loss = item.get("train_loss")

            if epoch is not None and loss is not None:
                epochs.append(epoch)
                losses.append(loss)

        if epochs:
            plt.plot(
                epochs,
                losses,
                marker="o",
                label=get_experiment_name(r),
            )
            plotted = True

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Epoch")

    if plotted:
        plt.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(
        plot_dir / "plot1_training_loss.png",
        dpi=180,
    )
    plt.close()

    # ============================================================
    # Plot 2: Validation Loss vs Epoch
    # ============================================================

    plt.figure(figsize=(9, 6))

    plotted = False

    for r in rows:
        history = r.get("history", [])

        if not isinstance(history, list) or not history:
            continue

        epochs = []
        losses = []

        for item in history:
            if not isinstance(item, dict):
                continue

            epoch = item.get("epoch")
            loss = item.get("validation_loss")

            if loss is None:
                loss = item.get("val_loss")

            if epoch is not None and loss is not None:
                epochs.append(epoch)
                losses.append(loss)

        if epochs:
            plt.plot(
                epochs,
                losses,
                marker="o",
                label=get_experiment_name(r),
            )
            plotted = True

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss vs Epoch")

    if plotted:
        plt.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(
        plot_dir / "plot2_validation_loss.png",
        dpi=180,
    )
    plt.close()

    # ============================================================
    # Plot 3: Accuracy / F1 vs Experiment
    # ============================================================

    labels = [get_experiment_name(r) for r in rows]

    accuracy = [
        get_metric(r, "accuracy")
        for r in rows
    ]

    f1 = [
        get_metric(r, "f1")
        for r in rows
    ]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(
        figsize=(max(10, len(labels) * 1.2), 6)
    )

    plt.bar(
        x - width / 2,
        accuracy,
        width,
        label="Accuracy",
    )

    plt.bar(
        x + width / 2,
        f1,
        width,
        label="F1",
    )

    plt.xticks(
        x,
        labels,
        rotation=45,
        ha="right",
    )

    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Accuracy / F1 vs Experiment")
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        plot_dir / "plot3_accuracy_f1.png",
        dpi=180,
    )
    plt.close()

    # ============================================================
    # Plot 4: Confusion Matrix
    # ============================================================

    # Find the last result that actually contains a
    # confusion matrix rather than blindly assuming the
    # final JSON has one.
    selected_row = None
    selected_cm = None

    for r in reversed(rows):
        cm = get_confusion_matrix(r)

        if cm is not None:
            selected_row = r
            selected_cm = cm
            break

    if selected_row is not None and selected_cm is not None:
        ConfusionMatrixDisplay(
            confusion_matrix=selected_cm,
            display_labels=["ham", "spam"],
        ).plot()

        plt.title(
            f"Confusion Matrix — "
            f"{get_experiment_name(selected_row)}"
        )

        plt.tight_layout()

        plt.savefig(
            plot_dir / "plot4_confusion_matrix.png",
            dpi=180,
        )

        plt.close()