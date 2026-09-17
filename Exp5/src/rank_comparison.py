import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config():
    with open(ROOT / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def read_json(path):
    if not path.exists():
        return None

    with open(path, "r") as f:
        return json.load(f)


def average_score(data):
    if not data:
        return None

    queries = data.get("queries", [])

    scores = [
        x["score"]
        for x in queries
        if x.get("score") is not None
    ]

    if not scores:
        return None

    return sum(scores) / len(scores)


def main():
    config = load_config()

    ranks = config["experiments"]["ranks"]

    rows = []

    for rank in ranks:

        train_data = read_json(
            ROOT
            / "results"
            / f"train_rank_{rank}.json"
        )

        eval_data = read_json(
            ROOT
            / "results"
            / f"finetuned_rank_{rank}.json"
        )

        if train_data is None:
            print(
                f"Skipping rank {rank}: "
                f"training result not found."
            )
            continue

        rows.append(
            {
                "LoRA Rank": rank,

                "Total Parameters":
                    train_data.get(
                        "total_parameters"
                    ),

                "Trainable Parameters":
                    train_data.get(
                        "trainable_parameters"
                    ),

                "Trainable %":
                    train_data.get(
                        "trainable_percentage"
                    ),

                "Training Time (min)":
                    train_data.get(
                        "training_time_minutes"
                    ),

                "Epochs":
                    train_data.get(
                        "epochs"
                    ),

                "Train Loss":
                    train_data.get(
                        "train_loss"
                    ),

                "Validation Loss":
                    train_data.get(
                        "validation_loss"
                    ),

                "Average Test Score":
                    average_score(eval_data),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        print("No rank results available.")
        return

    output_csv = (
        ROOT
        / "results"
        / "rank_comparison.csv"
    )

    output_json = (
        ROOT
        / "results"
        / "rank_comparison.json"
    )

    df.to_csv(
        output_csv,
        index=False,
    )

    with open(output_json, "w") as f:
        json.dump(
            rows,
            f,
            indent=2,
        )

    print("\nLoRA Rank Comparison")
    print("=" * 80)
    print(df.to_string(index=False))
    print("\nSaved:")
    print(output_csv)
    print(output_json)

    print(
        "\nNote: Average Test Score will remain "
        "blank until the 5 responses are manually "
        "scored from 1-5."
    )


if __name__ == "__main__":
    main()