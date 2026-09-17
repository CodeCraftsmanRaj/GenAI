import json
import random
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config():
    with open(ROOT / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def clean_dataframe(df):
    df = df.copy()

    df = df.dropna(subset=["question", "response"])

    df["question"] = df["question"].astype(str).str.strip()
    df["response"] = df["response"].astype(str).str.strip()

    df = df[
        (df["question"] != "")
        & (df["response"] != "")
    ]

    before = len(df)

    df = df.drop_duplicates(
        subset=["question", "response"]
    )

    duplicates_removed = before - len(df)

    return df, duplicates_removed


def make_variants(question):
    q = question.strip()

    templates = [
        lambda x: x,
        lambda x: f"Could you please explain {x[0].lower() + x[1:]}",
        lambda x: f"Please help me with this: {x}",
        lambda x: f"I need assistance with this. {x}",
        lambda x: f"Can you tell me {x[0].lower() + x[1:]}",
        lambda x: f"I would like to know: {x}",
    ]

    return [fn(q) for fn in templates]


def create_synthetic_data(seed_df, target_size, seed):
    rng = random.Random(seed)

    rows = []

    for _, row in seed_df.iterrows():
        variants = make_variants(row["question"])

        for variant in variants:
            rows.append(
                {
                    "question": variant,
                    "response": row["response"],
                }
            )

    # Add further deterministic combinations if required.
    while len(rows) < target_size:
        row = seed_df.iloc[rng.randrange(len(seed_df))]
        variants = make_variants(row["question"])

        candidate = {
            "question": rng.choice(variants),
            "response": row["response"],
        }

        rows.append(candidate)

    df = pd.DataFrame(rows)

    df = df.drop_duplicates(
        subset=["question", "response"]
    ).reset_index(drop=True)

    # If deduplication reduced the size, keep generating.
    while len(df) < target_size:
        row = seed_df.iloc[rng.randrange(len(seed_df))]
        base = row["question"]

        prefixes = [
            "I need help with this.",
            "Could you assist me?",
            "Please advise.",
            "I have a question.",
        ]

        suffixes = [
            "",
            " Please explain.",
            " What should I do?",
            " I need assistance.",
        ]

        question = f"{rng.choice(prefixes)} {base}{rng.choice(suffixes)}"

        new_row = pd.DataFrame(
            [{
                "question": question,
                "response": row["response"],
            }]
        )

        df = pd.concat(
            [df, new_row],
            ignore_index=True
        ).drop_duplicates(
            subset=["question", "response"]
        )

    return df.head(target_size).reset_index(drop=True)


def format_record(question, response, instruction):
    return {
        "instruction": instruction,
        "input": question,
        "response": response,
    }


def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )


def main():
    config = load_config()

    data_cfg = config["data"]
    seed = config["seed"]

    csv_path = ROOT / data_cfg["csv_path"]

    seed_df = pd.read_csv(csv_path)

    seed_records = len(seed_df)

    missing_values = int(
        seed_df[["question", "response"]]
        .isna()
        .sum()
        .sum()
    )

    if data_cfg["source"] == "synthetic":
        df = create_synthetic_data(
            seed_df,
            data_cfg["synthetic_samples"],
            seed,
        )
    else:
        df = seed_df.copy()

    records_before_cleaning = len(df)

    df, duplicates_removed = clean_dataframe(df)

    random_state = seed

    train_df = df.sample(
        frac=data_cfg["train_ratio"],
        random_state=random_state,
    )

    remaining = df.drop(train_df.index)

    val_fraction = (
        data_cfg["val_ratio"]
        / (
            data_cfg["val_ratio"]
            + data_cfg["test_ratio"]
        )
    )

    val_df = remaining.sample(
        frac=val_fraction,
        random_state=random_state,
    )

    test_df = remaining.drop(val_df.index)

    instruction = data_cfg["instruction"]

    train_records = [
        format_record(
            row.question,
            row.response,
            instruction,
        )
        for row in train_df.itertuples()
    ]

    val_records = [
        format_record(
            row.question,
            row.response,
            instruction,
        )
        for row in val_df.itertuples()
    ]

    test_records = [
        format_record(
            row.question,
            row.response,
            instruction,
        )
        for row in test_df.itertuples()
    ]

    data_dir = ROOT / "data"
    results_dir = ROOT / "results"

    data_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    save_jsonl(
        train_records,
        data_dir / "train.jsonl",
    )

    save_jsonl(
        val_records,
        data_dir / "validation.jsonl",
    )

    save_jsonl(
        test_records,
        data_dir / "test.jsonl",
    )

    with open(
        data_dir / "train_preview.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            train_records[:5],
            f,
            indent=2,
            ensure_ascii=False,
        )

    stats = {
        "source": data_cfg["source"],
        "seed_records": seed_records,
        "records_before_cleaning": records_before_cleaning,
        "missing_values": missing_values,
        "duplicates_removed": duplicates_removed,
        "records_after_cleaning": len(df),
        "train": len(train_records),
        "validation": len(val_records),
        "test": len(test_records),
        "split_ratios": {
            "train": data_cfg["train_ratio"],
            "validation": data_cfg["val_ratio"],
            "test": data_cfg["test_ratio"],
        },
        "instruction": instruction,
    }

    with open(
        results_dir / "dataset_stats.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            stats,
            f,
            indent=2,
        )

    print("\nExample records:\n")

    for record in train_records[:5]:
        print(json.dumps(
            record,
            indent=2,
            ensure_ascii=False,
        ))

    print("\nDataset preparation completed.")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()