from pathlib import Path
import json
import statistics

import yaml
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def load_config():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path):
    """
    Read a JSONL file safely.

    Each physical line must contain exactly one JSON object.
    """

    records = []

    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}.\n"
                    f"Error: {e}\n"
                    f"Content: {line[:300]}"
                ) from e

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} in {path} "
                    "does not contain a JSON object."
                )

            records.append(record)

    return records


def format_record(record):
    """
    Convert an instruction/input/response record into
    the text format used by the causal language model.
    """

    instruction = str(record.get("instruction", "")).strip()
    user_input = str(record.get("input", "")).strip()
    response = str(record.get("response", "")).strip()

    return (
        f"Instruction: {instruction}\n"
        f"Customer: {user_input}\n"
        f"Agent: {response}"
    )


def main():
    config = load_config()

    model_name = config["model"]["name"]
    max_length = int(config["model"]["max_length"])

    data_dir = ROOT / "data"
    results_dir = ROOT / config["project"]["output_dir"]

    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT-style causal LMs may not have a pad token.
    # Use EOS as the padding token when necessary.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------------------------------------------------
    # Load datasets
    # ---------------------------------------------------------
    train_records = load_jsonl(data_dir / "train.jsonl")
    val_records = load_jsonl(data_dir / "validation.jsonl")
    test_records = load_jsonl(data_dir / "test.jsonl")

    all_records = (
        train_records
        + val_records
        + test_records
    )

    if not all_records:
        raise RuntimeError("No records found for tokenization.")

    # ---------------------------------------------------------
    # Tokenize
    # ---------------------------------------------------------
    lengths = []

    for record in all_records:

        text = format_record(record)

        encoded = tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=max_length,
        )

        lengths.append(len(encoded["input_ids"]))

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------
    truncated_count = sum(
        length >= max_length
        for length in lengths
    )

    stats = {
        "model": model_name,
        "tokenizer": tokenizer.__class__.__name__,
        "max_sequence_length": max_length,

        "total_records": len(all_records),

        "train_records": len(train_records),
        "validation_records": len(val_records),
        "test_records": len(test_records),

        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "mean_tokens": round(statistics.mean(lengths), 2),
        "median_tokens": statistics.median(lengths),

        "records_reaching_max_length": truncated_count,

        "truncation": True,
        "padding": "dynamic / applied during training",

        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,

        "sequence_length_justification": (
            f"{max_length} tokens provides sufficient room for the "
            "instruction, customer query and banking-support response "
            "while keeping CPU memory and training time manageable."
        ),
    }

    # ---------------------------------------------------------
    # Save statistics
    # ---------------------------------------------------------
    with open(
        results_dir / "tokenization_stats.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            stats,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ---------------------------------------------------------
    # Display example
    # ---------------------------------------------------------
    example = all_records[0]
    example_text = format_record(example)

    example_tokens = tokenizer(
        example_text,
        truncation=True,
        padding=False,
        max_length=max_length,
    )

    print("\nExample formatted record:")
    print(example_text)

    print("\nToken IDs:")
    print(example_tokens["input_ids"])

    print("\nToken count:")
    print(len(example_tokens["input_ids"]))

    print("\nTokenization statistics:")
    print(json.dumps(stats, indent=2))

    print("\nTokenization completed.")


if __name__ == "__main__":
    main()