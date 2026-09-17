import json
import sys
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


ROOT = Path(__file__).resolve().parents[1]


QUERIES = [
    "I noticed a cash withdrawal that I do not recognize. What should I do?",
    "My mobile banking app says that my account is temporarily blocked. How can I regain access?",
    "I transferred money yesterday but the recipient still has not received it. What should I check?",
    "I want to change the phone number linked to my bank account. What is the process?",
    "My debit card is not working even though I have sufficient balance. What should I check?",
]


def load_config():
    with open(ROOT / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def generate(model, tokenizer, prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            num_beams=3,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][
        inputs["input_ids"].shape[1]:
    ]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()


def main():
    config = load_config()

    if "--rank" not in sys.argv:
        raise SystemExit(
            "Usage: uv run src/evaluate.py --rank 4"
        )

    rank = int(
        sys.argv[
            sys.argv.index("--rank") + 1
        ]
    )

    model_name = config["model"]["name"]

    adapter_path = (
        ROOT
        / "models"
        / f"lora_rank_{rank}"
    )

    if not adapter_path.exists():
        raise FileNotFoundError(
            f"LoRA model not found:\n"
            f"{adapter_path}\n\n"
            f"Train rank {rank} first."
        )

    print(
        f"Loading base model: {model_name}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = (
        AutoModelForCausalLM.from_pretrained(
            model_name
        )
    )

    print(
        f"Loading LoRA adapter: "
        f"{adapter_path}"
    )

    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
    )

    model.eval()

    instruction = config["data"]["instruction"]

    results = []

    print(
        f"\nFine-tuned evaluation "
        f"(LoRA rank {rank})\n"
    )

    for i, query in enumerate(QUERIES, 1):

        prompt = (
            f"Instruction: {instruction}\n"
            f"Customer: {query}\n"
            f"Agent:"
        )

        response = generate(
            model,
            tokenizer,
            prompt,
        )

        print(f"Q{i}: {query}")
        print(f"Response: {response}\n")

        results.append(
            {
                "query_id": i,
                "query": query,
                "response": response,
                "score": None,
            }
        )

    path = (
        ROOT
        / "results"
        / f"finetuned_rank_{rank}.json"
    )

    with open(path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "rank": rank,
                "queries": results,
            },
            f,
            indent=2,
        )

    print(f"Saved to: {path}")

    print(
        "\nScores remain null until manually "
        "evaluated on the 1-5 scale."
    )


if __name__ == "__main__":
    main()