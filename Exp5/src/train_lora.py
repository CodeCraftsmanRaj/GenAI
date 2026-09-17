import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


ROOT = Path(__file__).resolve().parents[1]


def load_config():
    with open(ROOT / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def get_rank(config):
    if "--rank" in sys.argv:
        idx = sys.argv.index("--rank")
        return int(sys.argv[idx + 1])

    return int(config["lora"]["r"])


def format_prompt(instruction, question):
    return (
        f"Instruction: {instruction}\n"
        f"Customer: {question}\n"
        f"Agent:"
    )


def tokenize_example(example, tokenizer, max_length):
    prompt = format_prompt(
        example["instruction"],
        example["input"],
    )

    response = example["response"]

    prompt_text = prompt + " "
    full_text = (
        prompt_text
        + response
        + tokenizer.eos_token
    )

    full = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        padding=False,
    )

    prompt_ids = tokenizer(
        prompt_text,
        max_length=max_length,
        truncation=True,
        padding=False,
    )["input_ids"]

    labels = full["input_ids"].copy()

    prompt_length = min(
        len(prompt_ids),
        len(labels),
    )

    labels[:prompt_length] = (
        [-100] * prompt_length
    )

    full["labels"] = labels

    return full


def save_loss_plot(log_history, path, rank):
    train_points = [
        (x["epoch"], x["loss"])
        for x in log_history
        if "loss" in x and "epoch" in x
    ]

    eval_points = [
        (x["epoch"], x["eval_loss"])
        for x in log_history
        if "eval_loss" in x and "epoch" in x
    ]

    if not train_points and not eval_points:
        return

    plt.figure(figsize=(8, 5))

    if train_points:
        plt.plot(
            [x[0] for x in train_points],
            [x[1] for x in train_points],
            marker="o",
            label="Training Loss",
        )

    if eval_points:
        plt.plot(
            [x[0] for x in eval_points],
            [x[1] for x in eval_points],
            marker="o",
            label="Validation Loss",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"LoRA Fine-Tuning Loss - Rank {rank}"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()


def main():
    config = load_config()

    rank = get_rank(config)

    model_name = config["model"]["name"]
    max_length = config["model"]["max_length"]

    data_dir = ROOT / "data"
    results_dir = ROOT / "results"
    model_dir = (
        ROOT
        / "models"
        / f"lora_rank_{rank}"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(f"Loading model: {model_name}")
    print(f"LoRA rank: {rank}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name
    )

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        task_type=config["lora"]["task_type"],
        bias="none",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    trainable_percentage = (
        100.0
        * trainable_parameters
        / total_parameters
    )

    print()
    print("LoRA parameter analysis")
    print("-" * 40)
    print(f"LoRA rank          : {rank}")
    print(
        f"LoRA alpha         : "
        f"{config['lora']['alpha']}"
    )
    print(
        f"LoRA dropout       : "
        f"{config['lora']['dropout']}"
    )
    print(
        f"Target modules     : "
        f"{config['lora']['target_modules']}"
    )
    print(
        f"Total parameters   : "
        f"{total_parameters:,}"
    )
    print(
        f"Trainable params   : "
        f"{trainable_parameters:,}"
    )
    print(
        f"Trainable %        : "
        f"{trainable_percentage:.4f}%"
    )
    print("-" * 40)

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(
                data_dir / "train.jsonl"
            ),
            "validation": str(
                data_dir / "validation.jsonl"
            ),
        },
    )

    tokenized = dataset.map(
        lambda x: tokenize_example(
            x,
            tokenizer,
            max_length,
        ),
        remove_columns=[
            "instruction",
            "input",
            "response",
        ],
    )

    batch_size = config["training"][
        "batch_size"
    ]

    gradient_accumulation = config[
        "training"
    ]["gradient_accumulation_steps"]

    epochs = config["training"]["epochs"]

    effective_batch = (
        batch_size
        * gradient_accumulation
    )

    steps_per_epoch = max(
        1,
        (
            len(tokenized["train"])
            + effective_batch
            - 1
        )
        // effective_batch,
    )

    total_steps = (
        steps_per_epoch * epochs
    )

    warmup_steps = max(
        1,
        int(
            total_steps
            * config["training"][
                "warmup_ratio"
            ]
        ),
    )

    print()
    print(
        f"Estimated training steps : "
        f"{total_steps}"
    )
    print(
        f"Warmup steps             : "
        f"{warmup_steps}"
    )

    training_args = TrainingArguments(
        output_dir=str(model_dir),

        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,

        gradient_accumulation_steps=(
            gradient_accumulation
        ),

        num_train_epochs=epochs,

        learning_rate=config["training"][
            "learning_rate"
        ],

        weight_decay=config["training"][
            "weight_decay"
        ],

        warmup_steps=warmup_steps,

        logging_steps=config["training"][
            "logging_steps"
        ],

        save_strategy=config["training"][
            "save_strategy"
        ],

        eval_strategy="epoch",

        report_to="none",

        use_cpu=True,

        fp16=False,

        dataloader_pin_memory=False,

        remove_unused_columns=True,

        seed=config["seed"],
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
    )

    print()
    print("Starting LoRA fine-tuning...")
    print(f"Rank: {rank}")
    print()

    start_time = time.time()

    trainer.train()

    training_time_minutes = (
        time.time() - start_time
    ) / 60

    evaluation = trainer.evaluate()

    validation_loss = evaluation.get(
        "eval_loss"
    )

    log_history = trainer.state.log_history

    train_losses = [
        x["loss"]
        for x in log_history
        if "loss" in x
    ]

    train_loss = (
        train_losses[-1]
        if train_losses
        else None
    )

    model.save_pretrained(
        model_dir
    )

    tokenizer.save_pretrained(
        model_dir
    )

    metrics = {
        "rank": rank,
        "model": model_name,

        "total_parameters":
            total_parameters,

        "trainable_parameters":
            trainable_parameters,

        "trainable_percentage":
            round(
                trainable_percentage,
                4,
            ),

        "training_time_minutes":
            round(
                training_time_minutes,
                4,
            ),

        "epochs": epochs,

        "learning_rate":
            config["training"][
                "learning_rate"
            ],

        "batch_size": batch_size,

        "gradient_accumulation_steps":
            gradient_accumulation,

        "train_loss":
            train_loss,

        "validation_loss":
            validation_loss,

        "log_history":
            log_history,
    }

    result_path = (
        results_dir
        / f"train_rank_{rank}.json"
    )

    with open(
        result_path,
        "w",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    save_loss_plot(
        log_history,
        results_dir
        / f"loss_rank_{rank}.png",
        rank,
    )

    print()
    print("Training completed.")
    print(
        json.dumps(
            {
                "rank": rank,
                "total_parameters":
                    total_parameters,
                "trainable_parameters":
                    trainable_parameters,
                "trainable_percentage":
                    round(
                        trainable_percentage,
                        4,
                    ),
                "training_time_minutes":
                    round(
                        training_time_minutes,
                        4,
                    ),
                "train_loss":
                    train_loss,
                "validation_loss":
                    validation_loss,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()