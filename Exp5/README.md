# EXP5 — Banking Customer Support Fine-Tuning Using LoRA

This project implements the SPIT Generative AI Experiment 5 workflow:
domain-specific dataset preparation → 80/10/10 split → instruction-response formatting →
tokenization → baseline evaluation → LoRA → CPU fine-tuning → parameter analysis →
loss tracking → post-fine-tuning evaluation → LoRA rank comparison.

## Run

After `uv init` and `uv venv`:

```bash
uv pip install -r requirements.txt
uv run main.py
```

Individual scripts:

```bash
uv run src/prepare_data.py
uv run src/tokenize.py
uv run src/baseline.py
uv run src/train_lora.py --rank 8
uv run src/evaluate.py --rank 8
uv run src/rank_comparison.py
```

## Dataset

Default: a small synthetic banking customer-support dataset generated locally.

You can instead use a CSV by changing:

```yaml
data:
  source: "csv"
  csv_path: "data/banking_support.csv"
```

Required CSV columns:

```text
question,response
```

## Model

The default `sshleifer/tiny-gpt2` is intentionally tiny so the experiment remains practical on CPU.

To use another causal LM, change `model.name` in `config.yaml`. **LoRA target modules depend on the architecture.** For GPT-2, `c_attn` is used. For another model, inspect its module names and update `target_modules`.

## Configurable experiment parameters

All major settings are in `config.yaml`: dataset source, model, maximum sequence length, LoRA rank/alpha/dropout/targets, learning rate, batch size, accumulation, epochs, and rank comparison.

## Generated results

`results/` contains dataset statistics, tokenization statistics, baseline outputs, LoRA parameter counts, training/validation losses, fine-tuned outputs and rank-comparison data.

The five evaluation queries are fixed and reused before/after fine-tuning for a fair comparison.

**Important:** baseline/fine-tuned scores are intentionally left `null` for manual 1–5 scoring using the same rubric. This prevents the script from inventing subjective quality scores.
