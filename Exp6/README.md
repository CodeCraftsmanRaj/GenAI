
## 0. Important implementation note

The pipeline passes explicit `text` and `label` columns into each experiment. This avoids duplicate pandas column names when switching between the original, minimal, and normalized text versions.

# EXP6 — Fine-Tuning and Comparative Evaluation of a Pretrained BERT Model for Sentence Classification

**Course:** Generative AI (PE III), CE413  
**Experiment:** 6  
**Task:** SMS Spam Detection (`ham` / `spam`)  
**Model:** `bert-base-uncased`

> This implementation follows the SPIT AY 2026–27 lab manual. The manual asks for a real-world sentence-classification task, preprocessing/tokenization analysis, BERT fine-tuning, sequence-length and hyperparameter experiments, classification-head comparison, imbalance handling, robustness/error analysis, plots, and evidence-based discussion.

## 1. Project structure

```text
.
├── config.yaml
├── main.py
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── experiments.py
│   ├── metrics.py
│   ├── modeling.py
│   ├── pipeline.py
│   ├── plots.py
│   ├── trainer.py
│   └── utils.py
├── data/
├── models/
└── results/
```

## 2. Installation and execution

Using `uv`:

```bash
uv add -r requirements.txt
uv run main.py
```

For a fast smoke test before the full run:

```bash
uv run main.py --quick
```

The full run downloads the dataset and pretrained BERT model from Hugging Face. A GPU is recommended by the manual; the code automatically uses CUDA when PyTorch detects it and otherwise uses CPU.

## 3. Dataset

The experiment uses the `ucirvine/sms_spam` Hugging Face dataset. The loader automatically detects common text and label column names and converts the two SMS classes to:

- `0` = ham
- `1` = spam

The dataset is cleaned by removing duplicate `(text, label)` records and is split using stratification into:

- Training: 70%
- Validation: 15%
- Testing: 15%

The exact counts, class distribution, and label mapping are written to `results/dataset_report.json` and `results/split_sizes.json`.

## 4. Experiment design

### Version A — Minimal preprocessing

Only whitespace is normalized. URLs, punctuation, casing, abbreviations, and other language information are retained because BERT's tokenizer is designed to process natural text.

### Version B — Cleaned/normalized

Whitespace is normalized and URLs/emails are replaced by placeholder tokens. This is compared with Version A rather than assumed to be better.

### BERT tokenization

The tokenizer is:

```text
bert-base-uncased
```

The experiment records tokens, token IDs, attention masks, padding, truncation, and non-padding token counts for:

- `max_length = 32`
- `max_length = 128`

### Classification heads

The implementation supports the three heads specified by the manual:

1. Linear: `BERT → Linear → Softmax`
2. Dropout: `BERT → Dropout → Linear → Softmax`
3. Dense + Dropout: `BERT → Dense → Dropout → Linear → Softmax`

Softmax is implicit during prediction through the logits; cross-entropy loss internally applies the appropriate log-softmax operation.

### Class imbalance

Two otherwise identical training conditions are compared:

- Standard cross-entropy
- Class-weighted cross-entropy

Accuracy, precision, recall, F1, macro-F1, and the confusion matrix are recorded.

### Hyperparameters

The manual suggests investigating learning rate, batch size, epochs, sequence length, and weight decay. The configuration includes the full candidate ranges and a selected default set:

- Learning rate: `2e-5`, `3e-5`, `5e-5`
- Batch size: `8`, `16`
- Epochs: `2`, `3`, `5`
- Sequence length: `32`, `64`, `128`
- Weight decay: `0`, `0.01`

The default run evaluates the three learning rates with the other baseline settings held constant. More combinations can be enabled directly in `config.yaml`.

## 5. Required plots

After execution:

```text
results/plots/plot1_training_loss.png
results/plots/plot2_validation_loss.png
results/plots/plot3_accuracy_f1.png
results/plots/plot4_confusion_matrix.png
```

These correspond to the four plot requirements in the manual.

## 6. Output files

Important generated files:

| File | Purpose |
|---|---|
| `results/dataset_report.json` | Missing values, duplicates, class distribution and label mapping |
| `results/split_sizes.json` | Train/validation/test sizes |
| `results/tokenization_report.json` | Tokens, IDs, masks and padding for sequence lengths |
| `results/comparison.csv` | Main comparative experiment table |
| `results/summary.json` | Machine-readable experiment summary |
| `results/robustness.json` | Challenging unseen inputs and predictions |
| `results/E*.json` | Full metrics and training history for each experiment |
| `results/plots/*` | Required figures |
| `models/*` | Saved model weights and metadata |

---

# 7. RESULTS — APPEND AFTER RUNNING

Do **not** invent these values. Paste the terminal output and/or the generated `results/comparison.csv`, `results/summary.json`, and `results/robustness.json` after the run. This section is intentionally a template so the final report is based on measured results.

## Dataset results

```text
[PASTE results/dataset_report.json SUMMARY HERE]
```

## Split results

```text
[PASTE results/split_sizes.json HERE]
```

## Tokenization observations

```text
[PASTE relevant results/tokenization_report.json OBSERVATIONS HERE]
```

## Comparative results

| Experiment | Sequence Length | LR | Batch | Epochs | Head | Weighted | Accuracy | Precision | Recall | F1 | Val Loss | Training Time |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| E1 Baseline BERT | — | — | — | — | — | — | — | — | — | — | — | — |
| E2 Preprocessing | — | — | — | — | — | — | — | — | — | — | — | — |
| E3 Sequence length | — | — | — | — | — | — | — | — | — | — | — | — |
| E4 Hyperparameters | — | — | — | — | — | — | — | — | — | — | — | — |
| E5 Classification head | — | — | — | — | — | — | — | — | — | — | — | — |
| E6 Imbalance | — | — | — | — | — | — | — | — | — | — | — | — |

---

# 8. Results and Discussion — Questions from the Manual

The final answers below should be completed from the measured experiment results where the question asks what happened in this run.

## i. How does BERT tokenization differ from traditional NLP preprocessing?

Traditional NLP pipelines commonly perform operations such as stop-word removal, stemming, lemmatization, and manual vocabulary construction. BERT instead uses a learned WordPiece tokenizer. Text is converted into subword tokens and token IDs, with special tokens such as `[CLS]`, `[SEP]`, and `[PAD]`. The tokenizer also creates an attention mask so the model can distinguish actual tokens from padding.

For this experiment, the important observation is whether aggressive normalization changed the measured validation/test metrics relative to the minimal-preprocessing version.

**Experiment observation:** `[FILL FROM RESULTS]`

## ii. What is the role of `[CLS]` in sentence classification?

`[CLS]` is placed at the beginning of the input. In BERT sequence classification, its final contextual representation is used as the sentence-level representation and is passed to the classification head to produce class logits.

In this implementation, the custom heads explicitly use:

```text
last_hidden_state[:, 0, :]
```

which corresponds to `[CLS]`.

## iii. How did sequence length affect performance and computational cost?

A larger maximum sequence length permits more tokens to be retained before truncation, which can help when informative content occurs later in a sentence. However, self-attention has computational and memory cost that grows approximately quadratically with sequence length.

The experiment compares 32 and 128 tokens.

**Measured performance difference:** `[FILL FROM E3 RESULTS]`  
**Measured training-time difference:** `[FILL FROM E3 RESULTS]`  
**Observed memory/compute difference:** `[FILL FROM GPU/TERMINAL RESULTS]`

## iv. Which hyperparameter had the greatest effect on performance?

The experiment varies learning rate in the default hyperparameter comparison while keeping the other listed settings controlled.

**Measured result:** `[FILL FROM E4 RESULTS]`

Do not claim that a hyperparameter had the greatest effect unless the experimental differences support that conclusion.

## v. Did modifying the classification head improve generalization?

The manual asks the three heads to be compared under the same dataset and training conditions. The relevant comparison should consider test F1, validation loss, and the gap between validation and test behavior rather than accuracy alone.

**Measured result:** `[FILL FROM E5 RESULTS]`

## vi. How did class imbalance affect accuracy and F1-score?

Accuracy can be misleading when one class is substantially more common. A classifier can obtain a high accuracy by predicting the majority class frequently while performing poorly on the minority class. Macro-F1 gives equal weight to the classes and therefore provides additional information about minority-class performance.

The experiment compares standard cross-entropy with class-weighted cross-entropy.

**Measured accuracy difference:** `[FILL]`  
**Measured macro-F1 difference:** `[FILL]`  
**Confusion-matrix observation:** `[FILL]`

## vii. Which errors occurred most frequently?

Use the generated test predictions and robustness/error analysis. Errors should be grouped by observable category rather than attributed to an unsupported cause.

Suggested categories from the manual include:

- Spelling errors
- Abbreviations
- Negation
- Paraphrasing
- Short inputs
- Long inputs
- Ambiguous language
- Domain-specific terminology

**Most frequent observed error category:** `[FILL FROM RESULTS]`

## viii. How robust was the model to spelling errors and paraphrased inputs?

Robustness is evaluated with the unseen examples in `results/robustness.json`.

Report:

- Number correct
- Number incorrect
- Accuracy across the robustness examples
- Performance by category
- Representative failures

**Measured robustness:** `[FILL]`

## ix. Did the best-performing model also provide the best generalization?

Do not equate the highest test accuracy with generalization automatically. Compare validation loss, test F1/macro-F1, robustness performance, and the behavior of the model across controlled changes.

**Measured observation:** `[FILL FROM COMPARATIVE + ROBUSTNESS RESULTS]`

## x. What are the major limitations of the developed BERT classifier?

Relevant limitations include:

1. The model is specialized for the selected SMS classification task.
2. Robustness examples are small and manually constructed, so they are not a statistically representative robustness benchmark.
3. Results depend on the train/validation/test split and random seed.
4. Maximum sequence length creates a truncation trade-off.
5. Fine-tuning BERT requires substantially more computational resources than a lightweight traditional classifier.
6. Class imbalance can make accuracy alone insufficient.
7. Dataset-specific vocabulary and message patterns may limit performance on unseen domains.
8. A high score on a held-out test set does not guarantee robustness to distribution shift, adversarial inputs, or future spam patterns.

---

# 9. Final conclusion — fill from measured evidence

The experiment demonstrates how a pretrained BERT Transformer can be fine-tuned for sentence classification and how preprocessing, tokenization, sequence length, classification-head architecture, hyperparameters, and imbalance handling affect performance.

The final configuration should be identified from the complete experimental evidence rather than simply choosing the largest accuracy value. The final discussion should explicitly mention the observed validation loss, F1/macro-F1, training cost, confusion matrix, and robustness behavior.

**Final selected configuration:** `[FILL AFTER EXPERIMENT]`

**Evidence supporting the selection:** `[FILL AFTER EXPERIMENT]`

**Main limitation observed in this run:** `[FILL AFTER EXPERIMENT]`