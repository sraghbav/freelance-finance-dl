# Freelance Finance Deep Learning Project

## Project Overview

This project investigates how deep learning can identify unusual financial behavior in personal finance transaction data. Rather than forecasting future spending, the model learns what *normal* spending looks like for a given user and category, then flags transactions that deviate significantly from that learned pattern.

The approach uses a **sequence autoencoder**: the model is trained to compress and reconstruct short sequences of transaction amounts. Sequences it cannot reconstruct accurately are treated as potential anomalies.

```text
Raw Transactions
      ↓
Category Normalization + Sequence Generation
      ↓
Per-Sequence Min-Max Normalization
      ↓
LSTM Sequence Autoencoder (Encoder → Latent → Decoder)
      ↓
Reconstruction Error (MSE)
      ↓
Anomaly Threshold (95th percentile of validation errors)
      ↓
Flagged Anomalies
```

---

## Dataset

**BudgetWise Personal Finance Dataset** — Kaggle:
https://www.kaggle.com/datasets/mohammedarfathr/budgetwise-personal-finance-dataset

The dataset contains 15,900 transaction records with fields: `user_id`, `date`, `transaction_type`, `category`, `amount`, `payment_mode`, `location`, `notes`.

The raw data has significant category noise — 46 spelling/casing variants across 11 logical categories (e.g. `FOOD`, `Food`, `Fod`, `Foodd`, `Foods` all meaning food). The dataloader normalizes these to canonical names before sequence generation, which grows the usable sequence count from 47 to 1,104.

The dataset is included in `data/`.

---

## Repository Structure

```text
freelance-finance-dl/
│
├── freelance_finance_dl/
│   ├── __init__.py
│   ├── dataloader.py      # Dataset, category normalization, sequence generation
│   ├── model.py           # LSTM sequence autoencoder
│   ├── train.py           # Training loop with checkpointing
│   └── evaluate.py        # Reconstruction error scoring, anomaly threshold,
│                          # synthetic anomaly evaluation
│
├── data/
│   └── budgetwise_finance_dataset.csv
│
├── notebooks/
│   └── data_demo.ipynb    # Full walkthrough with all visualizations
│
├── checkpoints/           # Saved model weights (git-ignored)
├── results/               # Anomaly CSV output (git-ignored)
├── README.md
├── requirements.txt
└── setup.py
```

---

## Installation

```bash
git clone https://github.com/sraghbav/freelance-finance-dl.git
cd freelance-finance-dl
pip install -e .
```

---

## Usage

### Train

```bash
python -m freelance_finance_dl.train \
    --csv data/budgetwise_finance_dataset.csv \
    --output-dir checkpoints \
    --epochs 50 \
    --hidden-dim 64 \
    --latent-dim 16
```

The best model (lowest validation MSE) is saved to `checkpoints/best_model.pt`.

### Evaluate

```bash
python -m freelance_finance_dl.evaluate \
    --csv data/budgetwise_finance_dataset.csv \
    --checkpoint checkpoints/best_model.pt \
    --output results/anomalies.csv \
    --threshold-percentile 95
```

Prints reconstruction error stats, anomaly counts by category, and saves a full results CSV.

---

## Model

`TransactionAutoencoder` in [`freelance_finance_dl/model.py`](freelance_finance_dl/model.py) is an LSTM sequence autoencoder:

- **Encoder**: an LSTM reads the input sequence one timestep at a time; the final hidden state is projected to a fixed-size latent vector via a linear layer
- **Decoder**: the latent vector is projected back to `hidden_dim`, repeated `sequence_length` times, then fed through a second LSTM; a final linear layer maps each timestep to a scalar output

Input and output shape: `(batch, sequence_length, 1)`

Default hyperparameters: `hidden_dim=64`, `latent_dim=16`, `sequence_length=5`

---

## Anomaly Detection

The anomaly threshold is the **95th percentile of reconstruction errors on the validation set** — sequences the model never trained on. This ensures the threshold reflects generalization to unseen normal behavior, not memorized training patterns.

Any sequence whose MSE exceeds the threshold is flagged as a potential anomaly.

### Synthetic Evaluation

Since the dataset has no ground-truth anomaly labels, detection ability was measured by injecting known anomalies into validation sequences:

- **Spike**: one position set to `2.0` (double the normalized maximum) — simulates an abnormally large charge
- **Drop**: one position set to `-0.5` (below the normalized minimum) — simulates a missing or erroneous entry

| Condition | Result |
|---|---|
| Spike recall | 98.2% |
| Drop recall | 30.3% |
| False positive rate | 5.5% |

The model detects spikes almost perfectly. Drop detection is weaker because mean MSE across all 5 positions dilutes the signal from a single corrupted step. A position-wise max error threshold would improve drop sensitivity.

---

## Notebook

`notebooks/data_demo.ipynb` walks through the full pipeline and includes:

- Raw data exploration and category distribution
- Dataset and dataloader demonstration
- Model architecture overview
- Training (50 epochs, LSTM autoencoder)
- Reconstruction error distribution with anomaly threshold
- Per-category reconstruction error boxplot
- Anomaly count by category
- Threshold justification (validation error histogram + ECDF)
- Qualitative anomaly inspection — original vs reconstructed sequences for the top 6 flagged cases
- Synthetic anomaly evaluation — MSE distributions and recall/FPR bar chart
- Written evaluation summary

---

## Evaluation Results

| Metric | Value |
|---|---|
| Final validation MSE | ~0.049 |
| Anomaly threshold (95th pct of val errors) | ~0.138 |
| Total sequences evaluated | 1,104 |
| Flagged anomalies | 64 (5.8%) |
| Spike recall (synthetic) | 98.2% |
| Drop recall (synthetic) | 30.3% |
| False positive rate | 5.5% |

---

## Future Work

This project is the deep learning foundation for a larger honors thesis involving LLM-based financial guidance. Planned extensions:

- LLM explanation layer that translates flagged anomalies into plain-language descriptions
- Tax planning assistance using detected income/expense patterns
- Additional input features: transaction type, day of week, payment method

---

## Author

Sriraghav Bavineni  
DSCI 410L — Introduction to Deep Learning  
University of Oregon
