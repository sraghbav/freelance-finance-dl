# Freelance Finance Deep Learning Project

**DSCI 410L — Introduction to Deep Learning**
Sriraghav Bavineni · University of Oregon

---

## Project Purpose

Many individuals, especially freelancers, experience irregular income and spending patterns that make budgeting and financial planning difficult. This project investigates how deep learning can identify unusual financial behavior from transaction histories — not by forecasting future spending, but by learning what *normal* spending looks like and flagging anything that deviates significantly from it.

The model is a **LSTM sequence autoencoder** trained on category-specific transaction sequences. It compresses each sequence to a compact latent representation and reconstructs it. Sequences it cannot reconstruct accurately receive a high reconstruction error (MSE) and are flagged as potential anomalies. This approach is well-suited to real-world financial monitoring because it requires no labeled anomaly examples — only examples of normal behavior.

---

## Dataset

**BudgetWise Personal Finance Dataset**
Source: https://www.kaggle.com/datasets/mohammedarfathr/budgetwise-personal-finance-dataset

The dataset contains ~15,900 transaction records with the following fields: `user_id`, `date`, `transaction_type`, `category`, `amount`, `payment_mode`, `location`, `notes`.

**How sequences are created:**
1. Transactions are grouped by `(user_id, category)` — e.g., all food purchases for user U042
2. Within each group, transactions are sorted by date
3. A sliding window of length 5 produces overlapping sequences: positions [0–4], [1–5], [2–6], etc.
4. Each sequence is min-max normalized independently to [0, 1] so that the model is not dominated by high-value categories (e.g. rent vs. coffee)

**Category normalization:** The raw data contains 46 spelling and casing variants across 11 logical categories (e.g. `FOOD`, `Food`, `Fod`, `Foodd`, `Foods` all mean food). These are mapped to canonical names in `dataloader.py` before sequence generation. Without this step only 47 usable sequences are produced; after normalization the dataset yields **1,104 sequences**.

**Dataset location:** `data/budgetwise_finance_dataset.csv`

---

## Repository Structure

```text
freelance-finance-dl/
│
├── freelance_finance_dl/
│   ├── __init__.py
│   ├── dataloader.py      # Dataset class, category normalization, sequence generation
│   ├── model.py           # LSTM sequence autoencoder
│   ├── train.py           # Training loop with checkpointing
│   └── evaluate.py        # Reconstruction error scoring, anomaly threshold,
│                          # synthetic anomaly evaluation
│
├── data/
│   └── budgetwise_finance_dataset.csv
│
├── checkpoints/
│   └── best_model.pt      # Trained model weights
│
├── notebooks/
│   ├── data_demo.ipynb    # Dataset exploration and dataloader walkthrough
│   └── evaluation.ipynb   # Training, all visualizations, evaluation results
│
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

## How to Train the Model

```bash
python -m freelance_finance_dl.train \
    --csv data/budgetwise_finance_dataset.csv \
    --output-dir checkpoints \
    --epochs 50 \
    --hidden-dim 64 \
    --latent-dim 16 \
    --lr 1e-3 \
    --val-split 0.15
```

The script prints train and validation MSE each epoch and saves the best checkpoint (lowest validation MSE) to `checkpoints/best_model.pt`.

To run evaluation and flag anomalies:

```bash
python -m freelance_finance_dl.evaluate \
    --csv data/budgetwise_finance_dataset.csv \
    --checkpoint checkpoints/best_model.pt \
    --output results/anomalies.csv \
    --threshold-percentile 95
```

**Trained weights location:** `checkpoints/best_model.pt`

---

## Model

`TransactionAutoencoder` in `freelance_finance_dl/model.py` is an LSTM sequence autoencoder:

- **Encoder:** an LSTM reads the input sequence one timestep at a time; the final hidden state is projected to a fixed-size latent vector via a linear layer
- **Decoder:** the latent vector is projected to `hidden_dim`, repeated `sequence_length` times, passed through a second LSTM, and mapped back to a scalar output at each timestep

Input and output shape: `(batch, sequence_length, 1)` — the same tensor is used as both input and reconstruction target.

Default hyperparameters: `sequence_length=5`, `hidden_dim=64`, `latent_dim=16`

---

## Results

### Training

The model converged steadily over 50 epochs with no sign of overfitting:

| Epoch | Train MSE | Val MSE |
|-------|-----------|---------|
| 1     | 0.2756    | 0.2061  |
| 10    | 0.1328    | 0.1284  |
| 25    | 0.0952    | 0.0995  |
| 50    | 0.0459    | 0.0491  |

### Anomaly Detection

| Metric | Value |
|--------|-------|
| Final validation MSE | 0.049 |
| Anomaly threshold (95th pct of val errors) | 0.138 |
| Total sequences evaluated | 1,104 |
| Flagged anomalies | 64 (5.8%) |

### Synthetic Evaluation

Since the dataset has no ground-truth anomaly labels, detection performance was measured by injecting known anomalies into validation sequences:

- **Spike:** one position set to `2.0` — simulates an abnormally large charge
- **Drop:** one position set to `-0.5` — simulates a missing or erroneous entry

| Condition | Result |
|-----------|--------|
| Spike recall | **98.2%** (162/165 detected) |
| Drop recall | **30.3%** (50/165 detected) |
| False positive rate | **5.5%** (9/165 clean sequences wrongly flagged) |

The model detects spikes nearly perfectly. Drop recall is lower because averaging MSE over all 5 positions dilutes the signal from a single corrupted step.

### Visualization

The evaluation notebook (`notebooks/evaluation.ipynb`) includes:

- Reconstruction error distribution (normal vs. anomalous sequences)
- Per-category reconstruction error boxplot
- Anomaly count by category
- Threshold justification — validation error histogram and ECDF with threshold marked
- **Qualitative inspection** — top 6 flagged sequences plotted as original vs. reconstructed (shaded gap shows where the model disagrees with the actual transaction)
- Synthetic evaluation — MSE distributions for clean/spike/drop and recall/FPR bar chart

---

## Discussion: Limitations and Use

**What the model does well:**
The LSTM autoencoder reliably detects sequences that contain a large spending spike — 98.2% recall on synthetic spike injection. It is unsupervised, requiring no labeled anomaly examples, which makes it practical for real-world deployment where anomalies are rare and unlabeled.

**Limitations:**
- *Single feature:* the model uses only transaction amount. Adding temporal features (day of week, month) or transaction type (income vs. expense) would give the model richer signal.
- *Drop sensitivity:* detecting unusually low or missing values is harder because the MSE is averaged across all 5 positions. A position-wise max error threshold would improve this.
- *Dataset size:* 1,104 sequences is small. The model may not learn stable representations for low-frequency categories (e.g. health, bonus) that have fewer than 5 sequences.
- *No ground truth:* the synthetic evaluation approximates real anomaly detection performance but does not replace labeled evaluation data.
- *Overlapping windows and sequence-level split:* the sliding window means consecutive sequences share 4 out of 5 values. Combined with a random sequence-level train/val split, sequences from the same user can appear in both sets — slightly inflating validation performance. The correct approach would be a user-level split (train on a set of users, validate on a completely separate set of users) so the model is evaluated on spending patterns it has never seen in any form.

**Intended use:**
This model is a foundation for a larger honors thesis project that will add an LLM explanation layer — translating flagged anomalies into plain-language descriptions and providing tax planning guidance for freelancers. Results are educational only and not financial or tax advice.

---

## Generative AI Disclosure

In accordance with the course syllabus policy on generative AI, this project used Claude (Anthropic) as an assistant throughout development. Specifically, AI assistance was used for:

- **Code organization** — structuring the project into a clean package layout (`freelance_finance_dl/`) with separate files for the dataset, model, training loop, and evaluation
- **Docstrings and inline comments** — drafting explanatory comments throughout the codebase
- **Debugging** — identifying shape mismatches between tensors, fixing the category normalization fallback logic, and resolving tensor dimension issues in the model
- **README drafting** — organizing and writing the README to match submission requirements
- **Evaluation design** — suggesting the synthetic anomaly injection approach (spike and drop) in response to instructor feedback about the lack of ground-truth labels
- **Running and verifying outputs** — executing the training pipeline, evaluation scripts, and notebook re-runs to confirm everything worked end-to-end

All core project decisions — the choice of LSTM autoencoder for anomaly detection, the sequence generation approach, the threshold methodology, and the interpretation of results — were made by the student. The AI was used as a coding and organizational tool, not as a replacement for understanding the material.

---

## Author

Sriraghav Bavineni
DSCI 410L — Introduction to Deep Learning
University of Oregon
