# Freelance Finance Deep Learning Project

Sriraghav Bavineni · University of Oregon

Started as the **DSCI 410L (Introduction to Deep Learning)** class project; being extended into a **Clark Honors College thesis** (defense Spring 2027). Everything below describes the current state, with the original class-project results kept for comparison at the end.

---

## Project Purpose

Many individuals, especially freelancers, experience irregular income and spending patterns that make budgeting and financial planning difficult. This project investigates how deep learning can identify unusual financial behavior from transaction histories — not by forecasting future spending, but by learning what *normal* spending looks like and flagging anything that deviates significantly from it.

The pipeline has four stages:

```
transactions → LSTM autoencoder flags unusual sequences → plain-language explanation → educational tax set-aside note
```

The detector is an **LSTM sequence autoencoder** trained on category-specific transaction sequences. Sequences it cannot reconstruct accurately receive a high reconstruction error and are flagged. It requires no labeled anomalies, only examples of normal behavior. The explanation layer turns each flag into a short plain-language description. **All numbers in an explanation are computed in code; the language model only phrases them**, and an automatic grounding check rejects any language-model text that contradicts the data.

Results are educational only and are not financial or tax advice.

---

## Dataset

**BudgetWise Personal Finance Dataset**
Source: https://www.kaggle.com/datasets/mohammedarfathr/budgetwise-personal-finance-dataset

The file has 15,900 rows, but only **4,977 (31%) are usable**: the rest are dropped because `user_id`, `date`, `category`, or `amount` is missing or unparseable. Fields `user_id`, `date`, `transaction_type`, `category`, `amount`, `payment_mode`, `location`, `notes`, covering 150 users from 2021 to 2024.

**How sequences are created:**
1. Transactions are grouped by `(user_id, category)` — e.g. all food purchases for user U042
2. Within each group, transactions are sorted by date
3. A sliding window of length 5 produces overlapping sequences: positions [0–4], [1–5], [2–6], etc.
4. Each sequence is min-max normalized independently to [0, 1] so the model is not dominated by high-value categories (rent vs. coffee)

**Category normalization:** the raw data contains 46 spelling and casing variants of 11 categories (e.g. `FOOD`, `Fod`, `Foods`). They are mapped to canonical names in `dataloader.py`. Without this step only 47 usable sequences exist; after normalization there are **1,104** (148 users have at least one).

**Data-quality finding:** the `amount` column contains corrupt values — 29 rows of `999999999`, 28 of `999999`, and 88 that are zero or negative. **173 of 1,104 sequences (15.7%) contain at least one**, yet only 4 of the 52 flagged anomalies do. Per-sequence min-max scaling squashes a window with a 999,999,999 entry to near-constant values, which the model reconstructs well, so the detector is effectively blind to these errors. An opt-in `--clean` mode drops these rows before building sequences (1,015 sequences remain).

**Other caveats:** the amounts look randomly generated rather than behavioral (e.g. one user's rent windows swing between 1,370 and 16,334), and the locations (Mumbai, Bangalore, Chennai) suggest rupees, so amounts are reported as plain numbers without a currency.

**Dataset location:** `data/budgetwise_finance_dataset.csv`

---

## Repository Structure

```text
freelance-finance-dl/
│
├── freelance_finance_dl/
│   ├── dataloader.py          # Dataset, category normalization, sequences, user-level split, clean mode
│   ├── model.py               # LSTM sequence autoencoder
│   ├── train.py               # Training loop with checkpointing
│   ├── evaluate.py            # Scoring (mean/max), threshold, synthetic anomaly evaluation
│   ├── sweep.py               # Injection-magnitude sensitivity sweep and plots
│   ├── explain.py             # Facts, template baseline, LLM backends, grounding check
│   ├── tax.py                 # Educational quarterly set-aside estimate
│   ├── pipeline.py            # End-to-end report: flag -> explain -> tax note
│   └── eval_explanations.py   # Explanation quality evaluation (auto check + blind rubric)
│
├── checkpoints/               # best_model.pt (user split) and best_model_old_split.pt (original)
├── checkpoints_clean/         # model trained with --clean
├── docs/                      # figures, sweep CSVs, explanation evaluation files
├── notebooks/                 # data_demo.ipynb, evaluation.ipynb
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
pip install -r requirements.txt
```

**Optional, for LLM explanations (free, runs locally):** install [Ollama](https://ollama.com), then `ollama pull llama3.2:3b` and keep `ollama serve` running. The code can also use the Anthropic API (`--llm anthropic`, needs `ANTHROPIC_API_KEY`), but that path has **not been tested** in this project.

---

## How to Run

```bash
# Train (user-level train/val split; add --clean to drop corrupt amounts)
python -m freelance_finance_dl.train --csv data/budgetwise_finance_dataset.csv \
    --output-dir checkpoints --epochs 50 --hidden-dim 64 --latent-dim 16 --lr 1e-3 --val-split 0.15

# Flag anomalies (score = mean or max over positions)
python -m freelance_finance_dl.evaluate --checkpoint checkpoints/best_model.pt --score mean

# Sensitivity sweep -> docs/sweep_<tag>.csv and docs/sensitivity_<tag>.png
python -m freelance_finance_dl.sweep --checkpoint checkpoints/best_model.pt --tag original

# End-to-end report (add --llm ollama for model-written explanations)
python -m freelance_finance_dl.pipeline --top-n 10 --llm ollama

# Explanation evaluation: generate -> fill the blind sheet -> summarize
python -m freelance_finance_dl.eval_explanations --n 20 --tag _v3
python -m freelance_finance_dl.eval_explanations --summarize --tag _v3 --scores <scored.csv>
```

---

## Model

`TransactionAutoencoder` in `freelance_finance_dl/model.py` is an LSTM sequence autoencoder:

- **Encoder:** an LSTM reads the sequence; the final hidden state is projected to a 16-dim latent vector
- **Decoder:** the latent vector is projected to `hidden_dim`, repeated `sequence_length` times, passed through a second LSTM, and mapped to a scalar at each timestep

Input and output shape: `(batch, 5, 1)`. Default hyperparameters: `sequence_length=5`, `hidden_dim=64`, `latent_dim=16`.

---

## Results

### Train/validation split: sequence-level vs. user-level

The original split shuffled individual sequences, so overlapping windows from the same user could land in both train and validation. Validation now holds out **whole users** (126 train users / 22 val users; 955 / 149 sequences).

| | Sequence-level split (original) | User-level split |
|---|---|---|
| Final train MSE | 0.0459 | 0.0454 |
| Final validation MSE | 0.0491 | **0.0577** |
| Threshold (95th pct of val errors) | 0.138 | 0.146 |
| Flagged anomalies | 64 (5.8%) | 52 (4.7%) |

Validation error rose, as expected once leakage was removed. Validation MSE was still falling at epoch 50 (training was not extended, to keep the comparison fair).

### Synthetic evaluation

There are no ground-truth anomaly labels, so detection was measured by injecting known anomalies into validation sequences (spike = one position set to 2.0; drop = one set to -0.5).

| | Original split | User-level split |
|---|---|---|
| Spike recall | 98.2% (162/165) | 100% (149/149) |
| Drop recall | 30.3% (50/165) | 32.9% (49/149) |
| False positive rate | 5.5% | 5.4% (8/149) |

### Sensitivity sweep and max-error scoring

Recall at the same ~5.4% false-positive rate, for several injection magnitudes (`docs/sensitivity_original.png`, `docs/sweep_original.csv`):

| Spike value | 1.2 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|
| Recall, mean-error score | 20.8% | 53.0% | 100% | 100% |
| Recall, max-error score | 18.8% | 45.0% | 100% | 100% |

| Drop value | -0.1 | -0.3 | -0.5 | -1.5 |
|---|---|---|---|---|
| Recall, mean-error score | 10.7% | 22.1% | 32.9% | 100% |
| Recall, max-error score | 6.0% | 17.4% | 24.8% | 100% |

Detection is reliable only for large anomalies (spike ≥ 2.0, drop ≤ -1.5). **Position-wise max-error scoring did not improve drop detection** (24.8% vs. 32.9% at -0.5); the earlier hypothesis that averaging was diluting the signal was wrong.

### Cleaned-data model

Trained with `--clean` (1,015 sequences): validation MSE 0.0599, threshold 0.155, 38 flagged (3.7%). Synthetic recall at (spike 2.0, drop -0.5): mean score 91.2% / 32.0%, max score 94.4% / 36.8%; false positive rate 5.6%. Max error is slightly ahead here, but with 149 validation sequences a gap this size is within sampling noise. Full table: `docs/sweep_clean.csv`.

### Explanation layer evaluation

For 20 randomly chosen flagged sequences (seed 42), each explanation was written two ways — a rule-based **template** and a **local LLM** (llama3.2:3b via Ollama, temperature 0) — and scored on four 1/0 criteria: direction correct, numbers correct and used correctly, plain language, no invented causes. A strict **grounding check** rejects LLM text with a wrong direction, placeholders, currency symbols, invented data-entry errors, or numbers not in the facts, and falls back to the template.

| | LLM, prompt v2 | LLM, prompt v3 | Template |
|---|---|---|---|
| All four criteria OK | 55% | 65% | 90% |
| Numbers correct and used correctly | 65% | 80% | 100% |
| No invented causes | 100% | 90% | 100% |
| **Shipped text** (LLM if it passes the check, else template) | 60% | **90%** | 90% |

Prompt v3 stopped sending the detector's internal score fields to the model; in v2 the model repeatedly mislabeled a severity ratio as "the threshold". The checker's precision on passed texts rose from 11/18 to 13/15, with no good text wrongly rejected. Raw outputs from the first two prompt versions: `docs/report_ollama_llama3.2-3b_v1.md`, `_v2.md`.

**Scoring caveats.** The scores in the table were produced by Claude acting as an LLM judge (`docs/explanation_scoring_claude*.csv`), not by independent humans. The student independently scored 12 of the prompt-v2 rows (`docs/explanation_scoring_human.csv`): **39/48 criterion judgements agreed (81%)** and 9/12 rows agreed on the overall verdict; plain language was the least consistent criterion (58%). The sheet is only weakly blind, since the template text is formulaic.

**Conclusion:** with a 3B model, the checked pipeline matches the template on correctness (90% vs. 90%); the LLM adds phrasing variety, not accuracy. A larger model was not evaluated.

---

## Discussion: Limitations and Use

**What works:** the detector catches large spikes and large drops, is unsupervised, and no longer overstates its validation performance. The explanation layer is safe by construction because every number is computed in code and unsupported text is rejected.

**Limitations:**
- *Data quality and realism:* the dataset contains corrupt amounts that the detector largely misses, and the amounts appear randomly generated rather than behavioral, so "unusual" often means "high spread within a window". For example, a flagged window can have its flagged position equal to the user's typical amount ("1.0x higher"). Conclusions about real spending behavior should not be drawn.
- *Detection range:* reliable only for large anomalies; drops of -0.5 or smaller are caught under 40% of the time.
- *Single feature:* only amount is used (no day of week, transaction type, or payment mode).
- *Small data:* 1,104 sequences (149 in validation); differences of a few percentage points are noise. Some categories have very few sequences.
- *No ground truth:* synthetic injection approximates, but does not replace, labeled anomalies.
- *Explanation evaluation:* 20 sequences, one small local model, LLM-judge scores (with an 81% agreement check on 12 rows), weakly blind. Not a statistically powered result.
- *Checker limits:* it verifies that quoted numbers exist and that direction is right, but cannot verify what a number is used for or catch every invented claim (e.g. "part of a pattern" passed).
- *Tax module:* a flat 25% placeholder set-aside on freelance income minus candidate expense categories (travel, education, utilities). It is not a tax calculation, and the data appear to be in rupees, so no jurisdiction-specific rules are applied.
- *Overlapping windows:* consecutive sequences share 4 of 5 values; the user-level split removes cross-set leakage, but sequences within a set remain correlated.

**Intended use:** an educational prototype. Results are not financial or tax advice.

---

## Original DSCI 410L Results (sequence-level split)

For reference, the results as submitted for the class (train 0.0459 / val 0.0491 at epoch 50; threshold 0.138; 64 of 1,104 flagged; spike recall 98.2%, drop recall 30.3%, false positive rate 5.5%). The weights are kept at `checkpoints/best_model_old_split.pt`. Plots for that model are in `notebooks/evaluation.ipynb` and `docs/qualitative_plot.png`:

![Qualitative anomaly inspection — original vs reconstructed sequences](docs/qualitative_plot.png)

---

## Generative AI Disclosure

### DSCI 410L class project

In accordance with the course syllabus policy on generative AI, the class project used Claude (Anthropic) as an assistant. It was used for code organization, docstrings and inline comments, debugging (tensor shape mismatches, category-normalization fallback logic), README drafting, suggesting the synthetic anomaly injection approach in response to instructor feedback, and running and verifying outputs. The core decisions — LSTM autoencoder, sequence generation, threshold methodology, and interpretation of results — were made by the student.

### Honors thesis extension (October 2026)

Claude (Anthropic, via Claude Code) was used substantially more in this phase. To state it plainly:

- **Student:** chose the thesis direction (explanation layer, tax note, free local model), wrote `get_user_split` in `dataloader.py` with Claude's guidance, ran training and evaluation commands, reviewed outputs, and independently scored 12 explanation rows.
- **Claude:** wrote the remaining new code — max-error scoring and the parameterized synthetic evaluation in `evaluate.py`, the `--clean` option, `sweep.py`, `explain.py` (facts, template, backends, grounding check), `tax.py`, `pipeline.py`, and `eval_explanations.py` — ran the experiments, identified the corrupt-amount issue and the checker's false positives, scored all 40 explanation texts as an LLM judge (Claude also wrote the checker and prompts it scored, so it is not an independent rater), and drafted this README.
- The explanation layer's language model is a locally run open model (llama3.2:3b); no financial data left the student's machine.
