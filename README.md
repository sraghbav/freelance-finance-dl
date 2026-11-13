# Freelance Finance Deep Learning Project

Sriraghav Bavineni · University of Oregon

Started as the **DSCI 410L (Introduction to Deep Learning)** class project; extended into a **Clark Honors College thesis** (defense Spring 2027). This README describes the current state. Re-examining the class-project pipeline during the thesis phase found and fixed several problems (data parsing, input scaling, the anomaly evaluation itself, and window grouping); the original numbers are kept at the end and marked **superseded**.

---

## Project Purpose

Many freelancers have irregular income and spending, which makes budgeting and planning difficult. This project asks whether deep learning can identify unusual transactions in a person's history and explain them in plain language.

```
transactions → LSTM autoencoder flags unusual windows → plain-language explanation → educational tax set-aside note
```

**What counts as unusual.** An unexpected amount, on the expense side or the income side: one transaction whose amount is far from the user's typical amount in that category (expenses) or typical income (all income sources pooled per user). Four cases: unusually high expense, unusually low expense, unusually high income, unusually low income. Not covered: timing, location, merchant, duplicates, and changes in recurring patterns. This is a point-anomaly definition in amount.

The detector is an **LSTM sequence autoencoder** over windows of 5 consecutive transactions. It needs no labeled anomalies. In the explanation layer, **every number is computed in code; a small local language model only phrases them**, and an automatic grounding check rejects text that contradicts the data (falling back to a fixed template).

Results are educational only and are not financial or tax advice.

---

## Datasets

### BudgetWise Personal Finance Dataset (main)
Source: https://www.kaggle.com/datasets/mohammedarfathr/budgetwise-personal-finance-dataset. 15,900 rows, 150 users, 2021–2024; fields `transaction_id`, `user_id`, `date`, `transaction_type`, `category`, `amount`, `payment_mode`, `location`, `notes`. Location: `data/budgetwise_finance_dataset.csv`.

**Data quality (all handled in `dataloader.py`):**
- *Category spellings:* 46 raw spellings map to 13 canonical categories.
- *Mixed date formats:* `2023-04-25`, `08/05/2022` (always month/day/year), `31-12-23` and `21-09-2024` (always day-month). 3,713 dates have both fields ≤ 12, so their order is assumed from their format. The original loader parsed only ISO dates and dropped the rest.
- *Amounts with currency text:* `₹5070`, `Rs.828`, `5107 INR` (the data is in **rupees**); the original loader dropped these too.
- *Row accounting:* the original loader kept **4,977 of 15,900 rows (31.3%)**. Current pipeline: 14,896 rows have all required fields (1,004 do not); removing 843 duplicates leaves **14,053 (88.4%)**; dropping 373 corrupt amounts (75 × `999999999`, 83 × `999999`, 72 zeros, 143 negatives) leaves **13,680 (86.0%)**. 890 rows share a transaction id with a different transaction and are kept.
- *Windows touched by corrupt amounts:* 1,133 of 8,882 (12.8%) before they are dropped (`--clean` drops them).

**Windows.** Transactions are grouped by `(user, transaction type, category)`, sorted by date, and a window of 5 slides one step at a time. All of a user's income, from every source, is pooled into one `income` history, because income was split over five categories with about three transactions per user each, which left only 144 income windows (2%). After cleaning: **8,537 windows from all 150 users: 7,000 expense and 1,537 income.** Food, rent, and travel are 61% of windows. (An earlier version grouped by user and category only, which put income and expense rows of the "other" category in one history: 108 of 140 windows. `--legacy-grouping` and old checkpoints reproduce that behavior.)

**No structure to learn.** Within a user and category the lag-1 autocorrelation of log-amount averages −0.07 (509 histories); the share of variance explained by user (ICC) is −0.004 for food, −0.004 for rent, −0.007 for travel; pooled income has ICC −0.001 and autocorrelation −0.11. Non-rent expense amounts never exceed 10,000 (99th percentile 9,744 to 9,932), while 35% of rent amounts do (99th percentile 48,986). Income amounts: median 46,713, 1st percentile 10,712, 4.6% above 100,000. The data look synthetic. Reproduce: `python -m freelance_finance_dl.structure`.

### Sparkov simulated card transactions (second dataset, real fraud labels)
Kaggle `kartik2112/fraud-detection` (`fraudTrain.csv` + `fraudTest.csv`, placed in `data/sparkov/`, gitignored). 1,852,394 transactions, 999 customers, 14 categories, 2019–2020, fraud rate 0.58% (train file) and 0.39% (test file); simulated, generated from per-customer profiles. Structure: lag-1 autocorrelation 0.019; ICC 0.71 for `grocery_pos`, 0.09 for `gas_transport` and `home`, so some categories have a persistent per-user level and none has time structure. 1,800,686 windows after cleaning.

---

## Input features

Each amount is converted to **log(amount / the user's typical amount for that history)** (the median of the user's valid amounts in that history; ratios clipped to [0.001, 1000]). 0 means typical, +1.1 about 3× typical, −1.1 about a third. This keeps absolute size, unlike the class project's per-window min-max scaling (`--features minmax`). The typical amount uses the user's whole history, including later transactions and validation users (a mild look-ahead).

---

## Repository Structure

```text
freelance-finance-dl/
├── freelance_finance_dl/
│   ├── dataloader.py          # Parsing, cleaning, windows, features, user-level split
│   ├── model.py               # LSTM sequence autoencoder
│   ├── train.py               # Training loop with checkpointing
│   ├── evaluate.py            # Scoring, threshold, raw-space anomaly injection
│   ├── sweep.py               # Single-split injection sweep and plots
│   ├── crossval.py            # 5-fold user-level cross-validation (all / expense / income)
│   ├── baseline.py            # Untrained one-line rule baselines on the same folds
│   ├── structure.py           # Per-user / time structure checks (ICC, autocorrelation)
│   ├── sparkov.py, fraud_eval.py   # Second dataset: loader and real-fraud-label evaluation
│   ├── explain.py             # Facts, template, LLM backends, grounding check
│   ├── tax.py                 # Educational quarterly set-aside estimate
│   ├── pipeline.py            # End-to-end report: flag -> explain -> tax note
│   ├── eval_explanations.py   # Explanation quality evaluation
│   └── stats.py, plots.py     # Confidence intervals; thesis figures
├── checkpoints/
│   ├── primary_logratio_l3/        # PRIMARY: log-ratio, latent 3, cleaned, type-aware windows
│   ├── ablation_logratio_l16/      # latent-size ablation (same settings)
│   ├── prefix_logratio_l3/, prefix_logratio_l16/   # same, trained before the grouping fix
│   ├── robust_minmax/, robust_minmax_clean/        # robust parsing, min-max features
│   └── legacy_minmax_*/            # class-project models
├── docs/
│   ├── figures/                    # data-loss chart, flagged examples, detector vs rules, by stream
│   ├── sweeps/                     # cross-validation, baselines, Sparkov results (CSV/PNG)
│   ├── explanation_eval/           # scoring sheets, keys, summaries (v2 to v7) and prompts.md
│   └── superseded/                 # class-project figures
├── thesis/                         # abstract, chapters 1-6, appendices, references (Claude first draft, to be rewritten)
└── notebooks/                      # original class notebooks (NOT updated; stale)
```

---

## Installation

```bash
git clone https://github.com/sraghbav/freelance-finance-dl.git
cd freelance-finance-dl
pip install -e . && pip install -r requirements.txt
```

**Optional, for language-model explanations (free, local):** install [Ollama](https://ollama.com), `ollama pull llama3.2:3b`, keep `ollama serve` running. The Anthropic backend (`--llm anthropic`) exists but was **not tested**.

---

## How to Run

```bash
# Primary model
python -m freelance_finance_dl.train --output-dir checkpoints/primary_logratio_l3 --latent-dim 3 --epochs 50 --clean
# Class-project behavior: add --legacy-parsing --legacy-grouping --features minmax

# Cross-validation (each config, then summary), untrained baselines, structure checks, figures
python -m freelance_finance_dl.crossval --config logratio_l3     # also logratio_l16, minmax_l3
python -m freelance_finance_dl.crossval --summarize
python -m freelance_finance_dl.baseline
python -m freelance_finance_dl.structure
python -m freelance_finance_dl.plots

# Second dataset with real fraud labels (add --max-users 400 if memory is tight)
python -m freelance_finance_dl.fraud_eval --csv data/sparkov/fraudTrain.csv data/sparkov/fraudTest.csv --tag _l3_s42

# End-to-end report and explanation evaluation
python -m freelance_finance_dl.pipeline --top-n 10 --llm ollama
python -m freelance_finance_dl.eval_explanations --n 20 --tag _v8
python -m freelance_finance_dl.eval_explanations --summarize --tag _v8 --scores <scored.csv>
```

Older checkpoints record how they were trained and reload with the right settings automatically.

---

## Model

`TransactionAutoencoder` (`model.py`): a one-layer LSTM encoder (hidden size 64) over the 5 values; its final hidden state is projected to a latent vector (size 3 in the primary model); the decoder projects it back to 64, repeats it at each of 5 positions, runs a second LSTM, and outputs one value per position. 50,948 parameters (latent 3), 52,625 (latent 16). Trained with Adam (lr 0.001, batch 64), MSE loss, 50 epochs, seed 42, on CPU. A latent smaller than the 5 inputs matters: with latent 16 the network nearly copies its input on BudgetWise (validation error 0.0004 against 0.144). A window's score is its mean squared reconstruction error; the flag threshold is the 95th percentile of held-out scores.

---

## Results

### Why the original evaluation was wrong
The class project tested the detector by overwriting one *normalized* value with 2.0 or −0.5. Normalized data only ranges from 0 to 1, so those anomalies could never occur, and the 98.2% spike recall (100% after the user-level split) only showed that the model detects impossible values. A realistic test changes a **raw** amount (one charge ×k or ÷k, k = 1.5, 2, 3, 5, 10) and then processes the window like any other (`evaluate.raw_injection_eval`). Chance level is the false-positive rate, about 5%. Because a multiplicative change to a small amount may stay in the normal range, recall values compare detectors; they are not absolute detection rates.

**Min-max scaling, recall (%) at a ~5% false-positive rate:**

| Evaluation | Spike ×3 | ×5 | ×10 | Drop ÷3 | ÷5 | ÷10 |
|---|---|---|---|---|---|---|
| Class-project model, one split (127 windows, FPR 6.3%) | 8 | 2 | 1 | 4 | 3 | 6 |
| Robust parsing, old grouping, one split (1,070 windows) | 2 | 2 | 1 | 4 | 5 | 4 |
| Current data, 5-fold cross-validation, latent 3 | 2.7 ± 0.4 | 2.1 ± 0.5 | 2.1 ± 1.5 | 4.4 ± 0.4 | 4.3 ± 0.4 | 4.7 ± 0.4 |

Every min-max configuration is at or below chance.

### Cross-validated detection, all windows
5 folds of 30 users; fresh model per fold; mean ± sd across folds (`docs/sweeps/crossval_summary.csv`, `baseline_rule_summary.csv`):

| Recall (%) | Spike ×3 | ×5 | ×10 | Drop ÷3 | ÷5 | ÷10 |
|---|---|---|---|---|---|---|
| **Autoencoder, latent 3 (primary)** | 10.1 ± 1.3 | 20.3 ± 2.8 | **41.0 ± 3.7** | 7.3 ± 1.1 | 8.9 ± 0.9 | 11.5 ± 0.9 |
| Autoencoder, latent 16 | 9.1 ± 1.0 | 15.2 ± 3.0 | 32.9 ± 7.5 | 8.5 ± 1.1 | 12.9 ± 2.2 | 21.0 ± 3.8 |
| Rule: largest \|log-ratio\| (direction unknown) | 5.0 ± 0.2 | 7.4 ± 0.8 | 15.3 ± 1.6 | 8.3 ± 0.4 | 12.3 ± 1.0 | 23.5 ± 2.5 |
| Rule: largest log-ratio (knows it is a spike) | 16.3 ± 2.8 | 36.9 ± 6.9 | **73.6 ± 4.1** | | | |
| Rule: most negative log-ratio (knows it is a drop) | | | | 8.3 ± 0.4 | 12.4 ± 0.9 | **23.6 ± 2.4** |

The direction-aware rules use knowledge a deployed detector lacks, but they show a one-line rule matches or beats the autoencoder. Max-error scoring is equivalent to mean-error (single split, spike ×10: 44.6% vs 42.1%).

### Expenses versus income
Thresholds are calibrated within each stream (`docs/figures/detector_by_stream.png`):

| Recall (%) | Spike ×5 | ×10 | Drop ÷5 | ÷10 |
|---|---|---|---|---|
| **Expense** (7,000 windows): autoencoder, latent 3 | 23.0 ± 3.5 | 43.6 ± 3.0 | 9.5 ± 1.0 | 12.0 ± 1.2 |
| Expense: rule, spike known / drop known | 42.8 ± 5.2 | 76.0 ± 2.4 | 11.9 ± 1.0 | 21.8 ± 1.3 |
| **Income** (1,537 windows): autoencoder, latent 3 | 9.9 ± 4.8 | 24.4 ± 9.1 | 6.9 ± 1.4 | 8.3 ± 0.9 |
| Income: autoencoder, latent 16 | 10.5 ± 3.0 | 23.2 ± 8.2 | 10.4 ± 4.3 | 16.3 ± 8.5 |
| Income: rule, spike known / drop known | 8.0 ± 1.4 | 24.5 ± 10.1 | **59.7 ± 4.9** | **94.7 ± 1.4** |

For income, an unexpectedly **low** payment is easy for a one-line rule (94.7% at ÷10) and almost invisible to the autoencoder (8.3%), likely because income rarely falls below about 10,000, so a tenfold drop leaves the normal range, a boundary the rule uses directly (a hypothesis, not tested). Income spikes are hard for every method (about 24% at ×10, large variance): 4.6% of income amounts already exceed 100,000. Income has about 300 windows per fold, so these estimates are noisier.

### The primary model
Latent 3, 8,537 windows, split by user (128 train / 22 validation users; 7,289 / 1,248 windows). Validation error 0.144; threshold 0.573; **383 of 8,537 windows flagged (4.5%)**: 273 of 7,000 expense (3.9%) and 110 of 1,537 income (7.2%). Flags are concentrated in rent (211 of 383, 55%, against 20% of windows) because rent amounts are far more spread out. Flagged windows have a larger deviation from the user's typical amount (median largest |log-ratio| 2.16 against 1.31). One unusual transaction can appear in up to five flagged windows.

### Real fraud labels (Sparkov)
Trained on 150,000 of the 1.4 million training windows (15 epochs); evaluated on 390,505 windows of 199 held-out customers (5,358 contain a fraudulent transaction, base rate 1.4%). AUROC (0.5 = chance), average precision (chance 0.014), and fraud windows caught when the top 5% of windows are flagged:

| Score | AUROC | Avg. precision | Recall at 5% flagged |
|---|---|---|---|
| Autoencoder, latent 3 | 0.638 | 0.025 | 13.2% |
| Autoencoder, latent 3, 4× more training windows | 0.630 | 0.026 | 12.5% |
| Autoencoder, latent 3, different held-out customers (seed 7) | 0.653 | 0.026 | 12.7% |
| Autoencoder, latent 16 | 0.708 | 0.070 | 28.5% |
| Rule: largest \|log-ratio\| | **0.732** | **0.070** | **37.2%** |
| Rule: largest log-ratio | 0.729 | 0.072 | 37.1% |
| Rule: most negative log-ratio | 0.466 | 0.013 | 2.7% |

On real labels the best autoencoder (latent 16) ties the one-line rule on average precision but is lower on AUROC and recall; latent 3 is clearly worse and does not improve with more training. Only the amount is used; real fraud also depends on category, time, and location.

### Explanation layer evaluation
For 20 randomly chosen flagged windows per round, explanations were written by a rule-based **template** and a **local language model** (llama3.2:3b, temperature 0) and scored on four 1/0 criteria: direction correct, numbers correct and used correctly, plain language, no invented causes (all four = fully correct). A grounding check rejects text with placeholders, currency symbols, a wrong direction, income described as spending, an invented data-entry error, or numbers not in the facts, and falls back to the template.

| Round | Windows | Prompt | LLM fully correct (95% interval) | Template | Note |
|---|---|---|---|---|---|
| v2 | class-project detector's flags | v2 | 55% (34–74) | 90% | score ratio mislabeled "the threshold" |
| v3 | same 20 | v3 (hides score fields) | 65% (43–82) | 90% | currency symbol, wrong differences |
| v4 | earlier primary model (before grouping fix) | v3 | 95% (76–99) | 100% | |
| v5 | new primary model, income included | v3 | 70% (48–85) | 100% | **4 of 5 income texts said "spends"** |
| v6 | same 20 as v5 | v6 rewrite | checker passed 4/20, not hand-judged | | model added `$` in 16 of 20 |
| v7 | same 20 as v5 | v7 | **95% (76–99)** | 100% (84–100) | all but one text correct; one invented cause missed by the checker |

With 20 windows per round, the v7 language-model result and the template are statistically indistinguishable. Most v7 texts say only "higher/lower" and omit the size of the difference, so they pass partly by saying less; the template states the multiple. A 3-billion-parameter model is brittle: a small prompt change moved the checker pass rate from 95% to 20% (all prompts in `docs/explanation_eval/prompts.md`).

**Scoring caveats.** Scores are from Claude acting as an LLM judge, not independent humans; Claude also wrote the prompts and the checker. The author independently scored 12 of the round-v2 texts: **39/48 criterion judgments agreed (81%)**, 9/12 on the overall verdict, plain language least consistent (58%) (`docs/explanation_eval/explanation_scoring_human_v2.csv`). This was not repeated for later rounds. The sheets are only weakly blind.

---

## Limitations
- *Detection is weak* and the data are synthetic: about 41% of 10× spikes and 12% of 10× drops at a 5% false-positive rate; a one-line rule is as good or better on BudgetWise and Sparkov.
- *No structure to learn:* BudgetWise has no per-user or time structure; Sparkov has per-user levels in some categories only.
- *Injected anomalies* are a proxy; multiplicative changes to small amounts may stay in range; recall compares detectors only. Thresholds set by percentile fix the false-positive rate by construction.
- *Statistics:* five folds, so standard deviations are rough; income has about 300 windows per fold; explanation rounds use 20 windows.
- *Amount only:* no dates, merchant, location, or payment mode; Sparkov fraud depends on those.
- *Scaling uses full history,* including validation users.
- *Pooled income* treats all income sources as one stream; no recurring-income structure exists in this data to test habits.
- *Judge and checker:* LLM-judged scores; the checker verifies values and direction, not meaning (it missed "the user may have made a mistake when entering the amount").
- *Repeated flags:* overlapping windows duplicate flags; 55% of flags are rent windows.
- *Tax module:* a flat 25% placeholder on freelance income minus candidate expenses (travel, education, utilities); not tax advice; the data are in rupees. Not evaluated.
- *Notebooks* are from the class project and not re-run.

**Intended use:** an educational prototype; not financial or tax advice.

---

## Original DSCI 410L Results (superseded)

Original loader (4,977 rows, 1,104 windows), min-max features, out-of-range injection, so the recall figures do not measure realistic detection. Sequence-level split: validation error 0.0491, 64 windows flagged (5.8%), spike/drop recall 98.2% / 30.3%. User-level split: 0.0577, 52 flagged (4.7%), 100% / 32.9%. Weights in `checkpoints/legacy_minmax_*`; figures in `docs/superseded/`.

![Qualitative anomaly inspection from the class project](docs/superseded/qualitative_plot.png)

---

## Generative AI Disclosure

### DSCI 410L class project
Claude (Anthropic) was used as an assistant for code organization, docstrings and comments, debugging, drafting the README, suggesting the synthetic-injection evaluation, and running outputs. The core decisions were the student's.

### Honors thesis extension (August to November 2026)
- **Student:** chose the thesis direction and the definition of "unusual" (expenses and income), wrote `get_user_split` with Claude's guidance, ran commands, reviewed outputs, decided to fix the problems Claude reported, downloaded the second dataset, and independently scored 12 explanation texts.
- **Claude:** wrote the remaining code, ran the experiments, **identified** the data-loss, scaling, evaluation-design, and grouping problems, scored all explanation texts as an LLM judge (it also wrote the prompts and the checker, so it is not independent), wrote the README, and on 2026-11-12 drafted the first full draft of the thesis chapters in `thesis/`, which the student rewrites.
- The explanation layer's language model is a local open model (llama3.2:3b), so the explanation layer sends nothing to a third party. The development process is different: working with Claude through Claude Code sent code, command output, and excerpts of the datasets (including transaction rows) to Anthropic. Both datasets are public and simulated or synthetic-looking.
