# Freelance Finance Deep Learning Project

Sriraghav Bavineni · University of Oregon

Started as the **DSCI 410L (Introduction to Deep Learning)** class project; extended into a **Clark Honors College thesis** (defense Spring 2027). This README describes the current state. Re-examining the class-project pipeline during the thesis phase found and fixed several problems (data parsing, input scaling, the anomaly evaluation itself, and window grouping); the original numbers are kept at the end and marked **superseded**.

---

## Project Purpose

Many freelancers have irregular income and spending, which makes budgeting and planning difficult. This project asks whether deep learning can identify unusual transactions in a person's history and explain them in plain language, and whether a deep model is needed for that at all.

```
transactions → detector flags unusual windows → plain-language explanation → educational tax set-aside note
```

**What counts as unusual.** An unexpected amount, on the expense side or the income side: one transaction whose amount is far from the user's typical amount in that category (expenses) or typical income (all income sources pooled per user). Four cases: unusually high expense, unusually low expense, unusually high income, unusually low income. Not covered: timing, location, merchant, duplicates, and changes in recurring patterns. This is a point-anomaly definition in amount.

The main detector studied is an **LSTM sequence autoencoder** over windows of 5 consecutive transactions, compared with untrained one-line rules and an Isolation Forest. In the explanation layer, **every number is computed in code; a small local language model only phrases them**, and an automatic grounding check rejects text that contradicts the data (falling back to a fixed template).

Results are educational only and are not financial or tax advice.

---

## Datasets

Three datasets are used: the original BudgetWise data, real bank data (Berka), and a simulation.

### 1. BudgetWise Personal Finance Dataset (original)
Source: https://www.kaggle.com/datasets/mohammedarfathr/budgetwise-personal-finance-dataset. 15,900 rows, 150 users, 2021–2024; fields `transaction_id`, `user_id`, `date`, `transaction_type`, `category`, `amount`, `payment_mode`, `location`, `notes`. Location: `data/budgetwise_finance_dataset.csv`. It appears synthetic and is in rupees.

**Data quality (all handled in `dataloader.py`):**
- *Category spellings:* 46 raw spellings map to 13 canonical categories.
- *Mixed date formats:* `2023-04-25`, `08/05/2022` (always month/day/year), `31-12-23` and `21-09-2024` (always day-month). 3,713 dates have both fields ≤ 12, so their order is assumed from their format. The original loader parsed only ISO dates and dropped the rest.
- *Amounts with currency text:* `₹5070`, `Rs.828`, `5107 INR`; the original loader dropped these too.
- *Row accounting:* the original loader kept **4,977 of 15,900 rows (31.3%)**. Current pipeline: 14,896 rows have all required fields (1,004 do not); removing 843 duplicates leaves **14,053 (88.4%)**; dropping 373 corrupt amounts (75 × `999999999`, 83 × `999999`, 72 zeros, 143 negatives) leaves **13,680 (86.0%)**. 890 rows share a transaction id with a different transaction and are kept.
- *Windows touched by corrupt amounts:* 1,133 of 8,882 (12.8%) before they are dropped (`--clean` drops them).

**Windows.** Transactions are grouped by `(user, transaction type, category)`, sorted by date, and a window of 5 slides one step at a time. All of a user's income is pooled into one `income` history (income was split over five categories with about three transactions per user each, leaving only 144 income windows). After cleaning: **8,537 windows from all 150 users: 7,000 expense and 1,537 income.** (An earlier grouping by user and category only put income and expense rows of the category "other" in one history: 108 of 140 windows. `--legacy-grouping` and old checkpoints reproduce that.)

**No structure to learn.** Within a user and category the lag-1 autocorrelation of log-amount averages −0.07 (509 histories); the share of variance explained by user (ICC) is −0.004 for food, −0.004 for rent, −0.007 for travel; pooled income has ICC −0.001. Non-rent expense amounts never exceed 10,000 (99th percentile 9,744 to 9,932), while 35% of rent amounts do. Reproduce: `python -m freelance_finance_dl.structure`.

### 2. Berka Czech bank data (real)
Real, anonymized transactions of 4,500 accounts of a Czech bank, 1993–1998, in koruna (PKDD'99 Discovery Challenge; Berka & Sochorová). Only `trans.csv` is used, placed in `data/berka/` (gitignored). Credits are income; debits and withdrawals are expenses; categories come from the payment-kind symbol (household payment, insurance, loan payment, pension) or the operation (cash withdrawal, card withdrawal, transfer out, transfer in, cash deposit). **Excluded as bank-generated, not user behavior:** interest credited (183,114 rows), statement fees (155,832), penalty interest (1,577), 340,523 rows in all, leaving 715,797 (493,828 expense, 221,969 income) and **653,628 windows from all 4,500 accounts**. Strong habits, no sequence dependence: ICC 0.999 for household payments, 1.000 for insurance and loan payments, 0.835 for transfers out, 0.447 for cash withdrawals, 0.511 for pooled income; lag-1 autocorrelation 0.013 (expenses), −0.054 (income). Half of the windows come from histories whose variability is at the floor of 0.05 (standing payments).

### 3. Simulation with habits and labeled anomalies (synthetic, by the author)
`habitsim.py` generates 100 users over 3 years: salaried or freelance income, nearly constant rent, a fixed subscription, utilities with a yearly cycle, volatile groceries and dining. Four habit levels: **strong**, **default**, **weak**, **none** (independent draws, no habits). With 1% probability per transaction (4% for utilities) a labeled anomaly replaces the amount: **spike** (×2 to ×10), **drop** (÷2 to ÷10), **seasonal** (utility bill of the opposite season), **duplicate** (repeat of the previous grocery or dining amount). Designed by the author, so results show what is possible given such structure, not what real data do. Data are generated into `data/sim/` (gitignored) from seeds.

---

## Input features

- **min-max** (class project): each window scaled to [0, 1]; discards absolute size.
- **log-ratio:** log(amount / the user's typical amount for that history) (median of the user's valid amounts; ratios clipped to [0.001, 1000]).
- **variability-aware** (`robustz`): arcsinh(log-ratio / spread), where the spread of a history is 1.4826 × the median absolute deviation of its log-ratios (floor 0.05). An amount is measured in units of how much that history normally varies. (An earlier clipped version saturated 19.8% of Berka income windows and made rule thresholds degenerate; it was replaced by arcsinh and **all results using these features were re-run**.)

The typical amount and spread use the user's whole history (a mild look-ahead).

---

## Repository Structure

```text
freelance-finance-dl/
├── freelance_finance_dl/
│   ├── dataloader.py          # Parsing, cleaning, windows, three feature types, user-level split
│   ├── berka.py               # Berka loader
│   ├── habitsim.py            # Simulation with habits and labeled anomalies
│   ├── model.py, train.py     # LSTM autoencoder and training
│   ├── evaluate.py            # Scoring, threshold, raw-space anomaly injection
│   ├── sweep.py, crossval.py  # BudgetWise sweep and 5-fold user-level cross-validation (all / expense / income)
│   ├── baseline.py            # Untrained one-line rules on the BudgetWise folds
│   ├── cv_eval.py             # Cross-validation with injection for Berka (AE, rules, Isolation Forest)
│   ├── sim_eval.py            # Simulation study on labeled anomalies
│   ├── habit_eval.py          # Per-category injection test on clean simulated data
│   ├── structure.py           # Per-user / time structure checks (ICC, autocorrelation)
│   ├── explain.py, pipeline.py, eval_explanations.py, tax.py   # Explanation layer, report, evaluation, tax note
│   └── stats.py, plots.py     # Confidence intervals; thesis figures
├── checkpoints/                    # primary_logratio_l3 and others (see below)
├── docs/{figures,sweeps,explanation_eval,superseded}/
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
# BudgetWise: primary model, cross-validation, rules, structure checks, figures
python -m freelance_finance_dl.train --output-dir checkpoints/primary_logratio_l3 --latent-dim 3 --epochs 50 --clean
python -m freelance_finance_dl.crossval --config logratio_l3     # also logratio_l16, minmax_l3, robustz_l3
python -m freelance_finance_dl.crossval --summarize
python -m freelance_finance_dl.baseline            # and: baseline robustz
python -m freelance_finance_dl.structure
python -m freelance_finance_dl.plots

# Berka (put trans.csv in data/berka/)
python -m freelance_finance_dl.cv_eval --source berka --csv data/berka/trans.csv --features robustz --latent 16 --tag _berka_robustz_l16
python -m freelance_finance_dl.structure --source berka --csv data/berka/trans.csv

# Simulation: labeled-anomaly study (one process per seed) and per-category test
python -m freelance_finance_dl.sim_eval --seed 1        # seeds 1, 2, 3
python -m freelance_finance_dl.sim_eval --summarize
python -m freelance_finance_dl.habitsim --level default --seed 7 --anomaly-rate 0 --out data/sim/clean_default_s7.csv
python -m freelance_finance_dl.habit_eval --csv data/sim/clean_default_s7.csv --features robustz --latent 16 --tag _robustz_l16

# End-to-end report and explanation evaluation (BudgetWise)
python -m freelance_finance_dl.pipeline --top-n 10 --llm ollama
python -m freelance_finance_dl.eval_explanations --n 20 --tag _v8
```

Older checkpoints record how they were trained and reload with the right settings.

---

## Model

`TransactionAutoencoder`: a one-layer LSTM encoder (hidden size 64) over the 5 values; its final hidden state is projected to a latent vector; the decoder projects it back to 64, repeats it at each of 5 positions, runs a second LSTM, and outputs one value per position. 50,948 parameters (latent 3), 52,625 (latent 16). Adam (lr 0.001, batch 64), MSE loss, seed 42, CPU. A window's score is its mean squared reconstruction error; the flag threshold is the 95th percentile of held-out scores (per stream where noted). The three one-line **rules** score a window by its largest |feature| (direction unknown), its largest feature (assumes a spike) or its most negative feature (assumes a drop); the direction-aware rules use knowledge a deployed detector lacks and show a ceiling for one-line rules. The **Isolation Forest** uses 100 trees on the same features.

---

## Results

All injected-anomaly results change one raw amount by ×k or ÷k and then process the window like any other; chance equals the false-positive rate, about 5%. Recall values compare detectors; they are not absolute detection rates, because a multiplicative change to a small amount may stay in the normal range.

### BudgetWise: the class-project evaluation was invalid
The class project overwrote one *normalized* value with 2.0 or −0.5, which cannot occur in normalized data (98.2% recall, 100% after the user-level split). Realistic injection shows min-max models at or below chance (spike ×10: 1% for the class-project model, 2.1 ± 1.5% in cross-validation on the current data).

### BudgetWise: cross-validated detection (5 folds of 30 users; mean ± sd)

| Recall (%), all windows | Spike ×3 | ×5 | ×10 | Drop ÷3 | ÷5 | ÷10 |
|---|---|---|---|---|---|---|
| **Autoencoder, log-ratio, latent 3 (primary)** | 10.1 ± 1.3 | 20.3 ± 2.8 | **41.0 ± 3.7** | 7.3 ± 1.1 | 8.9 ± 0.9 | 11.5 ± 0.9 |
| Autoencoder, log-ratio, latent 16 | 9.1 ± 1.0 | 15.2 ± 3.0 | 32.9 ± 7.5 | 8.5 ± 1.1 | 12.9 ± 2.2 | 21.0 ± 3.8 |
| Autoencoder, variability-aware, latent 3 | 14.4 ± 2.7 | 23.3 ± 3.7 | 36.6 ± 3.9 | 7.6 ± 1.3 | 9.3 ± 2.0 | 11.6 ± 2.0 |
| Autoencoder, min-max, latent 3 | 2.7 ± 0.4 | 2.1 ± 0.5 | 2.1 ± 1.5 | 4.4 ± 0.4 | 4.3 ± 0.4 | 4.7 ± 0.4 |
| Rule \|z\| (direction unknown), log-ratio | 5.0 ± 0.2 | 7.4 ± 0.8 | 15.3 ± 1.6 | 8.3 ± 0.4 | 12.3 ± 1.0 | 23.5 ± 2.5 |
| Rule, spike known, log-ratio | 16.3 ± 2.8 | 36.9 ± 6.9 | **73.6 ± 4.1** | | | |
| Rule, drop known, log-ratio | | | | 8.3 ± 0.4 | 12.4 ± 0.9 | **23.6 ± 2.4** |
| Rule \|z\|, variability-aware | 5.7 ± 0.5 | 8.4 ± 1.0 | 17.6 ± 2.8 | 7.9 ± 0.7 | 12.3 ± 1.1 | 22.8 ± 3.1 |

By stream (thresholds within the stream), the primary autoencoder finds 43.6 ± 3.0% of expense spikes at ×10 (spike-known rule 76.0 ± 2.4%) and 24.4 ± 9.1% of income spikes. For income drops, a one-line rule catches **94.7 ± 1.4%** at ÷10 against **8.3 ± 0.9%** for the autoencoder (variability-aware drop rule: 64.1 ± 8.8%). On this data the variability-aware features make no consistent difference (every category is equally random). The primary model flags 383 of 8,537 windows (4.5%; 273 expense, 110 income), 55% of them rent windows because rent amounts are far more spread out.

### Berka (real bank data): 5-fold cross-validation over accounts (mean ± sd, %)
40,000 training windows per fold, 3,000 evaluated windows per stream per fold, threshold = 95th percentile of the stream's held-out scores.

| Expense stream | Spike ×1.5 | ×3 | ×10 | Drop ÷1.5 | ÷3 | ÷10 |
|---|---|---|---|---|---|---|
| Autoencoder, variability-aware, latent 16 | 14 ± 6 | 31 ± 5 | 51 ± 7 | 21 ± 15 | 31 ± 16 | 40 ± 14 |
| Autoencoder, variability-aware, latent 3 | 6 ± 0 | 11 ± 1 | 30 ± 5 | 8 ± 3 | 8 ± 1 | 15 ± 1 |
| Autoencoder, log-ratio, latent 16 | 5 ± 0 | 8 ± 1 | 27 ± 4 | 5 ± 0 | 8 ± 1 | 21 ± 5 |
| Isolation Forest, variability-aware | 6 ± 1 | 12 ± 7 | 21 ± 7 | 6 ± 0 | 15 ± 6 | 49 ± 7 |
| **Rule \|z\| (direction unknown), variability-aware** | **43 ± 1** | **45 ± 1** | **62 ± 2** | **43 ± 1** | **47 ± 1** | **64 ± 2** |
| Rule, direction known, variability-aware | 45 ± 1 | 59 ± 2 | 84 ± 1 | 44 ± 1 | 48 ± 1 | 69 ± 1 |
| Rule \|z\| (direction unknown), log-ratio | 5 ± 0 | 7 ± 0 | 40 ± 18 | 6 ± 0 | 10 ± 0 | 40 ± 18 |
| Rule, direction known, log-ratio | 7 ± 0 | 18 ± 1 | 86 ± 1 | 6 ± 1 | 11 ± 0 | 73 ± 2 |

| Income stream | Spike ×1.5 | ×3 | ×10 | Drop ÷1.5 | ÷3 | ÷10 |
|---|---|---|---|---|---|---|
| Autoencoder, variability-aware, latent 16 | 13 ± 3 | 24 ± 4 | 34 ± 4 | 11 ± 1 | 14 ± 1 | 19 ± 3 |
| Autoencoder, log-ratio, latent 16 | 5 ± 0 | 6 ± 1 | 21 ± 4 | 5 ± 0 | 7 ± 1 | 14 ± 1 |
| Isolation Forest, variability-aware | 11 ± 1 | 12 ± 1 | 13 ± 1 | 10 ± 1 | 13 ± 1 | 16 ± 1 |
| Rule \|z\| (direction unknown), variability-aware | 5 ± 0 | 5 ± 0 | 5 ± 1 | 6 ± 0 | 6 ± 0 | 9 ± 0 |
| Rule, direction known, variability-aware | 12 ± 1 | 70 ± 2 | 73 ± 2 | 6 ± 0 | 6 ± 0 | 9 ± 0 |
| Rule, direction known, log-ratio | 8 ± 1 | 71 ± 28 | 94 ± 0 | 5 ± 0 | 7 ± 0 | 12 ± 1 |

On real bank expenses, the variability-aware **one-line rule catches 43% of 50% changes** (standing payments) at a 5% false-positive rate, against 14% for the best autoencoder and 5% to 7% for the log-ratio versions. The autoencoder improves with a larger bottleneck (latent 16) and with variability-aware features but stays below the rule. Income is hard: unexpected low income is caught by no method above 19% (the best rule 12%), unlike on BudgetWise; high income is caught by direction-aware rules (73% to 94% at ×10). Figure: `docs/figures/berka_detection.png`.

### Simulation (labeled anomalies; mean over 3 data seeds)
Average precision (chance about 0.05) for windows containing a labeled anomaly:

| Detector | Strong habits | Default | Weak | No habits |
|---|---|---|---|---|
| Autoencoder, log-ratio, latent 16 | 0.70 | 0.33 | 0.11 | 0.23 |
| Autoencoder, variability-aware, latent 16 | 0.52 | 0.38 | 0.27 | 0.14 |
| Isolation Forest | 0.37 | 0.26 | 0.18 | 0.15 |
| Rule \|z\|, log-ratio | 0.78 | 0.63 | 0.15 | 0.40 |
| **Rule \|z\|, variability-aware** | **0.77** | **0.69** | **0.43** | 0.33 |

A one-line rule is best or tied at every habit level; the autoencoder's average precision falls as habits weaken. **Seasonal and duplicate anomalies, which break a pattern in a sequence without being large in amount, are not found by any detector** (AUROC 0.4 to 0.6, average precision 0.01 to 0.05), so the autoencoder did not learn the sequence structure even when the data contained it. Per-category injection test on clean simulated data (default habits; spike recall % at ×1.5 / ×3 / ×10, one global threshold): with log-ratio features, rent and subscription are caught 0 / 18 / 100% by the autoencoder and 0 / 30 / 100% (rent) and 0 / 0 / 100% (subscription) by the direction-unknown rule, while with variability-aware features both the autoencoder and the rule catch **100 / 100 / 100%** with 0% false alarms; volatile categories (groceries, dining) stay hard for small changes (13 / 61 / 100 and 10 / 35 / 92 for the autoencoder), where the rule does somewhat better. Figure: `docs/figures/simulation_levels.png`.

### Explanation layer evaluation (BudgetWise windows)
For 20 randomly chosen flagged windows per round, explanations were written by a rule-based **template** and a **local language model** (llama3.2:3b, temperature 0) and scored on four 1/0 criteria: direction correct, numbers correct and used correctly, plain language, no invented causes (all four = fully correct). A grounding check rejects text with placeholders, currency symbols, a wrong direction, income described as spending, an invented data-entry error, or numbers not in the facts, and falls back to the template.

| Round | Windows | Prompt | LLM fully correct (95% interval) | Template | Note |
|---|---|---|---|---|---|
| v2 | class-project detector's flags | v2 | 55% (34–74) | 90% | score ratio mislabeled "the threshold" |
| v3 | same 20 | v3 (hides score fields) | 65% (43–82) | 90% | currency symbol, wrong differences |
| v4 | earlier primary model (before grouping fix) | v3 | 95% (76–99) | 100% | |
| v5 | new primary model, income included | v3 | 70% (48–85) | 100% | **4 of 5 income texts said "spends"** |
| v6 | same 20 as v5 | v6 rewrite | checker passed 4/20, not hand-judged | | model added `$` in 16 of 20 |
| v7 | same 20 as v5 | v7 | **95% (76–99)** | 100% (84–100) | one invented cause missed by the checker |

With 20 windows per round, the v7 language-model result and the template are statistically indistinguishable. Most v7 texts omit the size of the difference, so they pass partly by saying less. A 3-billion-parameter model is brittle to wording (all prompts: `docs/explanation_eval/prompts.md`). **Scoring caveats:** scores are from Claude acting as an LLM judge (which also wrote the prompts and the checker); the author scored 12 round-v2 texts: **39/48 judgments agreed (81%)**, plain language least consistent (58%); the sheets are only weakly blind.

### Follow-up: improved context models (2026-11-13)

A forecast model (previous 12 transactions plus dates, category, and stream; predicts the next amount and its uncertainty) and a masked fill-in model were tested against the rules and the original autoencoder on the simulation (with new late-paycheck and double-charge anomalies) and on Berka. They **beat every rule on seasonal anomalies** (simulation AUROC 0.93–0.94 vs 0.55–0.56) and on **Berka income** (×1.5 spike 63 ± 4% vs 4 ± 0%; halving 67 ± 4% vs 6 ± 1%), **tie the rule** on expenses and simple amount anomalies, and **lose to a simple timing rule** on late payments. Details, caveats, and tables: [docs/context_models.md](docs/context_models.md).

---

## Limitations
- *The learned detector is not justified here:* on all three datasets a one-line rule using the same features matches or beats the autoencoder (BudgetWise spike ×10: 73.6% vs 41.0%; Berka expenses ×1.5: 43% vs 14%; simulation: best or tied at every habit level).
- *Real data has habits but no sequence dependence* (Berka lag-1 autocorrelation about 0); the simulation's pattern-breaking anomalies were not detected by any model.
- *Injected and simulated anomalies are proxies:* multiplicative changes to small amounts may stay in range; thresholds set by percentile fix the false-positive rate; the simulation is circular by design (the author designed the habits).
- *Feature sensitivity:* autoencoder results depend on how inputs are compressed (clipping vs arcsinh changed its scores on the simulation); the rules' rankings do not.
- *Berka:* a single bank in the 1990s, amounts in koruna, bank-generated entries excluded by the author's judgment, evaluation on samples of 3,000 windows per stream per fold, and income spreads are heavy-tailed so variability-aware income scores saturate toward chance for direction-unknown rules.
- *Statistics:* five folds, rough standard deviations; explanation rounds use 20 windows; judge scores are LLM-judged.
- *Pipeline:* the explanation layer is demonstrated on the autoencoder's flags; a rule-based detector could replace it but was not run through the explanation evaluation.
- *Tax module:* a flat 25% placeholder; not tax advice; not evaluated. *Notebooks* are from the class project and not re-run.

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
- **Student:** chose the thesis direction and the definition of "unusual" (expenses and income), wrote `get_user_split` with Claude's guidance, ran commands, reviewed outputs, decided to fix the problems Claude reported, supplied the Berka data, chose to run the model on Berka and on a simulated dataset, and independently scored 12 explanation texts.
- **Claude:** wrote the remaining code (including the simulator), ran the experiments, **identified** the data-loss, scaling, evaluation-design, grouping, and feature-saturation problems, scored all explanation texts as an LLM judge (it also wrote the prompts and the checker, so it is not independent), wrote the README, and on 2026-11-12 drafted the first full draft of the thesis chapters in `thesis/`, which the student rewrites.
- The explanation layer's language model is a local open model (llama3.2:3b), so the explanation layer sends nothing to a third party. The development process is different: working with Claude through Claude Code sent code, command output, and excerpts of the datasets (including transaction rows) to Anthropic. The BudgetWise data and the simulation are synthetic; Berka is real but anonymized and public.
