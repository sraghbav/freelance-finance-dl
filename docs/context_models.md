# Context models: trying to improve the detector (2026-11-13)

This note records an extra experiment run after the main study. The goal was to make the model
better at the things the window-of-5 autoencoder could not do. Code: `freelance_finance_dl/context_model.py`
(models) and `freelance_finance_dl/context_eval.py` (evaluation). Raw results: `docs/sweeps/ctx_*.csv`.
All tables below are generated from those CSV files.

## What changed (the six improvements)

1. **Timing inputs.** Each transaction carries the days since the previous transaction in its stream,
   the day of the month, and the month of the year.
2. **Longer context.** The model reads the previous 12 transactions in the stream, not a window of 5.
3. **Prediction instead of reconstruction.** Two models:
   - *forecast*: reads the previous 12 transactions and predicts the next amount **and how uncertain it
     is** (a mean and a standard deviation), plus the gap until it arrives. Score = standardized
     residual r = (actual − predicted) / predicted sd.
   - *masked*: reads 6 transactions before and 6 after with the middle amount hidden, and predicts it.
     Neither model can copy the amount it is scoring, which was the autoencoder's problem.
4. **Category, stream, and direction.** Category and expense/income are inputs; scores are signed, so a
   spike, a drop, and a late arrival can each be scored in its own direction.
5. **Hybrid.** The rule and the model scores are each converted to a tail probability (calibrated on
   the training users, separately for expense and income) and combined by taking the most extreme one.
6. **Data with habits.** Trained and tested on the simulation (all four habit levels, 3 seeds) and on
   the real Berka bank data (5-fold user-level cross-validation), not BudgetWise.

Amounts use the same variability-aware feature as the main study, so differences come from the model,
not the scaling. Fair baselines were added for timing: **rule timing** = how unusual the gap since the
previous transaction is, relative to that stream's own gaps (median and MAD), the timing analog of the
amount rule.

**New anomaly types in the simulation** (`habitsim.py --timing`, drawn from a separate random generator
so the rest of the data is identical to the main study's files): *late* (a salaried paycheck arrives
7–20 days late) and *double_charge* (a grocery, dining, or subscription charge repeated 0–3 days later).

**Unit of evaluation.** Each transaction is scored (the main study scored windows of 5). The original
autoencoder was retrained here on the same streams and scored per transaction (its error at that
position, averaged over the windows that contain it), so all methods are compared on the same items.
Window-level results are also in `ctx_sim_summary.csv` (unit = window) and lead to the same conclusions.

## Main findings

1. **The improved model clearly beats every rule on seasonal anomalies.** A utility bill at the opposite
   season's amount: forecast AUROC 0.93–0.94 and recall 82–89% at strong/default habits, against the
   amount rule at 0.55–0.56 and 0% recall and the old autoencoder at 0.53–0.60 (Table S3). This is the
   first case in the project where a learned sequence model does something a rule cannot. Caveat:
   only 25 seasonal anomalies in total (about 8 per held-out set), so the seed-to-seed spread is large.
2. **On real bank income, the improved model is far better than the rule.** Berka pooled income: a ×1.5
   spike is caught 63 ± 4% of the time by the forecast model against 4 ± 0% for the rule; a halving
   (drop ×2) 67 ± 4% against 6 ± 1% (direction-known: 74 ± 3% against 31 ± 33%). The main study found no
   method above 19% for low income on Berka, so **this changes finding 5 of the thesis**. A likely
   reason (not tested) is that pooled income mixes sources (pension, transfers, deposits) and the model
   uses day of month and gaps to tell which source a payment is from, while the rule judges every
   payment against one median for the whole pooled stream.
3. **On expenses and on simple amount anomalies, the model only ties the rule.** Berka expenses: forecast
   47 ± 1% vs rule 45 ± 1% at ×1.5 spike, 86 ± 2% vs 84 ± 1% at ×10; direction-known versions identical.
   Simulation, all amount anomalies: AUROC 0.91–0.92 vs 0.89 at strong/default habits, equal at weak/none
   (Tables S1–S2). Both are far ahead of the old autoencoder (e.g. Berka expense ×10 drop: 33 ± 25%).
4. **Timing: the simple timing rule is as good or better.** Late paycheck in the simulation: rule AUROC
   0.93 vs forecast 0.82–0.86; double charge: 0.71 vs 0.71–0.73 (Tables S4–S5). On Berka, with the
   direction known, the rule catches more late arrivals at short delays (income +7 days: 69 ± 3% vs
   43 ± 13%), and the gap narrows by +30 days (95 ± 1% vs 90 ± 1%). The old autoencoder cannot see timing at all (5%, i.e. chance).
5. **The hybrid did not help.** Taking the most extreme of several calibrated scores spends the 5%
   false-alarm budget across all of them, so it was usually below the best single score, and on Berka
   income it was dragged down by the rule's heavy tails.
6. **Still not found:** a repeated amount with normal timing (Table S6, AUROC about 0.6 for every method).
   This is not unusual in amount or in timing, so no method here should be expected to find it.

**Bottom line.** "Is a deep model needed?" now has a more precise answer: *not for a single amount that
is too large or too small relative to a steady history* (a rule ties it), *yes for anomalies that depend
on context*: amounts that are wrong for the season, and income streams that mix several sources. The
old window-of-5 autoencoder was the wrong design for those; a model that predicts the next amount with
its uncertainty, using dates and a longer history, handles them.

## Limitations

- One architecture and one set of hyperparameters (LSTM, hidden 64, 10 epochs on the simulation, 6 on
  Berka, at most 60,000 / 150,000 training transactions); no tuning.
- Seasonal and late anomalies are few in the simulation (25 and 70 in total across seeds).
- The simulation's seasonal cycle has the same phase for every user, which makes it easy to learn.
- Berka anomalies are injected, not real; late arrival is injected by moving one transaction's date.
- Hybrid calibration uses in-sample training scores.

## Reproduce

```bash
python -m freelance_finance_dl.context_eval sim --seed 1 --timing    # and --seed 2, 3; and without --timing
python -m freelance_finance_dl.context_eval sim-summary
python -m freelance_finance_dl.context_eval berka --csv data/berka/trans.csv
```

The simulation runs take about 1 minute per level and seed; Berka about 30 minutes on a laptop CPU.

## Tables

Simulation: mean ± sd over 3 seeds, per transaction, held-out 20% of users. Berka: mean ± sd over 5
folds, 3,000 held-out transactions per stream and fold, threshold at the 95th percentile of clean
held-out scores (false-positive rate 3.7–5.1% on average for every method).

### S1. Simulation, amount anomalies only (same data as the main study): all anomalies, AUROC

| score | strong | default | weak | none |
|---|---|---|---|---|
| rule \|z\| | 0.89 ± 0.02 | 0.89 ± 0.02 | 0.83 ± 0.02 | 0.81 ± 0.03 |
| autoencoder (window 5, latent 16) | 0.84 ± 0.02 | 0.78 ± 0.09 | 0.73 ± 0.05 | 0.70 ± 0.02 |
| forecast \|r\| | 0.92 ± 0.01 | 0.91 ± 0.01 | 0.83 ± 0.02 | 0.81 ± 0.03 |
| masked \|r\| | 0.92 ± 0.01 | 0.91 ± 0.01 | 0.84 ± 0.02 | 0.81 ± 0.03 |
| hybrid rule+forecast | 0.91 ± 0.02 | 0.90 ± 0.01 | 0.80 ± 0.03 | 0.78 ± 0.02 |

Positives (total over 3 seeds): default 394, none 343, strong 394, weak 394

### S2. Same runs, average precision

| score | strong | default | weak | none |
|---|---|---|---|---|
| rule \|z\| | 0.74 ± 0.00 | 0.66 ± 0.02 | 0.40 ± 0.01 | 0.29 ± 0.05 |
| autoencoder (window 5, latent 16) | 0.28 ± 0.04 | 0.25 ± 0.09 | 0.18 ± 0.05 | 0.10 ± 0.01 |
| forecast \|r\| | 0.80 ± 0.01 | 0.71 ± 0.02 | 0.40 ± 0.01 | 0.27 ± 0.06 |
| masked \|r\| | 0.80 ± 0.01 | 0.71 ± 0.02 | 0.40 ± 0.01 | 0.27 ± 0.06 |
| hybrid rule+forecast | 0.42 ± 0.01 | 0.36 ± 0.04 | 0.22 ± 0.03 | 0.11 ± 0.02 |

Positives (total over 3 seeds): default 394, none 343, strong 394, weak 394

### S3. Seasonal anomalies only (utilities at the opposite season's amount), AUROC

| score | strong | default | weak |
|---|---|---|---|
| rule \|z\| | 0.56 ± 0.07 | 0.55 ± 0.08 | 0.55 ± 0.09 |
| autoencoder (window 5, latent 16) | 0.53 ± 0.12 | 0.60 ± 0.05 | 0.65 ± 0.04 |
| forecast \|r\| | 0.94 ± 0.06 | 0.93 ± 0.07 | 0.70 ± 0.04 |
| masked \|r\| | 0.94 ± 0.05 | 0.94 ± 0.07 | 0.72 ± 0.03 |
| hybrid rule+forecast | 0.93 ± 0.06 | 0.90 ± 0.09 | 0.62 ± 0.13 |

Positives (total over 3 seeds): default 25, strong 25, weak 25

### S4. Late paycheck (data with timing anomalies), AUROC

| score | strong | default | weak |
|---|---|---|---|
| rule timing \|gap z\| | 0.93 ± 0.04 | 0.93 ± 0.04 | 0.93 ± 0.04 |
| forecast timing \|r_gap\| | 0.86 ± 0.03 | 0.82 ± 0.04 | 0.84 ± 0.01 |
| autoencoder (window 5, latent 16) | 0.27 ± 0.08 | 0.35 ± 0.05 | 0.41 ± 0.06 |
| rule amount+timing | 0.90 ± 0.02 | 0.90 ± 0.02 | 0.90 ± 0.03 |
| hybrid rule+forecast | 0.89 ± 0.03 | 0.89 ± 0.03 | 0.89 ± 0.03 |

Positives (total over 3 seeds): default 70, strong 70, weak 70

### S5. Double charge, AUROC

| score | strong | default | weak | none |
|---|---|---|---|---|
| rule timing \|gap z\| | 0.71 ± 0.01 | 0.71 ± 0.01 | 0.71 ± 0.01 | 0.65 ± 0.04 |
| forecast timing \|r_gap\| | 0.73 ± 0.04 | 0.71 ± 0.06 | 0.72 ± 0.06 | 0.72 ± 0.03 |
| autoencoder (window 5, latent 16) | 0.53 ± 0.03 | 0.55 ± 0.02 | 0.53 ± 0.02 | 0.45 ± 0.04 |
| rule amount+timing | 0.70 ± 0.02 | 0.69 ± 0.02 | 0.69 ± 0.02 | 0.61 ± 0.02 |
| hybrid rule+forecast | 0.71 ± 0.01 | 0.69 ± 0.02 | 0.69 ± 0.01 | 0.67 ± 0.03 |

Positives (total over 3 seeds): default 231, none 246, strong 231, weak 231

### S6. Duplicate amount (same amount as the previous transaction, normal timing), AUROC

| score | strong | default | weak | none |
|---|---|---|---|---|
| rule \|z\| | 0.59 ± 0.07 | 0.58 ± 0.07 | 0.57 ± 0.07 | 0.50 ± 0.03 |
| autoencoder (window 5, latent 16) | 0.53 ± 0.10 | 0.53 ± 0.12 | 0.58 ± 0.08 | 0.46 ± 0.05 |
| forecast \|r\| | 0.60 ± 0.07 | 0.59 ± 0.07 | 0.56 ± 0.07 | 0.50 ± 0.03 |
| masked \|r\| | 0.61 ± 0.07 | 0.59 ± 0.07 | 0.56 ± 0.07 | 0.51 ± 0.03 |
| hybrid rule+forecast | 0.59 ± 0.10 | 0.58 ± 0.05 | 0.56 ± 0.07 | 0.45 ± 0.09 |

Positives (total over 3 seeds): default 76, none 85, strong 76, weak 76

### B. Berka expense, spike: recall % at a 5% false-positive rate

| score | ×1.5 | ×2 | ×3 | ×5 | ×10 |
|---|---|---|---|---|---|
| rule \|z\| | 45 ± 1 | 49 ± 1 | 58 ± 1 | 71 ± 1 | 84 ± 1 |
| rule z (spike known) | 54 ± 1 | 63 ± 1 | 73 ± 1 | 83 ± 1 | 92 ± 1 |
| autoencoder (window 5, latent 16) | 35 ± 16 | 33 ± 20 | 43 ± 15 | 48 ± 15 | 54 ± 14 |
| forecast \|r\| | 47 ± 1 | 52 ± 1 | 62 ± 1 | 74 ± 2 | 86 ± 2 |
| forecast r (spike known) | 53 ± 1 | 62 ± 2 | 73 ± 2 | 83 ± 2 | 92 ± 2 |
| masked \|r\| | 46 ± 2 | 50 ± 4 | 57 ± 7 | 69 ± 8 | 82 ± 7 |
| hybrid rule+forecast | 45 ± 1 | 46 ± 1 | 51 ± 2 | 61 ± 2 | 75 ± 1 |

### B. Berka expense, drop: recall % at a 5% false-positive rate

| score | ×1.5 | ×2 | ×3 | ×5 | ×10 |
|---|---|---|---|---|---|
| rule \|z\| | 47 ± 1 | 51 ± 1 | 60 ± 1 | 73 ± 1 | 86 ± 1 |
| rule -z (drop known) | 49 ± 1 | 55 ± 1 | 64 ± 1 | 77 ± 1 | 90 ± 1 |
| autoencoder (window 5, latent 16) | 13 ± 16 | 27 ± 20 | 28 ± 22 | 30 ± 23 | 33 ± 25 |
| forecast \|r\| | 46 ± 1 | 49 ± 2 | 57 ± 2 | 68 ± 3 | 83 ± 4 |
| forecast -r (drop known) | 49 ± 1 | 55 ± 1 | 64 ± 1 | 77 ± 1 | 90 ± 2 |
| masked \|r\| | 47 ± 1 | 51 ± 2 | 59 ± 3 | 71 ± 3 | 86 ± 3 |
| hybrid rule+forecast | 45 ± 1 | 47 ± 1 | 50 ± 1 | 57 ± 2 | 72 ± 2 |

### B. Berka expense, late arrival: recall %

| score | +3 days | +7 days | +14 days | +30 days |
|---|---|---|---|---|
| rule timing \|gap z\| | 4 ± 0 | 42 ± 1 | 43 ± 1 | 44 ± 1 |
| rule timing gap z (late known) | 43 ± 1 | 49 ± 2 | 56 ± 2 | 76 ± 2 |
| forecast timing \|r_gap\| | 17 ± 9 | 34 ± 6 | 40 ± 2 | 48 ± 5 |
| forecast timing r_gap (late known) | 29 ± 11 | 44 ± 2 | 53 ± 3 | 77 ± 5 |
| autoencoder (window 5, latent 16) | 5 ± 1 | 5 ± 1 | 5 ± 1 | 5 ± 1 |
| hybrid rule+forecast | 12 ± 7 | 32 ± 13 | 40 ± 3 | 44 ± 1 |

### B. Berka income, spike: recall % at a 5% false-positive rate

| score | ×1.5 | ×2 | ×3 | ×5 | ×10 |
|---|---|---|---|---|---|
| rule \|z\| | 4 ± 0 | 11 ± 1 | 69 ± 3 | 70 ± 2 | 72 ± 2 |
| rule z (spike known) | 9 ± 1 | 68 ± 2 | 69 ± 2 | 71 ± 2 | 73 ± 2 |
| autoencoder (window 5, latent 16) | 15 ± 5 | 20 ± 5 | 28 ± 9 | 38 ± 17 | 41 ± 17 |
| forecast \|r\| | 63 ± 4 | 68 ± 4 | 73 ± 4 | 78 ± 3 | 83 ± 3 |
| forecast r (spike known) | 69 ± 4 | 74 ± 4 | 79 ± 3 | 85 ± 3 | 90 ± 3 |
| masked \|r\| | 56 ± 9 | 63 ± 5 | 67 ± 4 | 70 ± 4 | 74 ± 5 |
| hybrid rule+forecast | 52 ± 8 | 58 ± 5 | 61 ± 4 | 65 ± 3 | 69 ± 3 |

### B. Berka income, drop: recall % at a 5% false-positive rate

| score | ×1.5 | ×2 | ×3 | ×5 | ×10 |
|---|---|---|---|---|---|
| rule \|z\| | 6 ± 1 | 6 ± 1 | 65 ± 2 | 73 ± 2 | 74 ± 2 |
| rule -z (drop known) | 6 ± 1 | 31 ± 33 | 68 ± 6 | 74 ± 2 | 74 ± 2 |
| autoencoder (window 5, latent 16) | 13 ± 4 | 16 ± 4 | 21 ± 12 | 21 ± 8 | 22 ± 6 |
| forecast \|r\| | 60 ± 3 | 67 ± 4 | 72 ± 4 | 76 ± 4 | 82 ± 4 |
| forecast -r (drop known) | 65 ± 2 | 74 ± 3 | 79 ± 2 | 84 ± 2 | 89 ± 2 |
| masked \|r\| | 54 ± 7 | 61 ± 4 | 66 ± 3 | 71 ± 3 | 74 ± 4 |
| hybrid rule+forecast | 53 ± 7 | 58 ± 5 | 63 ± 4 | 68 ± 3 | 73 ± 3 |

### B. Berka income, late arrival: recall %

| score | +3 days | +7 days | +14 days | +30 days |
|---|---|---|---|---|
| rule timing \|gap z\| | 4 ± 0 | 3 ± 0 | 1 ± 0 | 54 ± 3 |
| rule timing gap z (late known) | 60 ± 2 | 69 ± 3 | 76 ± 2 | 95 ± 1 |
| forecast timing \|r_gap\| | 9 ± 7 | 18 ± 12 | 28 ± 18 | 58 ± 6 |
| forecast timing r_gap (late known) | 22 ± 11 | 43 ± 13 | 70 ± 3 | 90 ± 1 |
| autoencoder (window 5, latent 16) | 5 ± 0 | 5 ± 0 | 5 ± 0 | 5 ± 0 |
| hybrid rule+forecast | 8 ± 7 | 16 ± 9 | 21 ± 12 | 36 ± 14 |

