> **DRAFT written by Claude on 2026-11-12 as a scaffold for the author to rewrite in their own words.** Every number was verified against the code, data, and result files that day; the source for each is the README and the commands in Appendix B. `[CHECK]` = verify; `[YOUR CALL]` = your judgment.

# 3. Methods

## 3.1 Dataset and data quality

I used the BudgetWise Personal Finance Dataset from Kaggle. It has 15,900 transaction rows for 150 users covering 1 January 2021 to 31 December 2024, with the fields `transaction_id`, `user_id`, `date`, `transaction_type`, `category`, `amount`, `payment_mode`, `location`, and `notes`. The dataset appears to be synthetic (Section 4.5), and the locations (Mumbai, Bangalore, Chennai) and amounts such as `₹5070` and `Rs.828` indicate that amounts are in rupees. I report amounts as plain numbers without a currency.

The raw file is messy, and how it is cleaned changes every later result. Table 3.1 lists the problems found and how each was handled.

**Table 3.1. Data problems and handling.**

| Problem | Evidence in the raw file | Handling |
|---|---|---|
| Category spellings | 46 distinct strings (for example `FOOD`, `Fod`, `Foods`) | Lowercased and mapped by a lookup table to 13 canonical categories |
| Mixed date formats | `2023-04-25` (6,091 rows), `08/05/2022` (4,676), `31-12-23` (3,044), `21-09-2024` (1,603); 486 rows have no date | Each format parsed with its own rule (below) |
| Amounts with currency text | 2,377 amounts such as `₹5070`, `Rs.828`, `5107 INR`; 291 amounts missing | Currency text and commas stripped before parsing |
| Missing required fields | 285 categories, 291 amounts, 486 dates, 0 user ids; 1,048 rows missing at least one | Rows dropped |
| Duplicates | 841 rows identical in transaction id, user, date, type, category, and amount | Removed |
| Shared transaction ids | 888 rows share an id with a different transaction | Kept |
| Corrupt amounts | 373 rows: 75 equal to 999,999,999, 83 equal to 999,999, 72 equal to zero, 143 negative | Dropped in the primary pipeline |

**Dates.** The four formats are each internally consistent. Slash dates are always month/day/year: none has a first field above 12, and 2,811 have a second field above 12. Dash dates are always day-month: none has a second field above 12, while 1,846 of the 3,044 two-digit-year dates and 953 of the 1,603 four-digit-year dates have a first field above 12. This is the evidence for the parsing rules. It cannot settle every row: 3,713 dates (1,865 slash, 1,198 two-digit dash, 650 four-digit dash) have both fields at or below 12, so their order is assumed from the rest of their format. A misread day and month would only shift a transaction by up to several months within the same user and category, but it can change the order of transactions within a window, so I treat the ordering of a few windows as uncertain.

**Row loss in the original loader.** The loader written for the course project parsed only ISO dates and plain numbers, and silently dropped everything else. It kept 4,977 of 15,900 rows (31.3%) and produced 1,104 sequences. After parsing every format and stripping currency text, 14,852 rows have all required fields; removing 841 duplicates leaves 14,011 rows (88.1%); dropping the 373 corrupt amounts leaves **13,638 rows** (Figure 4.1). I fixed the loader instead of only reporting the loss because every downstream quantity, including which users and categories appear at all, depends on it.

**Corrupt amounts.** I dropped the 373 rows instead of repairing them, because there is no basis for imputing a value: `999999` and `999999999` look like placeholders, and zero or negative amounts could be refunds or errors. Dropping removes any genuine refunds. `[YOUR CALL: say whether you agree with this tradeoff.]` Because each transaction lies in up to five overlapping windows, these 373 rows (2.7% of rows) touched 1,147 of 7,633 windows (15.0%) before they were dropped. The legacy pipeline is still available through `--legacy-parsing`; the corrupt-amount filter is the `--clean` option and is used in the primary configuration.

**What remains.** After cleaning, the data is dominated by a few categories: food, rent, and travel account for 71% of the windows, and income categories contribute very few (Table 3.2). Section 4.5 shows that the amounts also lack per-user and time structure.

## 3.2 Sequences and features

**Windows.** Transactions are grouped by `(user_id, category)` and sorted by date. A window of five consecutive transactions, advanced one step at a time, is one sequence; groups with fewer than five transactions produce none. Consecutive windows therefore share four of their five values. After cleaning there are **7,281 sequences** from all 150 users (Table 3.2).

**Table 3.2. Sequences by category after cleaning.**

| Category | Sequences | Category | Sequences |
|---|---|---|---|
| food | 2,330 | other | 140 |
| rent | 1,737 | health | 92 |
| travel | 1,140 | investment | 47 |
| utilities | 793 | salary | 39 |
| entertainment | 558 | bonus | 30 |
| education | 334 | freelance | 28 |
| | | savings | 13 |

**Features.** Each amount *a* in a window is replaced by log(*a* / *s*), where *s* is the median of that user's valid amounts in that category over their whole history. A value of 0 means the user's typical amount, +1.1 means about three times typical, and −1.1 about a third. Ratios are clipped to [0.001, 1000] before the logarithm so that extreme values stay finite. The class project instead scaled each window independently to [0, 1] (min-max scaling), which discards absolute size: a window of 100 to 500 and a window of 10,000 to 50,000 become identical. Min-max scaling is kept as a comparison (`--features minmax`).

The typical amount *s* uses each user's entire history, including transactions after the window and, for validation users, transactions the model never trains on. No anomaly labels are involved, but it is a mild look-ahead that a deployed system would not have. In the injection experiments (Section 3.6) *s* is held at its original value, so an injected anomaly does not change its own baseline.

## 3.3 Model and training

The model is an LSTM sequence autoencoder. The encoder is a one-layer LSTM (hidden size 64) over the five values; its final hidden state is projected by a linear layer to a latent vector of size 3. The decoder projects the latent vector back to size 64, repeats it at each of the five positions, passes it through a second one-layer LSTM (hidden size 64), and a linear layer outputs one value per position. The model has 50,948 parameters at latent size 3 (52,625 at size 16).

The latent size is chosen to be smaller than the five inputs. With a size of 16 the network has room to copy the input, and in practice it does: its validation error is 0.0005, against 0.151 at size 3 (Section 4.6). Training minimizes mean squared error with the Adam optimizer (learning rate 0.001, batch size 64) for 50 epochs with random seed 42, on CPU. The saved checkpoint is the epoch with the lowest validation error, which is epoch 50 for the primary model and epoch 43 for the latent-16 ablation.

## 3.4 User-level train and validation split

Because consecutive windows share four of five values, splitting windows at random would put near-copies of the same data on both sides and flatter the validation error. I split by user instead: the 150 users are shuffled with seed 42, the first 22 (15%) are validation users, and the other 128 are training users, giving 6,211 training and 1,070 validation sequences. All of a user's windows go to the same side.

## 3.5 Scores and threshold

A window's score is its reconstruction error: the mean over the five positions of the squared difference between input and output. I also tested the maximum over the five positions instead of the mean (Section 4.6). A window is flagged when its score exceeds a threshold, set to the 95th percentile of the scores of the held-out (validation) windows. By construction about 5% of validation windows are flagged, so the false-positive rate is fixed at about 5% and says nothing about how many real anomalies exist. Over all 7,281 windows, 349 (4.8%) are flagged.

In the single-split analysis the same 22 validation users select the checkpoint epoch, set the threshold, and supply the injected-anomaly test. This is mildly optimistic, and it is one reason the cross-validation below matters.

## 3.6 Anomaly evaluation

**Why the original method was invalid.** The course project tested the detector by overwriting one normalized value with 2.0 or −0.5. Normalized inputs always lie between 0 and 1, so these values could never occur in real data, and detecting them shows only that the model can detect impossible inputs.

**Raw-space injection.** In the corrected evaluation, one raw amount in a window is multiplied (spike) or divided (drop) by a factor of 1.5, 2, 3, 5, or 10. The altered window is then converted to features exactly like any other, with the same baseline *s*, scored, and compared with the threshold. The altered position is chosen uniformly at random once per window (seed 42) and reused across factors. I report recall (the share of altered windows flagged) at the threshold above. Chance level is the false-positive rate, about 5%: a detector that ignores the anomaly would flag about as many altered windows as clean ones. This design has a limitation. A multiplicative change to a small amount is not necessarily unusual in absolute terms (turning 200 into 2,000 stays inside the range of normal amounts for most categories), so 100% recall is unattainable even for a perfect detector. Recall values should therefore be used to compare detectors, not as absolute detection rates.

**Cross-validation.** A single split of 22 users is a noisy basis for a claim. I therefore used five-fold cross-validation over users: users are shuffled (seed 42) into five groups of 30; for each fold a fresh model is trained on the other 120 users for 50 epochs (seed 42 plus the fold number), the threshold is the 95th percentile of the held-out windows' scores, and the injection test is run on the held-out windows. No checkpoint selection is used. I report the mean and standard deviation of recall across the five folds. I also computed Wilson confidence intervals (Wilson, 1927) from the pooled counts, but they treat the 7,281 windows as independent, which they are not (overlapping windows, and several windows per user), so they are too narrow and I do not use them as the main measure of uncertainty. Three configurations were compared: log-ratio features with latent size 3 (primary), log-ratio with latent size 16, and min-max scaling with latent size 3.

**Untrained baselines.** To judge whether the autoencoder adds anything, I scored the same held-out windows and injected the same anomalies with three rules that need no training, all computed from the log-ratio features: the largest absolute value in the window (direction unknown); the largest value (assumes the anomaly is a spike); and the most negative value (assumes it is a drop). Each uses a threshold at the 95th percentile of the clean held-out scores. The two direction-aware rules use knowledge that a deployed detector would not have, so they show what a one-line rule can reach at best.

## 3.7 Explanation layer

For each flagged window the explanation layer first computes a small set of facts in ordinary code: the position in the window with the largest reconstruction error (the flagged position), the amount there, the user's typical amount for the category (the median *s*), the ratio of the two, whether the amount is "higher" or "lower", how many times higher or lower (rounded to one decimal), the median of the other four amounts, any invalid amounts, and the window's score, threshold, and ratio between them. Two texts are then produced from these facts.

1. **Template.** A rule-based sentence with fixed wording that states the position, the amount, how many times higher or lower than typical it is, how far above the threshold the window scored, and that it is an automated flag, not proof of a problem.
2. **Language model.** Llama 3.2 with 3 billion parameters (`llama3.2:3b`), run locally with Ollama at temperature 0 and seed 42. The system prompt instructs it to use only numbers in the facts, write two or three plain sentences, say that this is an automated flag and not proof, use no currency symbol (the currency is not given), state the direction from the facts exactly as given, mention data-entry errors only when the facts say invalid values are present, and give no tax, legal, or investment advice. The user message is the facts as JSON. From the third prompt version on, the model is not shown the score, threshold, or their ratio, because earlier versions mislabeled that ratio.

A leading line such as "Here's an explanation:" is removed before checking. The text is then checked by a **grounding check** that rejects it if any of the following holds: a bracketed placeholder or a field name with an underscore appears; a currency symbol appears; the direction word is missing or the opposite word is present (when no invalid values exist); vocabulary suggesting a data-entry error (such as "data-entry", "placeholder", "negative", "zero", "error") appears when no invalid values exist, or such vocabulary is missing when they do; or any number in the text, apart from the digits in the user's id, differs from every number in the facts by more than rounding. If the check fails, the template text is shown instead. The check can verify that a number exists in the facts and that the direction is right. It cannot verify what a number is being used for, or catch an invented claim phrased outside its vocabulary (Section 4.7).

## 3.8 Explanation evaluation

For each evaluation round I drew 20 flagged windows uniformly at random (seed 42) and produced a template text and a language-model text for each. Every text was scored on four binary criteria against the facts: **direction correct** (it says higher or lower correctly); **numbers correct and used correctly** (every number quoted is right and refers to what it should); **plain language** (a non-expert could follow it, with no field names or jargon); and **no invented causes** (it does not assert reasons the data do not support). The text is "fully correct" if it meets all four.

Four rounds were run. The first (v1) was an informal inspection of five outputs. Round v2 added explicit direction fields and a no-currency rule to the prompt, v3 hid the score fields from the model, and v4 used the v3 prompt on the primary model's flags. Rounds v2 and v3 were run on the same 20 windows, taken from the course-project detector's flags (min-max features, 52 flagged), so they are directly comparable; v4 used the primary detector's flags, which are larger, clearer deviations, so v4 is **not** comparable with v2 and v3.

The scoring was done by Claude acting as a language-model judge, for all 40 texts in each round. This is a limitation: Claude also wrote the prompts and the grounding check that are being evaluated, so it is not an independent rater, and the sheets are only weakly blind because the template's wording is formulaic. To measure how far the judge's scores can be trusted, I scored 12 of the 40 texts in round v2 myself (`[CONFIRM: this was done by the author]`) and compared them with the judge's scores. Because of the small number of windows per round, I report 95% Wilson intervals for the rates (Wilson, 1927).

## 3.9 Tax set-aside note

The tax module is an educational placeholder. For a user and calendar quarter, it sums the user's valid freelance-income transactions, sums valid expenses in the categories travel, education, and utilities (treated only as candidate business expenses), takes the difference (not below zero), and multiplies by a flat 25% set-aside rate. Every note carries a disclaimer that it is not tax advice. It was not evaluated, because there is no ground truth, and it applies no jurisdiction's rules.

## 3.10 Software and reproducibility

Everything was run on an Apple laptop CPU with 8 GB of memory, using Python 3.12.12, PyTorch 2.5.1, pandas 2.2.2, NumPy 1.26.4, and Matplotlib 3.10.8. The language model is `llama3.2:3b` served by Ollama (model digest in Appendix B). All random seeds are 42 unless stated. Commands to reproduce each result are in Appendix B, and the code is at github.com/sraghbav/freelance-finance-dl.
