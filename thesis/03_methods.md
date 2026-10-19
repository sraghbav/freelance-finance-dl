# Methods draft (your prose goes here)

Write under each heading. The bullets are verified facts and questions to answer, not text to copy.
Source for every number: the README, `dataloader.py`, and the commands in the README.

---

## 3.1 Dataset and data quality

**Target:** about one page, 3–4 paragraphs. Order: what the data is → what was wrong with it → how you handled each problem → what remained.

**Facts to place (verified):**
- Kaggle "BudgetWise Personal Finance Dataset"; 15,900 rows, 150 users, 2021-01-01 to 2024-12-31; fields `transaction_id, user_id, date, transaction_type, category, amount, payment_mode, location, notes`
- Categories: 46 raw spellings (`FOOD`, `Fod`, `Foods`) → 13 canonical categories, via a lookup table
- Dates: four formats — `2023-04-25` (6,091 rows), `08/05/2022` (4,676), `31-12-23` (3,044), `21-09-2024` (1,603); 486 rows have no date
  - Slash dates are always month/day/year: 0 have a first field above 12, 2,811 have a second field above 12
  - Dash dates are always day-month: none has a second field above 12; 1,846 of the 3,044 two-digit-year dates and 953 of the 1,603 four-digit-year dates have a first field above 12
  - 3,713 dates (1,865 slash, 1,198 two-digit dash, 650 four-digit dash) have both fields ≤ 12, so their order is *assumed* from the rest of their format, not checkable one by one
- Amounts: `5070`, `₹5070`, `Rs.828`, `5107 INR` — the rupee symbol and "Rs." show the data is in rupees. 291 amounts are missing
- Missing: 285 categories, 291 amounts, 486 dates, 0 user ids; 1,048 rows are missing at least one required field
- Duplicates: 900 fully duplicated rows in the raw file; after parsing, about 840 duplicate rows (same transaction id, user, date, type, category, amount) were removed. 740 rows share a transaction id with a *different* transaction and were kept
- Corrupt amounts: 373 rows are `999999999`, `999999`, zero, or negative
- Outcome: the original loader kept 4,977 rows (31%); the robust loader keeps 14,011 (88%); with corrupt amounts dropped, 13,638 rows
- Amounts look randomly generated, not behavioral; categories are very uneven (food, rent, travel are 71% of the sequences)

**Questions your paragraphs should answer:**
1. Why were rows dropped, and why was that a problem worth fixing rather than just reporting?
2. Why is it reasonable to assume the date formats you did? What is the evidence, and what *can't* you verify?
3. Why drop corrupt amounts instead of repairing them?
4. What does a reader need to know about the data's limits before trusting any result?

**Your draft:**


---

## 3.2 Sequences and features
Reminders: window of 5 within one user and category; overlapping windows; log-ratio of each amount to the user's median for that category; why the earlier min-max scaling discards size; the caveat that the median uses the user's full history.

**Your draft:**


---

## 3.3 Model and training
Reminders: LSTM autoencoder, hidden size 64, latent size 3; why a latent smaller than the 5 inputs matters (a 16-dim latent can copy its input); 50 epochs, Adam, learning rate 1e-3, mean squared error.

**Your draft:**


---

## 3.4 User-level split
Reminders: overlapping windows share 4 of 5 values, so a random sequence split leaks; split by whole users (128 train / 22 validation users; 6,211 / 1,070 sequences); fixed random seed.

**Your draft:**


---

## 3.5 Threshold and scoring
Reminders: reconstruction error per sequence, mean over the 5 positions (max over positions also tested); threshold is the 95th percentile of validation errors, so about 5% of normal validation sequences are flagged by construction.

**Your draft:**
