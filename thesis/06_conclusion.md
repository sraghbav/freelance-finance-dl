> **DRAFT written by Claude on 2026-11-12 as a scaffold for the author to rewrite in their own words.** `[YOUR CALL]` = a judgment only you can make.

# 6. Conclusion

This thesis set out to flag unusual spending for people with irregular income and to explain each flag in plain language. The most important result came from examining the earlier work rather than extending it. The course-project loader had discarded 69% of the data, per-window scaling had thrown away how large each amount was, and the test used to show that the detector worked measured only whether it could detect impossible values. Once these were fixed and anomalies were injected realistically, the original detector was at chance.

With amounts expressed relative to each user's typical amount, an LSTM autoencoder detected a charge ten times too large 49.7 ± 3.5% of the time at a 5% false-positive rate (five-fold cross-validation over users), but it was close to chance for changes of a factor of two or less and weak for drops. A one-line untrained rule that knows the anomaly is a spike did better (74.4 ± 1.7%), and the dataset has no per-user or time structure for a sequence model to use, so on this data the learned detector is not justified. That is a statement about this dataset, not about autoencoders in general.

For explanations, computing every number in code, letting a small local model only phrase the facts, and rejecting any text that contradicts them gave output that was fully correct for 90% and 95% of 20 sampled flags in the last two rounds, statistically indistinguishable from a plain template. The design is worth keeping because it limits the damage of any single failure, but the language model contributed phrasing and not accuracy. Its characteristic failure, a true number used for the wrong thing, is one that a value-checking filter cannot catch.

The practical lessons are to audit data before modeling, to check that an evaluation could actually fail, to compare against simple baselines, and to report uncertainty and negative results. `[YOUR CALL: add one or two sentences on what you personally would do next.]`
