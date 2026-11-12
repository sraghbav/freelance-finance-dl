> **DRAFT written by Claude on 2026-11-12 as a scaffold for the author to rewrite in their own words.** This chapter contains interpretation, and the interpretation is yours to decide: edit anything you do not actually believe. Statements marked as hypotheses were not tested. `[CHECK]` = verify; `[YOUR CALL]` = your judgment.

# 5. Discussion

## 5.1 What the results show

**The earlier evaluation was the main problem.** The most consequential finding is not about the autoencoder. The course-project detector appeared to work (98.2% spike recall) because the test measured whether the model could detect inputs that real data cannot produce. Once anomalies were injected the way a real unusual charge would appear, the same detector was at chance. Two flaws compounded: scaling each window independently threw away absolute size, so a charge ten times too large looked like any other window, and the test never asked about size at all. A detector evaluated on an invalid test can look good for reasons unrelated to its purpose, and the corrected pipeline's lower numbers are more informative than the earlier high ones.

**The autoencoder detects large spikes and little else.** With user-relative features, it flags a charge ten times too large about half the time at a 5% false-positive rate, but is near chance for a factor of 2 or below, and weak for drops. A practical interpretation is that it can notice a charge that is wildly out of range for that user and category, and not much subtler deviations.

**On this dataset, the learned model is not justified.** A one-line rule that knows the anomaly is a spike beats the autoencoder (74.4% against 49.7% at ×10), and a rule that knows it is a drop matches the better autoencoder. The autoencoder is not told the direction, so the comparison is not entirely fair to it, and a fairer comparison against a rule that must handle both directions would use a combined score, which I did not build. Even so, the evidence in Section 4.5 explains why a sequence model has nothing to learn: users do not differ in their typical amounts, and consecutive transactions are unrelated. A model whose strength is exploiting order and personal history is being given data with neither. The result should be read as a statement about this dataset, not about autoencoders. Real transactions have habits (rent recurs, subscriptions repeat), and the question of whether an LSTM autoencoder helps on such data was not tested.

**Why the autoencoder beats the direction-unknown rule on spikes (hypothesis, not tested).** The training data contain very few large positive deviations from a user's typical amount, because non-rent amounts are bounded near 10,000, and many large negative deviations, because small amounts are common. A model that has learned this asymmetry would reconstruct a large positive value badly and a large negative one well, which would make it good at spikes and poor at drops, as observed. A rule on the absolute value cannot represent the asymmetry. Testing this would need a model trained on data with the opposite asymmetry.

## 5.2 What the grounding check does and cannot do

The grounding check removed the failures that are easy to define mechanically: placeholders, currency symbols, a wrong direction, an invented data-entry error, a number that does not appear in the facts. It never rejected a good text. It cannot detect a true number used for the wrong purpose, and in two of the four rounds that was the dominant failure ("the threshold of 1.06" for a severity ratio, "4 positions higher" for a window position). Two lessons follow. First, giving the model less is a good defense: hiding the three fields it misused removed that failure class completely, which no output filter did. Second, a filter that checks values is not a filter that checks meaning, and a stronger check would need to understand what each number refers to, for example by requiring the model to produce a structured statement ("amount X is Y times typical") that can be verified, rather than free text.

The general design, computing facts deterministically, letting a model only phrase them, verifying the result, and falling back to a template, is useful because it bounds the damage of any single failure: the worst case is the template text.

## 5.3 Template versus language model

In the last two rounds the checked pipeline's output was fully correct for 90% and 95% of 20 windows, against 90% and 100% for the template, and the intervals overlap heavily. Within this evaluation the language model added phrasing variety and did not add accuracy. Whether the variety is valuable to a reader was not tested; no user study was done. The result is for a 3-billion-parameter model, and a larger model, which could not be tried because there was no budget for one, might change it. The comparison in v4 is also easier than in v2 and v3, because the primary detector's flags are larger and clearer, so the high v4 number should not be read as the model having improved.

## 5.4 Limitations

- **Data.** The dataset appears synthetic, in rupees, with no per-user or time structure and an unusual rent distribution. Nothing here describes real spending. Many rows were corrupt, duplicated, or unparseable, and the parsing rules for 3,713 ambiguous dates rest on an assumption supported by the unambiguous ones.
- **Evaluation design.** Injected multiplicative anomalies are not necessarily unusual in absolute terms, so recall is bounded well below 100% even for a perfect detector, and the numbers support comparison rather than absolute claims. Thresholds are set by percentile, so the false-positive rate is built in and says nothing about real anomalies. Only two anomaly types were tested.
- **Statistics.** Cross-validation used five folds, so the standard deviations are rough. Wilson intervals treat correlated windows as independent and understate uncertainty. The explanation rounds each use 20 windows; differences of a few windows are noise.
- **Scaling.** The typical amount uses each user's whole history, including later and validation data. A deployed detector would only know the past.
- **Single feature.** Only the amount is used: no date features, transaction type, payment mode, or location.
- **Judge.** Explanation scores come from Claude, which also wrote the prompts and the checker, with a 12-text human check on one round. The check agreed on 81% of judgments, and on only 58% for plain language. The scoring is weakly blind.
- **Checker.** It verifies values and direction, not meaning, and misses invented claims outside its vocabulary.
- **Overlapping windows.** One unusual transaction produces several flagged windows, so flag counts are not counts of unusual transactions. In this data, 67% of flags are rent windows, an artifact of rent's wide amount distribution.
- **Tax module.** A flat 25% placeholder, not evaluated, with no jurisdiction-specific rules.
- **Model size.** Only one small local language model was tested.

## 5.5 Ethics and responsible use

The system is an educational prototype and every tax-related output carries a disclaimer: it is not financial or tax advice, and a wrong set-aside number could lead a real freelancer to under-save. Using a local model keeps financial data on the user's machine, and no real personal data were processed here, since the dataset appears synthetic. The system also flags and explains but never acts, so a person stays in the loop. The use of generative AI in building and evaluating this work was substantial and is disclosed in Appendix A. In particular, Claude both wrote parts of the system and scored its outputs, which is why the human agreement check was done and why the results should be read with that in mind.

## 5.6 Future work

- Evaluate on real, consented transaction data, or on a synthetic generator with realistic habits, so that sequence models have structure to learn.
- Build and compare a combined detector (separate spike and drop scores) with the autoencoder, and test models that use dates and transaction type.
- Evaluate against real or expert-labeled anomalies rather than injected ones.
- Replace free-text explanations with structured statements that a checker can verify, test larger models, and run a user study on whether people find the explanations useful.
- Re-do the human scoring on all rounds with several independent raters.
