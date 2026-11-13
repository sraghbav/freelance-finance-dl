> **DRAFT written by Claude on 2026-11-12 as a scaffold for the author to rewrite in their own words.** Numbers verified against code, data, and result files that day. `[CHECK]` = verify before relying on it; `[YOUR CALL]` = a judgment only you can make.

# 1. Introduction

## 1.1 Motivation

People with irregular income, such as freelancers, cannot rely on a steady paycheck to make budgeting easy. A single unusually large or small transaction can matter more for them than for a salaried worker, and in some tax systems they are responsible for setting money aside for tax themselves rather than having it withheld (Internal Revenue Service, n.d., Publication 505). A tool that watches their transactions therefore has two jobs: notice what is unusual, and say why in words a non-expert can act on. A bare anomaly score does neither. This thesis studies both jobs, and it spends as much effort on checking whether each one works as on building it.

## 1.2 Research questions

1. **Detection.** How well does an unsupervised LSTM autoencoder detect an unusually large or small transaction in a user's history for a spending category, as a function of how large the anomaly is, and does it beat simple rules that need no training?
2. **Explanation.** Can a small language model, run locally and for free, turn a flag into a faithful plain-language explanation, and does a deterministic verification step with a template fallback make that safe to show?

A methodological question runs through both: with no labeled anomalies, how should such a system be evaluated, and which evaluations are misleading?

## 1.3 What counts as unusual

In this thesis, "unusual" means an **unexpected amount, on the expense side or the income side**: a single transaction whose amount is far from what that user typically spends in that category, or typically receives as income. For a freelancer both directions matter. An unexpectedly large expense can break a budget, and unexpected income, whether a windfall or a shortfall, changes both cash flow and how much should be set aside for tax. Four cases are therefore in scope: an unusually high expense, an unusually low expense (for example a missing or refunded charge), unusually high income, and unusually low income (for example a missed payment). All of a user's income, from every source, is treated as one stream.

Not in scope: when or where a purchase happened, which merchant it was with, duplicate charges, changes in a recurring pattern such as a bill that stops, and anything that is not visible in the amount. In the terms of Chandola et al. (2009), this is a point-anomaly definition `[CHECK: confirm the point, contextual, and collective taxonomy and the page in the survey]`. I chose it because it can be measured from amounts alone, because it can be tested by controlled changes to a single amount (Section 3.6), and because it matches the freelancer use case above. The choice has a consequence that the results return to: for a point anomaly in amount, a simple rule is a strong baseline, and a sequence model can only show an advantage on data that contain habits, such as recurring bills or steady income, that the model can learn.

## 1.4 Approach

The pipeline has four stages: transactions are grouped into windows of five by user, transaction type, and category (with each user's income pooled); an LSTM autoencoder scores each window by how poorly it reconstructs it; each flagged window is turned into a small set of facts computed in ordinary code (which amount is unusual, how it compares with the user's typical amount); and either a rule-based template or a local language model (Llama 3.2, 3 billion parameters) phrases those facts. Language-model output is accepted only if an automatic grounding check finds no contradiction with the facts; otherwise the template text is shown. A separate educational module estimates a quarterly tax set-aside from freelance income.

The work began as a course project (DSCI 410L) and is extended here. Most of what is reported is what was found when that earlier work was re-examined.

## 1.5 Findings and contributions

1. **A data audit.** The original loader kept only 31% of the 15,900 rows because of mixed date formats and amounts written with currency text; a corrected loader keeps 86% after cleaning. Windows also mixed income with expenses and contained almost no income, which was corrected by grouping by transaction type and pooling each user's income (Section 4.1).
2. **A corrected evaluation.** The course project tested the detector by overwriting a normalized value with a number outside the possible range. Re-testing by changing one raw amount and processing it like real data showed that the course-project detector performed at chance (Sections 4.1 and 4.2).
3. **A detector evaluated with uncertainty.** Using amounts relative to each user's typical amount, the autoencoder detects a charge ten times too large about 41% of the time at a 5% false-positive rate (41.0 ± 3.7%, five-fold cross-validation over users), and is weak for smaller or downward changes (Section 4.3).
4. **Negative results.** A one-line untrained rule that knows the anomaly is a spike does better than the autoencoder (73.6 ± 4.1%). For unexpectedly low income, a one-line rule catches 94.7 ± 1.4% of tenfold drops against 8.3 ± 0.9% for the autoencoder (Sections 4.4 and 4.5). On a second, simulated dataset with real fraud labels, the best autoencoder at most ties the rule (Section 4.8), and neither dataset has the time structure a sequence model could use (Section 4.6). For this definition of unusual, the learned detector is not justified on this evidence.
5. **A verification-first explanation design.** Facts are computed in code, a model only phrases them, and a check rejects unsupported text. In the final round the checked pipeline was fully correct for 95% of 20 sampled windows against 100% for a plain template, which are statistically indistinguishable, and the model was brittle to prompt wording (Section 4.9).

## 1.6 Scope and disclaimers

This is an educational prototype. The dataset appears to be synthetic and its amounts are in rupees, so nothing here describes how real people spend money, and the tax module applies a flat placeholder rate rather than any jurisdiction's rules. Results are not financial or tax advice. The use of generative AI in this work, which was substantial, is described in Appendix A.

## 1.7 Outline

Chapter 2 gives background. Chapter 3 describes the data, model, evaluation, and explanation layer. Chapter 4 reports results. Chapter 5 discusses what they do and do not show, and Chapter 6 concludes.
