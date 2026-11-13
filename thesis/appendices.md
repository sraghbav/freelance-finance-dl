> **DRAFT written by Claude on 2026-11-12 (updated that evening) as a scaffold for the author to rewrite in their own words.** Appendix A is a statement about what *you* did and what Claude did, so it must be rewritten by you to match the truth. `[CONFIRM]` marks facts only you can confirm.

# Appendices

## A. Generative AI disclosure

**Course project (DSCI 410L).** Claude (Anthropic) was used as an assistant for code organization, docstrings and comments, debugging (tensor shapes, category-normalization logic), drafting the README, suggesting the synthetic-injection evaluation in response to instructor feedback, and running and verifying outputs. The core decisions (LSTM autoencoder, sequence construction, threshold method, interpretation of results) were the student's.

**Thesis extension (August to November 2026).** Claude (Anthropic), through the Claude Code tool, was used substantially more, and this statement says plainly what it did.

- *The student:* chose the thesis direction (an explanation layer, a tax note, and a free local model); wrote the `get_user_split` method in `dataloader.py` with Claude's guidance; ran commands and reviewed outputs; decided to fix the data parsing and to adopt user-relative features after Claude reported the problems; scored 12 of the round-v2 explanation texts independently; and wrote the final text of this thesis. `[CONFIRM and edit: list exactly what you did.]`
- *Claude:* wrote the remaining new code (robust parsing and de-duplication, log-ratio features, raw-space anomaly injection, the sweep and cross-validation code, the untrained baselines, the explanation layer including the grounding check, the tax module, the pipeline, the evaluation scripts, and the figures); ran the experiments; **identified** the data-loss, scaling, and invalid-evaluation problems and the lack of per-user and time structure in the data; scored all explanation texts as a language-model judge (Claude also wrote the prompts and the checker it scored, so it is not an independent rater); drafted the README; and, on 12 November 2026, wrote a first full draft of the thesis chapters, with numbers checked against the code and result files, which the student then rewrote. `[CONFIRM: that the final text is your own rewrite.]`
- *The language model under test:* the explanation model is a locally run open model (Llama 3.2, 3 billion parameters, via Ollama), so the explanation layer sends nothing to a third party. The *development process* is different: working with Claude through Claude Code sent code, command output, and excerpts of the public dataset (including rows of transactions) to Anthropic's service. The dataset is public and appears synthetic, so no real personal financial data were exposed, but the statement "nothing left the machine" would be false for the development process. `[CONFIRM and keep this distinction.]`

## B. Reproducibility

Code and results: github.com/sraghbav/freelance-finance-dl (commit hash `[add the final commit hash]`).

**Environment.** Apple laptop, CPU only, 8 GB memory; Python 3.12.12, PyTorch 2.5.1, pandas 2.2.2, NumPy 1.26.4, Matplotlib 3.10.8. Language model: `llama3.2:3b` served by Ollama (client 0.40.3), model digest `a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72`, temperature 0, seed 42. Random seed 42 throughout unless stated.

**Commands.**

```bash
# Primary model (log-ratio features, cleaned amounts, latent size 3, type-aware windows with pooled income)
python -m freelance_finance_dl.train --output-dir checkpoints/primary_logratio_l3 --latent-dim 3 --epochs 50 --clean

# Single-split sweep, cross-validation (run each config), untrained baselines, structure checks, figures
python -m freelance_finance_dl.sweep --checkpoint checkpoints/primary_logratio_l3/best_model.pt --tag logratio_l3
python -m freelance_finance_dl.crossval --config logratio_l3     # also logratio_l16, minmax_l3
python -m freelance_finance_dl.crossval --summarize
python -m freelance_finance_dl.baseline
python -m freelance_finance_dl.structure
python -m freelance_finance_dl.plots

# Second dataset (download fraudTrain.csv and fraudTest.csv from Kaggle into data/sparkov/)
python -m freelance_finance_dl.structure --source sparkov --csv data/sparkov/fraudTrain.csv data/sparkov/fraudTest.csv
python -m freelance_finance_dl.fraud_eval --csv data/sparkov/fraudTrain.csv data/sparkov/fraudTest.csv --tag _l3_s42
#   variants: --latent 16 --tag _l16_s42 | --seed 7 --tag _l3_s7 | --train-windows 600000 --epochs 8 --tag _l3_s42_big

# End-to-end report with local-model explanations; explanation evaluation round
python -m freelance_finance_dl.pipeline --top-n 10 --llm ollama
python -m freelance_finance_dl.eval_explanations --n 20 --tag _v8
```

The earlier grouping (user and category only, income not pooled) is available with `--legacy-grouping`; models trained before the fix are in `checkpoints/prefix_*`.

## C. Where the full tables are

| Content | File |
|---|---|
| Cross-validated recall, all factors, configurations, and streams | `docs/sweeps/crossval_summary.csv` (per fold: `crossval_*.csv`) |
| Real-fraud-label results (Sparkov) | `docs/sweeps/fraud_eval_sparkov*.csv` |
| Every system prompt tried | `docs/explanation_eval/prompts.md` |
| Untrained rule baselines | `docs/sweeps/baseline_rule_summary.csv` (per fold: `baseline_rule_folds.csv`) |
| Single-split sweeps | `docs/sweeps/rawsweep_*.csv` |
| Explanation texts, keys, and LLM-judge scores | `docs/explanation_eval/` |
| Student's scores for 12 round-v2 texts | `docs/explanation_eval/explanation_scoring_human_v2.csv` |
| Figures | `docs/figures/`, `docs/sweeps/crossval_comparison.png` |
