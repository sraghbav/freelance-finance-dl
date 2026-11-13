> **DRAFT written by Claude on 2026-11-12 (updated that evening) as a scaffold for the author to rewrite in their own words.** Appendix A is a statement about what *you* did and what Claude did, so it must be rewritten by you to match the truth. `[CONFIRM]` marks facts only you can confirm.

# Appendices

## A. Generative AI disclosure

**Course project (DSCI 410L).** Claude (Anthropic) was used as an assistant for code organization, docstrings and comments, debugging (tensor shapes, category-normalization logic), drafting the README, suggesting the synthetic-injection evaluation in response to instructor feedback, and running and verifying outputs. The core decisions (LSTM autoencoder, sequence construction, threshold method, interpretation of results) were the student's.

**Thesis extension (August to November 2026).** Claude (Anthropic), through the Claude Code tool, was used substantially more, and this statement says plainly what it did.

- *The student:* chose the thesis direction (an explanation layer, a tax note, and a free local model) and the definition of "unusual" (unexpected expenses and income); wrote the `get_user_split` method in `dataloader.py` with Claude's guidance; ran commands and reviewed outputs; decided to fix the data parsing and to adopt user-relative features after Claude reported the problems; supplied the Berka data and decided to run the model on it and on a simulated dataset; scored 12 of the round-v2 explanation texts independently; and wrote the final text of this thesis. `[CONFIRM and edit: list exactly what you did.]`
- *Claude:* wrote the remaining new code (robust parsing and de-duplication, the three feature types, raw-space anomaly injection, the cross-validation and baseline code, the Berka loader, the simulator, the explanation layer including the grounding check, the tax module, the pipeline, the evaluation scripts, and the figures); ran the experiments; **identified** the data-loss, scaling, invalid-evaluation, window-grouping, and feature-saturation problems and the lack of per-user and time structure in the data; scored all explanation texts as a language-model judge (Claude also wrote the prompts and the checker it scored, so it is not an independent rater); drafted the README; and, on 12 November 2026, wrote a first full draft of the thesis chapters, with numbers checked against the code and result files, which the student then rewrote. `[CONFIRM: that the final text is your own rewrite.]`
- *The language model under test:* the explanation model is a locally run open model (Llama 3.2, 3 billion parameters, via Ollama), so the explanation layer sends nothing to a third party. The *development process* is different: working with Claude through Claude Code sent code, command output, and excerpts of the datasets (including rows of transactions) to Anthropic's service. BudgetWise and the simulation contain no real people, and Berka is anonymized and public, so no real personal financial data were exposed, but the statement "nothing left the machine" would be false for the development process. `[CONFIRM and keep this distinction.]`

## B. Reproducibility

Code and results: github.com/sraghbav/freelance-finance-dl (commit hash `[add the final commit hash]`).

**Environment.** Apple laptop, CPU only, 8 GB memory; Python 3.12.12, PyTorch 2.5.1, pandas 2.2.2, NumPy 1.26.4, Matplotlib 3.10.8. Language model: `llama3.2:3b` served by Ollama (client 0.40.3), model digest `a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72`, temperature 0, seed 42. Random seed 42 throughout unless stated.

**Commands.**

```bash
# BudgetWise: primary model, cross-validation (each config), baselines, structure, figures
python -m freelance_finance_dl.train --output-dir checkpoints/primary_logratio_l3 --latent-dim 3 --epochs 50 --clean
python -m freelance_finance_dl.crossval --config logratio_l3     # also logratio_l16, minmax_l3, robustz_l3
python -m freelance_finance_dl.crossval --summarize
python -m freelance_finance_dl.baseline ; python -m freelance_finance_dl.baseline robustz
python -m freelance_finance_dl.structure
python -m freelance_finance_dl.plots

# Berka (put trans.csv in data/berka/)
python -m freelance_finance_dl.cv_eval --source berka --csv data/berka/trans.csv --features robustz --latent 16 --tag _berka_robustz_l16
#   variants: --features robustz --latent 3 --tag _berka_robustz_l3 | --features logratio --latent 16 --tag _berka_logratio_l16
python -m freelance_finance_dl.structure --source berka --csv data/berka/trans.csv

# Simulation (one process per seed 1, 2, 3), summary, and per-category test
python -m freelance_finance_dl.sim_eval --seed 1
python -m freelance_finance_dl.sim_eval --summarize
python -m freelance_finance_dl.habitsim --level default --seed 7 --anomaly-rate 0 --out data/sim/clean_default_s7.csv
python -m freelance_finance_dl.habit_eval --csv data/sim/clean_default_s7.csv --features robustz --latent 16 --tag _robustz_l16

# End-to-end report with local-model explanations; explanation evaluation round
python -m freelance_finance_dl.pipeline --top-n 10 --llm ollama
python -m freelance_finance_dl.eval_explanations --n 20 --tag _v8
```

The earlier grouping (user and category only, income not pooled) is available with `--legacy-grouping`; models trained before that fix are in `checkpoints/prefix_*`.

## C. Where the full tables are

| Content | File |
|---|---|
| BudgetWise cross-validated recall, all factors, configurations, and streams | `docs/sweeps/crossval_summary.csv` (per fold: `crossval_*.csv`) |
| Berka cross-validated recall | `docs/sweeps/cv_berka_*_summary.csv` (per fold: `cv_berka_*.csv`) |
| Simulation study (by level, detector, anomaly type) | `docs/sweeps/sim_summary.csv` (per seed: `sim_*_s*.csv`) |
| Simulation per-category test | `docs/sweeps/habit_eval_*.csv` |
| Every system prompt tried | `docs/explanation_eval/prompts.md` |
| Untrained rule baselines | `docs/sweeps/baseline_rule_summary.csv` (per fold: `baseline_rule_folds.csv`) |
| Single-split sweeps | `docs/sweeps/rawsweep_*.csv` |
| Explanation texts, keys, and LLM-judge scores | `docs/explanation_eval/` |
| Student's scores for 12 round-v2 texts | `docs/explanation_eval/explanation_scoring_human_v2.csv` |
| Figures | `docs/figures/`, `docs/sweeps/crossval_comparison.png` |
