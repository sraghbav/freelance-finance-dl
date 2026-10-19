# Thesis outline (working draft, 2026-10-20)

**Working title:** Explaining Unusual Spending for Freelancers: An LSTM Autoencoder with Grounded Plain-Language Explanations

**Not yet checked:** the Clark Honors College thesis template and formatting rules. Download them from the CHC thesis page and map these chapters onto the template before drafting in it. Citations below are placeholders `[find source]`; none have been gathered yet.

## Research questions
1. **Detection:** how reliably can an unsupervised LSTM autoencoder flag unusual spending sequences, and how large must an anomaly be before it is caught?
2. **Explanation:** can a small, free, locally run language model explain a flag faithfully, and does it beat a rule-based template?
3. **Scope note:** the tax set-aside note is an educational add-on, not a research question. Keep it to a short section.

## 1. Introduction (~3 pp)
- Irregular income and budgeting for freelancers; why explanations matter more than raw flags
- Research questions and contributions (detector evaluation, grounded explanation pipeline, honest negative results)
- Scope and disclaimer: educational prototype, not financial or tax advice

## 2. Background (~5 pp)
- Autoencoder anomaly detection; LSTM sequence autoencoders `[find source]`
- Evaluating detectors without labels: synthetic injection `[find source]`
- LLM hallucination and faithfulness; grounding and verification `[find source]`
- Freelancer finances and quarterly estimated tax (brief, US-centric context vs. the dataset) `[find source]`

## 3. Data and Methods (~10 pp)
| Section | Content | Source in repo |
|---|---|---|
| 3.1 Dataset and data quality | 15,900 rows; mixed date formats and currency text (rupees) dropped 69% of rows in the class-project loader; ~840 duplicates; 373 corrupt amounts; random-looking amounts | `dataloader.py`, README Dataset |
| 3.2 Sequences and features | window of 5; log-ratio vs the user's typical amount; why per-sequence min-max discards size | `dataloader.py` |
| 3.3 Model and training | LSTM autoencoder, latent size and why 3 (< input dim 5), 50 epochs | `model.py`, `train.py` |
| 3.4 User-level split | why sequence-level leaks; 128/22 users; full-history scale caveat | `get_user_split` |
| 3.5 Threshold and scoring | 95th percentile of val errors; mean vs. max over positions | `evaluate.py` |
| 3.6 Anomaly evaluation | raw-space injection (x or / factor), 5 factors, only valid sequences; **why the original normalized-space injection was invalid** | `evaluate.py`, `sweep.py` |
| 3.7 Explanation layer | facts computed in code, template, LLM (llama3.2:3b, temp 0), grounding check, fallback | `explain.py`, `pipeline.py` |
| 3.8 Explanation evaluation | 4-criterion rubric, LLM judge, human agreement check on 12 rows, prompt versions v2-v4 | `eval_explanations.py` |
| 3.9 Tax note | flat 25% placeholder; candidate categories; rupee data | `tax.py` |

## 4. Results (~8 pp)
- 4.1 **A debugging narrative as a result:** data loss (31% -> 88% rows kept), the invalid evaluation, and the scaling flaw, each found and corrected (README "Results")
- 4.2 Raw-space sensitivity (figure `docs/rawsweep_*.png`): min-max models at chance; log-ratio latent 3: spike x10 52%, x5 26%, x3 13%; drops <= 13%
- 4.3 Latent size: latent 16 copies its input (val MSE 0.0005) vs latent 3 (0.151); spikes 52% vs 43%, drops 13% vs 20%
- 4.4 Mean vs. max scoring: equivalent (52% vs 54%); the earlier hypothesis is not supported
- 4.5 Primary model: 7,281 sequences, 349 flagged (4.8%), examples of top flags
- 4.6 Explanation evaluation v2 -> v3 -> v4 (table in README); shipped quality equals the template's; checker blind spot (number used wrongly)
- 4.7 Judge vs. human agreement on 12 rows: 81% of criteria, plain language 58%
- 4.8 (Historical) class-project results and why the injection made them misleading

## 5. Discussion (~5 pp)
- What the results mean: only large spikes are detectable; random-looking data limits how far conclusions reach; the main contribution is a corrected, honest evaluation
- Why the checker mattered, and what it cannot catch (the "part of a pattern" case)
- The template matches the LLM on correctness; the LLM adds phrasing only (with a 3B model)
- Limitations (from README): data realism, small n, single feature, one small model, LLM judge, weak blinding, tax module jurisdiction
- Ethics: no financial advice, privacy benefit of a local model, generative AI use
- Future work: real transaction data, labeled anomalies, richer features, larger models, a user study

## 6. Conclusion (~1 p)

## Appendices
- A. Generative AI disclosure (copy from README and expand)
- B. Reproducibility: commands, seeds, versions
- C. Full sweep tables (`docs/sweep_*.csv`) and explanation scoring files (`docs/explanation_*`)

## Writing schedule (from the 10/18 plan)
- Wed 10/21 (done): data fix, evaluation fix, new features, results compiled
- Thu 10/22: Methods 3.1–3.5
- Fri 10/23: Methods 3.6–3.9
- Sat 10/24: Results 4.1–4.4; email advisor
- Sun 10/25: Results 4.5–4.8
- Week of 10/26: Background and Limitations

## Questions for the advisor (for the Saturday email)
1. The dataset looks synthetic (random amounts) and is in rupees. Is that acceptable for the thesis claims, or should I add a second dataset?
2. Should the tax module stay as a short educational section, or be cut?
3. Is an LLM-judge evaluation with a 12-row human agreement check acceptable, or should a human score all 40?
4. Expected length and format for a computational thesis in the honors college.
5. Is it acceptable to present the template-vs-LLM comparison as a negative result (the LLM adds no accuracy)?
