"""
Evaluate explanation quality: template baseline vs. LLM, with an automatic grounding check
and a blind manual rubric.

Step 1 - generate (needs the Ollama server running):
    python -m freelance_finance_dl.eval_explanations --n 20 --tag _v5

Step 2 - open docs/explanation_eval/explanation_scoring_blind<tag>.csv, fill the four 1/0 columns for every row
(rows are shuffled and unlabeled), save it.

Step 3 - summarize:
    python -m freelance_finance_dl.eval_explanations --summarize --tag _v5 --scores <scored.csv>
"""

import argparse
from collections import Counter

import numpy as np
import pandas as pd

from freelance_finance_dl.explain import (
    LLM_HIDDEN_FIELDS, build_facts, check_grounding, strip_preamble, template_explanation)
from freelance_finance_dl.pipeline import make_generator, score_and_flag

EVAL_DIR = "docs/explanation_eval"
CRITERIA = ["direction_correct", "only_real_numbers", "plain_language", "no_invented_causes"]


def truth_summary(f):
    if f["has_invalid_values"]:
        return f"window {f['amounts_in_window']}; invalid values present: {f['invalid_values']}"
    return (f"flagged amount {f['flagged_amount']:,.0f} at position {f['flagged_position']}; "
            f"user's typical {f['users_typical_amount_for_category']:,.0f}; "
            f"{f['times_vs_users_typical']}x {f['direction_vs_users_typical']}; "
            f"no invalid values")


def paths(tag):
    return (f"{EVAL_DIR}/explanation_eval_auto{tag}.csv",
            f"{EVAL_DIR}/explanation_scoring_blind{tag}.csv",
            f"{EVAL_DIR}/explanation_scoring_key{tag}.csv")


def generate(args):
    AUTO_CSV, BLIND_CSV, KEY_CSV = paths(args.tag)
    model, dataset, device, errors, threshold, flagged = score_and_flag(
        args.csv, args.checkpoint, args.score)
    rng = np.random.default_rng(args.seed)
    chosen = sorted(rng.choice(flagged, size=min(args.n, len(flagged)), replace=False).tolist())
    generate_text = make_generator(args.llm)

    auto_rows, blind_rows = [], []
    for idx in chosen:
        f = build_facts(dataset, model, device, idx, errors[idx], threshold)
        template = template_explanation(f)
        raw = generate_text(f)
        cleaned, had_preamble = strip_preamble(raw)
        llm_ok, reasons = check_grounding(cleaned, f, LLM_HIDDEN_FIELDS)
        template_ok, _ = check_grounding(template, f)
        auto_rows.append({
            "idx": idx, "user_id": f["user_id"], "category": f["category"],
            "has_invalid_values": f["has_invalid_values"],
            "template_passes_check": template_ok,
            "llm_text": cleaned, "had_preamble": had_preamble,
            "llm_passes_check": llm_ok, "reasons": "; ".join(reasons),
        })
        for version, text in [("template", template), ("llm", cleaned)]:
            blind_rows.append({"idx": idx, "version": version, "truth": truth_summary(f),
                               "text": text})

    auto = pd.DataFrame(auto_rows)
    auto.to_csv(AUTO_CSV, index=False)

    blind = pd.DataFrame(blind_rows).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    blind.insert(0, "row_id", range(1, len(blind) + 1))
    blind[["row_id", "idx", "version"]].to_csv(KEY_CSV, index=False)
    sheet = blind[["row_id", "truth", "text"]].copy()
    for criterion in CRITERIA:
        sheet[criterion] = ""
    sheet.to_csv(BLIND_CSV, index=False)

    print(f"Sequences evaluated:             {len(auto)}")
    print(f"Template passes check:           {auto['template_passes_check'].mean():.0%} "
          f"(sanity check of the checker; should be 100%)")
    print(f"LLM passes grounding check:      {auto['llm_passes_check'].mean():.0%}")
    print(f"LLM fell back to template:       {(~auto['llm_passes_check']).mean():.0%}")
    print(f"LLM added a chatty preamble:     {auto['had_preamble'].mean():.0%}")
    reasons = Counter(r.split(" ")[0] + " " + r.split(" ")[1] if r.startswith("number")
                      else r for row in auto["reasons"] for r in row.split("; ") if r)
    print("Failure reasons:", dict(reasons))
    print(f"\nWrote {AUTO_CSV}, {BLIND_CSV} (fill this in), {KEY_CSV} (don't open until done)")


def summarize(scores_path, tag=""):
    AUTO_CSV, BLIND_CSV, KEY_CSV = paths(tag)
    sheet = pd.read_csv(scores_path)
    key = pd.read_csv(KEY_CSV)
    auto = pd.read_csv(AUTO_CSV)
    scored = sheet.merge(key, on="row_id")
    if scored[CRITERIA].isna().any().any():
        raise SystemExit("Some rubric cells are still empty. Fill every 1/0 in the blind sheet.")
    scored["all_four"] = scored[CRITERIA].min(axis=1)

    table = scored.groupby("version")[CRITERIA + ["all_four"]].mean().round(2)
    print("Share of explanations meeting each criterion (manual, blind):\n")
    print(table.to_string())
    table.to_csv(scores_path.replace(".csv", "_summary.csv"))

    llm = scored[scored["version"] == "llm"].merge(
        auto[["idx", "llm_passes_check"]], on="idx")
    both = pd.crosstab(llm["llm_passes_check"], llm["all_four"] == 1,
                       rownames=["checker passed"], colnames=["human: all four OK"])
    print("\nChecker vs. human judgement on the LLM texts:\n")
    print(both.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explanation quality evaluation")
    parser.add_argument("--csv", default="data/budgetwise_finance_dataset.csv")
    parser.add_argument("--checkpoint", default="checkpoints/primary_logratio_l3/best_model.pt")
    parser.add_argument("--score", choices=["mean", "max"], default="mean")
    parser.add_argument("--llm", choices=["anthropic", "ollama"], default="ollama")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--scores", default=None, help="Scored sheet to summarize")
    parser.add_argument("--tag", required=True,
                        help="Suffix for output files, e.g. _v5 (required so existing sheets are never overwritten)")
    args = parser.parse_args()
    summarize(args.scores or paths(args.tag)[1], args.tag) if args.summarize else generate(args)
