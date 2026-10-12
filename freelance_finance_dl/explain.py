"""
Explanation layer: turn a flagged sequence into plain-language text.

Design: every number in an explanation is computed here from the data (build_facts), never
produced by the LLM. The LLM only phrases the facts; a template baseline phrases the same
facts without an LLM so the two can be compared.
"""

import json
import os
import urllib.request

import numpy as np
import torch

from freelance_finance_dl.dataloader import INVALID_AMOUNT_CUTOFF

MODEL = os.environ.get("EXPLAIN_MODEL", "claude-sonnet-5-5")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = (
    "You explain automated spending-anomaly flags to a freelancer. "
    "Use ONLY the numbers in the facts JSON; never invent amounts, dates, causes, or currencies. "
    "Write 2-3 sentences in plain language. "
    "Say this is an automated flag, not proof that anything is wrong. "
    "If invalid_values is non-empty, say those values look like data-entry errors "
    "(zero, negative, or placeholder numbers) rather than real spending. "
    "Currency is unknown: write plain numbers with no currency symbol. "
    "State clearly whether the flagged amount is higher or lower than the user's typical amount, "
    "using direction_vs_users_typical and times_vs_users_typical exactly as given. "
    "Only mention data-entry errors if has_invalid_values is true. "
    "Do not give tax, legal, or investment advice."
)

USER_TEMPLATE = "Facts:\n{facts}\n\nWrite the 2-3 sentence explanation for the user now."


def _is_valid(amount):
    return 0 < amount < INVALID_AMOUNT_CUTOFF


def position_errors(model, dataset, idx, device):
    """Squared reconstruction error at each of the sequence positions (normalized scale)."""
    x, _ = dataset[idx]
    with torch.no_grad():
        recon = model(x.unsqueeze(0).to(device)).squeeze(0).cpu()
    return ((recon - x) ** 2).squeeze(-1).numpy()


def build_facts(dataset, model, device, idx, score, threshold):
    meta = dataset.get_sample_metadata(idx)
    amounts = [float(a) for a in meta["sequence"]]
    errors = position_errors(model, dataset, idx, device)
    pos = int(errors.argmax())
    flagged_amount = amounts[pos]

    data = dataset.data
    history = data[
        (data["user_id"] == meta["user_id"]) & (data["category"] == meta["category"])
    ]["amount"]
    valid_history = history[(history > 0) & (history < INVALID_AMOUNT_CUTOFF)]
    typical = float(valid_history.median()) if len(valid_history) else None

    others = [a for i, a in enumerate(amounts) if i != pos and _is_valid(a)]
    window_median = float(np.median(others)) if others else None

    ratio = (
        round(flagged_amount / typical, 2) if typical and _is_valid(flagged_amount) else None
    )
    return {
        "user_id": meta["user_id"],
        "category": meta["category"],
        "amounts_in_window": amounts,
        "flagged_position": pos + 1,
        "flagged_amount": flagged_amount,
        "users_typical_amount_for_category": typical,
        "ratio_to_users_typical": ratio,
        "direction_vs_users_typical": (
            None if ratio is None else ("higher" if ratio >= 1 else "lower")
        ),
        "times_vs_users_typical": (
            None if ratio is None else round(max(ratio, 1 / ratio), 1)
        ),
        "median_of_other_amounts_in_window": window_median,
        "invalid_values": [a for a in amounts if not _is_valid(a)],
        "has_invalid_values": any(not _is_valid(a) for a in amounts),
        "anomaly_score": round(float(score), 4),
        "threshold": round(float(threshold), 4),
        "severity_vs_threshold": round(float(score) / float(threshold), 2),
    }


def template_explanation(f):
    """Rule-based baseline: same facts, no LLM."""
    if f["invalid_values"]:
        vals = ", ".join(f"{v:,.0f}" for v in f["invalid_values"])
        return (
            f"This {f['category']} sequence for {f['user_id']} contains value(s) that look like "
            f"data-entry errors ({vals}) rather than real spending. Please check them."
        )
    ratio = f["ratio_to_users_typical"]
    amt = f["flagged_amount"]
    typ = f["users_typical_amount_for_category"]
    if ratio is None:
        return f"This {f['category']} sequence for {f['user_id']} was flagged as unusual."
    if ratio >= 1:
        rel = f"{ratio:.1f}x higher than"
    else:
        rel = f"{1 / ratio:.1f}x lower than"
    return (
        f"Transaction {f['flagged_position']} of 5 in this {f['category']} sequence for "
        f"{f['user_id']} ({amt:,.0f}) is {rel} their typical {f['category']} amount "
        f"({typ:,.0f}), and the pattern scored {f['severity_vs_threshold']:.1f}x the anomaly "
        f"threshold. This is an automated flag, not proof of a problem."
    )


def llm_explanation(f, client=None):
    """Phrase the facts with Claude. Requires the `anthropic` package and ANTHROPIC_API_KEY."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_TEMPLATE.format(facts=json.dumps(f))}],
    )
    return response.content[0].text.strip()


def ollama_explanation(f, model=OLLAMA_MODEL):
    """Phrase the facts with a free local model served by Ollama (https://ollama.com)."""
    body = json.dumps({
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(facts=json.dumps(f))},
        ],
    }).encode()
    request = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())["message"]["content"].strip()
