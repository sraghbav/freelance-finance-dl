"""
Explanation layer: turn a flagged sequence into plain-language text.

Design: every number in an explanation is computed here from the data (build_facts), never
produced by the LLM. The LLM only phrases the facts; a template baseline phrases the same
facts without an LLM so the two can be compared.
"""

import json
import os
import re
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

LLM_HIDDEN_FIELDS = ("anomaly_score", "threshold", "severity_vs_threshold")
USER_TEMPLATE = "Facts:\n{facts}\n\nWrite the 2-3 sentence explanation for the user now."


def facts_for_llm(f):
    """The LLM never sees the detector's internal score fields; the template still uses them."""
    return {k: v for k, v in f.items() if k not in LLM_HIDDEN_FIELDS}


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
        messages=[{"role": "user", "content": USER_TEMPLATE.format(facts=json.dumps(facts_for_llm(f)))}],
    )
    return response.content[0].text.strip()


def ollama_explanation(f, model=OLLAMA_MODEL):
    """Phrase the facts with a free local model served by Ollama (https://ollama.com)."""
    body = json.dumps({
        "model": model,
        "stream": False,
        "options": {"temperature": 0, "seed": 42},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(facts=json.dumps(facts_for_llm(f)))},
        ],
    }).encode()
    request = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())["message"]["content"].strip()


CURRENCY_SYMBOLS = "$\u20ac\u00a3\u20b9\u00a5"
INVENTED_ERROR_WORDS = ("data-entry", "data entry", "entry error", "placeholder",
                        "negative", "zero", "error")
NUMBER_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _allowed_numbers(f, hidden=()):
    """Every number the explanation is allowed to quote, taken from the facts."""
    allowed = {1.0, 2.0, 3.0, 4.0, 5.0}  # window positions
    for key, value in f.items():
        if key in hidden:
            continue
        values = value if isinstance(value, list) else [value]
        for v in values:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                allowed.update({float(v), abs(float(v))})
    return allowed


def check_grounding(text, f, hidden=()):
    """Return (passed, reasons). Strict on purpose: a failure falls back to the template."""
    reasons = []
    lowered = text.lower()

    if re.search(r"\[[^\]]*\]", text) or any("_" in key and key in text for key in f):
        reasons.append("leaked placeholder or field name")

    if any(symbol in text for symbol in CURRENCY_SYMBOLS):
        reasons.append("invented currency symbol")

    direction = f["direction_vs_users_typical"]
    if direction and not f["has_invalid_values"]:
        opposite = "lower" if direction == "higher" else "higher"
        if direction not in lowered:
            reasons.append(f"does not say the amount is {direction}")
        if opposite in lowered:
            reasons.append(f"says {opposite} but the data says {direction}")

    mentions_error = any(word in lowered for word in INVENTED_ERROR_WORDS)
    if mentions_error and not f["has_invalid_values"]:
        reasons.append("invents a data-entry error")
    if f["has_invalid_values"] and not mentions_error:
        reasons.append("omits the invalid values")

    allowed = _allowed_numbers(f, hidden)
    for token in NUMBER_PATTERN.findall(text.replace(f["user_id"], "")):
        number = float(token.replace(",", "").rstrip("."))
        decimals = len(token.split(".")[1]) if "." in token else 0
        tolerance = 0.5 * 10 ** -decimals + 1e-9
        if not any(abs(number - a) <= tolerance for a in allowed):
            reasons.append(f"number {token} is not in the facts")

    return (not reasons), reasons


PREAMBLE_PATTERN = re.compile(r"^\s*here(?:'s| is| are)[^\n]*:\s*\n+", re.IGNORECASE)


def strip_preamble(text):
    """Drop a leading 'Here's an explanation...:' line. Returns (clean_text, had_preamble)."""
    clean = PREAMBLE_PATTERN.sub("", text, count=1).strip()
    return clean, clean != text.strip()


def grounded_explanation(f, generate):
    """Ask the LLM (generate(facts) -> text); keep it only if it passes the grounding check."""
    raw = generate(f)
    cleaned, had_preamble = strip_preamble(raw)
    passed, reasons = check_grounding(cleaned, f, LLM_HIDDEN_FIELDS)
    return {
        "llm_raw": raw,
        "had_preamble": had_preamble,
        "grounded": passed,
        "reasons": reasons,
        "final": cleaned if passed else template_explanation(f),
        "final_source": "llm" if passed else "template_fallback",
    }
