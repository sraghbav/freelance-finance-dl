"""
Loader for the Berka Czech bank data (PKDD'99 Discovery Challenge): real, anonymized transactions
of 4,500 accounts, 1993-1998, amounts in Czech koruna. Only trans.csv is used.

Mapping to the standard columns:
  user       = account_id
  type       = income for credits (PRIJEM), expense for debits and withdrawals (VYDAJ, VYBER)
  category   = k_symbol when it names a payment kind, otherwise the operation
  excluded   = bank-generated entries that are not user behavior: interest credited (UROK),
               sanction interest (SANKC. UROK), and statement fees (SLUZBY)
All income is later pooled per account (see dataloader), as for the other datasets.
"""

import pandas as pd

EXCLUDED_K_SYMBOLS = {"UROK", "SANKC. UROK", "SLUZBY"}
K_SYMBOL_CATEGORY = {"SIPO": "household_payment", "POJISTNE": "insurance", "UVER": "loan_payment",
                     "DUCHOD": "pension"}
OPERATION_CATEGORY = {"VYBER": "cash_withdrawal", "VYBER KARTOU": "card_withdrawal",
                      "PREVOD NA UCET": "transfer_out", "PREVOD Z UCTU": "transfer_in",
                      "VKLAD": "cash_deposit"}


def load_berka(path, return_counts=False):
    d = pd.read_csv(path, sep=";", low_memory=False)
    counts = {"rows": len(d)}
    d["k_symbol"] = d["k_symbol"].fillna("").astype(str).str.strip()
    d["operation"] = d["operation"].fillna("").astype(str).str.strip()
    d = d[~d["k_symbol"].isin(EXCLUDED_K_SYMBOLS)].copy()
    counts["after_excluding_bank_generated"] = len(d)
    category = d["k_symbol"].map(K_SYMBOL_CATEGORY)
    category = category.fillna(d["operation"].map(OPERATION_CATEGORY)).fillna("other")
    out = pd.DataFrame({
        "transaction_id": d["trans_id"].astype(str),
        "user_id": d["account_id"].astype(str),
        "date": pd.to_datetime("19" + d["date"].astype(int).astype(str).str.zfill(6), format="%Y%m%d",
                               errors="coerce"),
        "transaction_type": (d["type"].astype(str).str.strip() == "PRIJEM").map({True: "income", False: "expense"}),
        "category": category,
        "amount": pd.to_numeric(d["amount"], errors="coerce"),
    })
    out = out.sort_values(["user_id", "date", "transaction_id"], kind="stable").reset_index(drop=True)
    return (out, counts) if return_counts else out
