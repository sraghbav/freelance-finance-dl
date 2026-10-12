"""
Educational quarterly set-aside estimate for freelance income. NOT tax advice.

This is a flat-rate placeholder, not a tax calculation: real obligations depend on
jurisdiction, filing status, deductions, and credits. The BudgetWise dataset appears to be
Indian-city data (Mumbai, Bangalore, ...), so amounts are reported in the dataset's own units.
"""

import pandas as pd

from freelance_finance_dl.dataloader import INVALID_AMOUNT_CUTOFF

DEFAULT_SET_ASIDE_RATE = 0.25
# Categories that MAY include business expenses; a tax professional decides what is deductible.
BUSINESS_EXPENSE_CANDIDATES = {"travel", "education", "utilities"}

DISCLAIMER = (
    "Educational estimate only, not tax or financial advice. Set-aside uses a flat "
    f"{DEFAULT_SET_ASIDE_RATE:.0%} placeholder rate; expense categories are only candidates "
    "that may be deductible. Check with a qualified tax professional."
)


def quarterly_summary(df, user_id, rate=DEFAULT_SET_ASIDE_RATE):
    """Per-quarter freelance income, candidate business expenses, and suggested set-aside."""
    u = df[(df["user_id"] == user_id) & (df["amount"] > 0) & (df["amount"] < INVALID_AMOUNT_CUTOFF)]
    kind = u["transaction_type"].str.lower()
    quarter = u["date"].dt.to_period("Q")

    income = u[(kind == "income") & (u["category"] == "freelance")]
    expenses = u[(kind == "expense") & u["category"].isin(BUSINESS_EXPENSE_CANDIDATES)]

    summary = pd.DataFrame({
        "freelance_income": income.groupby(quarter[income.index])["amount"].sum(),
        "candidate_business_expenses": expenses.groupby(quarter[expenses.index])["amount"].sum(),
    }).fillna(0.0)
    summary["net_for_estimate"] = (
        summary["freelance_income"] - summary["candidate_business_expenses"]
    ).clip(lower=0)
    summary["suggested_set_aside"] = summary["net_for_estimate"] * rate
    return summary.sort_index()


def tax_note(df, user_id, rate=DEFAULT_SET_ASIDE_RATE):
    summary = quarterly_summary(df, user_id, rate)
    with_income = summary[summary["freelance_income"] > 0]
    if with_income.empty:
        return f"{user_id}: no freelance income recorded, so no set-aside estimate. {DISCLAIMER}"
    q = with_income.index[-1]
    row = with_income.iloc[-1]
    return (
        f"{user_id}, {q}: freelance income {row['freelance_income']:,.0f}; candidate business "
        f"expenses {row['candidate_business_expenses']:,.0f} (travel/education/utilities); "
        f"suggested set-aside {row['suggested_set_aside']:,.0f} at {rate:.0%}. {DISCLAIMER}"
    )
