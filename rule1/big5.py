from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .fetcher import Financials
from .growth import cagr, window_average

WINDOWS = (10, 5, 3, 1)
PASS_THRESHOLD = 0.10  # 10% — Phil Town's Rule #1 minimum


@dataclass
class MetricResult:
    label: str
    values: dict[int, float | None]  # window years -> rate
    passes: bool
    unit: str = "pct"  # "pct" or "ratio" (ROIC displayed as pct too)

    def as_row(self) -> dict[str, Any]:
        row = {"metric": self.label}
        for w in WINDOWS:
            row[f"{w}yr"] = self.values.get(w)
        row["pass"] = self.passes
        return row


@dataclass
class Big5Result:
    roic: MetricResult
    sales: MetricResult
    eps: MetricResult
    equity: MetricResult
    fcf: MetricResult
    roic_by_year: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def all_pass(self) -> bool:
        return all(m.passes for m in (self.roic, self.sales, self.eps, self.equity, self.fcf))

    def as_dataframe(self) -> pd.DataFrame:
        rows = [m.as_row() for m in (self.roic, self.sales, self.eps, self.equity, self.fcf)]
        return pd.DataFrame(rows)


def _roic_per_year(fin: Financials) -> pd.Series:
    """ROIC = NOPAT / (Equity + Long-term Debt).

    NOPAT = EBIT * (1 - effective tax rate). Falls back to Net Income when
    EBIT/tax rows are unavailable — a common simplification in Rule #1
    calculators that still tracks the return trend closely.
    """
    if fin.ebit.empty or fin.tax_rate.empty:
        nopat = fin.net_income
    else:
        aligned = pd.concat([fin.ebit, fin.tax_rate], axis=1, join="inner")
        aligned.columns = ["ebit", "tax"]
        nopat = aligned["ebit"] * (1 - aligned["tax"])
        if nopat.empty:
            nopat = fin.net_income

    if fin.long_term_debt.empty:
        invested = fin.equity
    else:
        invested = fin.equity.add(fin.long_term_debt, fill_value=0)

    both = pd.concat([nopat, invested], axis=1, join="inner")
    both.columns = ["nopat", "invested"]
    both = both[both["invested"] > 0]
    if both.empty:
        return pd.Series(dtype=float)
    return both["nopat"] / both["invested"]


def _metric_from_cagr(label: str, series: pd.Series) -> MetricResult:
    values = {w: cagr(series, w) for w in WINDOWS}
    computed = [v for v in values.values() if v is not None]
    passes = bool(computed) and all(v >= PASS_THRESHOLD for v in computed)
    return MetricResult(label=label, values=values, passes=passes)


def _metric_from_average(label: str, series: pd.Series) -> MetricResult:
    """For ROIC: average of the last N years rather than CAGR."""
    values = {w: window_average(series, w) for w in WINDOWS}
    computed = [v for v in values.values() if v is not None]
    passes = bool(computed) and all(v >= PASS_THRESHOLD for v in computed)
    return MetricResult(label=label, values=values, passes=passes)


def _bvps(fin: Financials) -> pd.Series:
    """Book value per share = equity / diluted shares.

    Falls back to raw equity if shares data is missing (rare but happens
    for some ADRs). Growth of raw equity tracks BVPS growth so long as
    the share count is roughly stable.
    """
    if fin.shares.empty:
        return fin.equity
    aligned = pd.concat([fin.equity, fin.shares], axis=1, join="inner")
    aligned.columns = ["eq", "sh"]
    aligned = aligned[aligned["sh"] > 0]
    if aligned.empty:
        return fin.equity
    return aligned["eq"] / aligned["sh"]


def compute_big5(fin: Financials) -> Big5Result:
    roic_series = _roic_per_year(fin)
    return Big5Result(
        roic=_metric_from_average("ROIC growth rate", roic_series),
        sales=_metric_from_cagr("Sales growth rate", fin.revenue),
        eps=_metric_from_cagr("EPS growth rate", fin.eps),
        equity=_metric_from_cagr("Equity (BVPS) growth rate", _bvps(fin)),
        fcf=_metric_from_cagr("Free Cash Flow growth rate", fin.free_cash_flow),
        roic_by_year=roic_series,
    )
