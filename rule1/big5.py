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
class WonderfulnessScore:
    """A 0-10 scale answer to 'is this a wonderful company?'.

    Composed of three sub-scores that each capture a different quality dimension:
      - pass_rate: how many of the 20 checks pass 10% (breadth)
      - magnitude: how far above 10% the metrics actually sit (height)
      - consistency: how tight the range is across windows (durability)

    The overall score is the mean of the three, so 10 is theoretically possible
    only for a compounder that beats 10% by a wide margin in every window with
    almost no variance — very rare (MSFT-class businesses).
    """
    overall: float                       # 0-10
    label: str                           # "Wonderful" / "Very Good" / ...
    color: str                           # bargain / green / orange / red / gray
    pass_rate: float                     # 0-10 — % of 20 checks passing
    magnitude: float                     # 0-10 — cushion above 10%
    consistency: float                   # 0-10 — variance across windows
    checks_passed: int                   # e.g. 14
    checks_total: int                    # 20 typical (5 metrics × 4 windows)
    strengths: list[str]                 # up to 2 top metrics
    weaknesses: list[str]                # up to 2 bottom metrics


def _label_for_score(s: float) -> tuple[str, str]:
    """Map a 0-10 score to (label, color-key used elsewhere in the UI)."""
    if s >= 9.0:
        return ("Wonderful", "bargain")
    if s >= 7.0:
        return ("Very Good", "green")
    if s >= 5.0:
        return ("Decent", "orange")
    if s >= 3.0:
        return ("Mediocre", "orange")
    return ("Poor", "red")


def _growth_score(values: list[float]) -> float:
    """Convert a metric's average growth rate into a 0-10 score.

    The Rule #1 bar is 10%. We want the score to rise meaningfully from there,
    but without giving a single outlier a disproportionate effect. A metric that
    averages 20% growth earns a 10/10 score, while 10% is worth 5/10.
    """
    if not values:
        return 0.0
    avg_growth = sum(values) / len(values)
    if avg_growth <= 0:
        return 0.0
    return max(0.0, min(10.0, (avg_growth / 0.20) * 10.0))


def score_wonderfulness(big5: "Big5Result") -> WonderfulnessScore:
    """Compute a 0-10 wonderfulness score from a Big5Result.

    The score blends three things:
      - how many Big 5 checks pass the 10% Rule #1 bar,
      - how quickly the business is growing on average,
      - how stable that growth is across windows.

    Only counts checks that were computable (some windows return None on short
    history). This means a ticker with 4 years of yfinance-only data is scored
    on ~10 checks instead of 20 — the pass-rate ratio still works.
    """
    metrics = [big5.roic, big5.sales, big5.eps, big5.equity, big5.fcf]

    computed = []
    for m in metrics:
        for w, v in m.values.items():
            if v is not None:
                computed.append((m, w, v))
    checks_total = len(computed)
    checks_passed = sum(1 for _, _, v in computed if v >= PASS_THRESHOLD)
    pass_rate = (checks_passed / checks_total * 10) if checks_total else 0.0

    metric_scores = []
    for m in metrics:
        vals = [v for v in m.values.values() if v is not None]
        if vals:
            metric_scores.append(_growth_score(vals))
    magnitude = (sum(metric_scores) / len(metric_scores)) if metric_scores else 0.0

    per_metric_std = []
    for m in metrics:
        vals = [v for v in m.values.values() if v is not None]
        if len(vals) >= 2:
            mean = sum(vals) / len(vals)
            var = sum((x - mean) ** 2 for x in vals) / len(vals)
            per_metric_std.append(var ** 0.5)
    if per_metric_std:
        avg_std = sum(per_metric_std) / len(per_metric_std)
        # 0% volatility -> 10; 15 percentage points of average swing -> 0.
        consistency = max(0.0, min(10.0, (0.15 - avg_std) / 0.15 * 10))
    else:
        consistency = 0.0

    overall = 0.6 * magnitude + 0.25 * pass_rate + 0.15 * consistency
    label, color = _label_for_score(overall)

    metric_scores_by_name = []
    for m in metrics:
        vals = [v for v in m.values.values() if v is not None]
        if vals:
            metric_scores_by_name.append((m.label, sum(vals) / len(vals)))
    metric_scores_by_name.sort(key=lambda t: t[1], reverse=True)
    strengths = [f"{name} avg {avg*100:.0f}%" for name, avg in metric_scores_by_name[:2] if avg >= PASS_THRESHOLD]
    weaknesses = [f"{name} avg {avg*100:.0f}%" for name, avg in metric_scores_by_name[-2:][::-1] if avg < PASS_THRESHOLD]

    return WonderfulnessScore(
        overall=round(overall, 1),
        label=label,
        color=color,
        pass_rate=round(pass_rate, 1),
        magnitude=round(magnitude, 1),
        consistency=round(consistency, 1),
        checks_passed=checks_passed,
        checks_total=checks_total,
        strengths=strengths,
        weaknesses=weaknesses,
    )


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

    def wonderfulness(self) -> WonderfulnessScore:
        return score_wonderfulness(self)


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
