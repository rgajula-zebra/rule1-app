from __future__ import annotations

import pandas as pd


def cagr(series: pd.Series, years: int) -> float | None:
    """Compounded annual growth rate over the last `years` full periods.

    Returns None when the window can't be computed honestly:
      - fewer than `years + 1` observations
      - start value <= 0 (CAGR is undefined; a sign flip is not "growth")
      - end value <= 0 (company went negative — treat as no meaningful CAGR)
    """
    s = series.dropna()
    if len(s) < years + 1:
        return None
    start = float(s.iloc[-(years + 1)])
    end = float(s.iloc[-1])
    if start <= 0 or end <= 0:
        return None
    return (end / start) ** (1 / years) - 1


def window_average(series: pd.Series, years: int) -> float | None:
    """Simple mean of the last `years` observations. Used for ROIC.

    Returns None if fewer than `years` observations are available.
    """
    s = series.dropna()
    if len(s) < years:
        return None
    return float(s.iloc[-years:].mean())
