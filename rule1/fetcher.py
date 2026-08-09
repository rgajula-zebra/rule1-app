from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf


class FetchError(Exception):
    """Raised when we cannot get usable financials for a ticker."""


def _make_session() -> requests.Session:
    """Plain requests session for yfinance.

    yfinance 1.5.x defaults to curl_cffi, which fails SSL verification on
    corporate Windows machines behind TLS-inspecting proxies. Passing our
    own requests.Session sidesteps that entirely.
    """
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
    )
    return s


_SESSION = _make_session()


@dataclass
class Financials:
    ticker: str
    revenue: pd.Series
    net_income: pd.Series
    eps: pd.Series
    equity: pd.Series
    shares: pd.Series
    operating_cash_flow: pd.Series
    capex: pd.Series
    free_cash_flow: pd.Series
    long_term_debt: pd.Series
    tax_rate: pd.Series
    ebit: pd.Series
    current_price: float | None
    market_cap: float | None
    pe_ratio_ttm: float | None
    analyst_5yr_growth: float | None
    shares_outstanding: float | None = None
    dividend_yield: float | None = None
    beta: float | None = None
    book_value_per_share: float | None = None
    company_name: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def years_available(self) -> int:
        return int(self.revenue.dropna().shape[0])


_ROW_ALIASES = {
    "revenue": ["Total Revenue", "TotalRevenue", "Revenue", "Operating Revenue"],
    "net_income": [
        "Net Income",
        "NetIncome",
        "Net Income Common Stockholders",
        "Net Income From Continuing Operation Net Minority Interest",
    ],
    "eps": ["Diluted EPS", "DilutedEPS", "Basic EPS", "BasicEPS"],
    "shares": [
        "Diluted Average Shares",
        "DilutedAverageShares",
        "Basic Average Shares",
        "BasicAverageShares",
        "Share Issued",
    ],
    "ebit": ["EBIT", "Operating Income", "OperatingIncome"],
    "tax_provision": ["Tax Provision", "Income Tax Expense", "TaxProvision"],
    "pretax_income": ["Pretax Income", "PretaxIncome", "Income Before Tax"],
    "equity": [
        "Stockholders Equity",
        "StockholdersEquity",
        "Total Stockholder Equity",
        "Common Stock Equity",
    ],
    "long_term_debt": [
        "Long Term Debt",
        "LongTermDebt",
        "Long Term Debt And Capital Lease Obligation",
    ],
    "ocf": [
        "Operating Cash Flow",
        "OperatingCashFlow",
        "Cash Flow From Continuing Operating Activities",
        "Total Cash From Operating Activities",
    ],
    "capex": ["Capital Expenditure", "CapitalExpenditure", "Capital Expenditures"],
    "fcf": ["Free Cash Flow", "FreeCashFlow"],
}


def _pick_row(df: pd.DataFrame, keys: list[str]) -> pd.Series | None:
    if df is None or df.empty:
        return None
    for k in keys:
        if k in df.index:
            return df.loc[k]
    return None


def _to_year_series(row: pd.Series | None) -> pd.Series:
    if row is None:
        return pd.Series(dtype=float)
    s = row.copy()
    # yfinance columns are Timestamps; map to fiscal year int
    try:
        s.index = [pd.Timestamp(c).year for c in s.index]
    except Exception:
        pass
    s = s[~s.index.duplicated(keep="first")]
    s = s.sort_index()
    return pd.to_numeric(s, errors="coerce").dropna()


def fetch(ticker: str) -> Financials:
    """Pull normalized annual financials for `ticker` via yfinance.

    Raises FetchError with a user-friendly message if the ticker is invalid
    or returns no usable data.
    """
    if not ticker or not ticker.strip():
        raise FetchError("Please enter a ticker symbol.")

    symbol = ticker.strip().upper()
    yft = yf.Ticker(symbol, session=_SESSION)

    try:
        income = yft.income_stmt
        balance = yft.balance_sheet
        cash = yft.cashflow
    except Exception as e:
        raise FetchError(f"Could not fetch data for {symbol}: {e}") from e

    if (income is None or income.empty) and (balance is None or balance.empty):
        raise FetchError(
            f"No financial statements available for {symbol}. "
            "Check the ticker symbol or try a US-listed company."
        )

    revenue = _to_year_series(_pick_row(income, _ROW_ALIASES["revenue"]))
    net_income = _to_year_series(_pick_row(income, _ROW_ALIASES["net_income"]))
    eps = _to_year_series(_pick_row(income, _ROW_ALIASES["eps"]))
    shares = _to_year_series(_pick_row(income, _ROW_ALIASES["shares"]))
    ebit = _to_year_series(_pick_row(income, _ROW_ALIASES["ebit"]))
    tax_prov = _to_year_series(_pick_row(income, _ROW_ALIASES["tax_provision"]))
    pretax = _to_year_series(_pick_row(income, _ROW_ALIASES["pretax_income"]))

    equity = _to_year_series(_pick_row(balance, _ROW_ALIASES["equity"]))
    long_term_debt = _to_year_series(_pick_row(balance, _ROW_ALIASES["long_term_debt"]))

    ocf = _to_year_series(_pick_row(cash, _ROW_ALIASES["ocf"]))
    capex = _to_year_series(_pick_row(cash, _ROW_ALIASES["capex"]))
    fcf_direct = _to_year_series(_pick_row(cash, _ROW_ALIASES["fcf"]))

    # Prefer Yahoo's direct FCF row; fall back to OCF - |CapEx|.
    if not fcf_direct.empty:
        fcf = fcf_direct
    elif not ocf.empty and not capex.empty:
        aligned = pd.concat([ocf, capex], axis=1, join="inner")
        aligned.columns = ["ocf", "capex"]
        fcf = aligned["ocf"] - aligned["capex"].abs()
    else:
        fcf = pd.Series(dtype=float)

    # EPS fallback: net income / shares.
    if eps.empty and not net_income.empty and not shares.empty:
        aligned = pd.concat([net_income, shares], axis=1, join="inner")
        aligned.columns = ["ni", "sh"]
        aligned = aligned[aligned["sh"] > 0]
        eps = aligned["ni"] / aligned["sh"]

    # Effective tax rate per year, defaulting to 21% when unavailable.
    if not tax_prov.empty and not pretax.empty:
        aligned = pd.concat([tax_prov, pretax], axis=1, join="inner")
        aligned.columns = ["tax", "pretax"]
        aligned = aligned[aligned["pretax"] != 0]
        tax_rate = (aligned["tax"] / aligned["pretax"]).clip(lower=0, upper=0.5)
    else:
        tax_rate = pd.Series(dtype=float)

    # If we somehow got nothing at all, bail.
    if revenue.empty and net_income.empty and equity.empty:
        raise FetchError(
            f"No usable financials returned for {symbol}. The ticker may be "
            "delisted, non-US, or recently IPO'd with no annual filings yet."
        )

    # Snapshot info: price, market cap, PE, analyst growth.
    current_price = None
    market_cap = None
    company_name = ""
    try:
        fi = yft.fast_info
        current_price = float(fi.get("last_price") or fi.get("lastPrice") or np.nan)
        if np.isnan(current_price):
            current_price = None
        market_cap = fi.get("market_cap") or fi.get("marketCap")
        if market_cap is not None:
            market_cap = float(market_cap)
    except Exception:
        pass

    pe_ratio_ttm = None
    analyst_5yr_growth = None
    shares_outstanding = None
    dividend_yield = None
    beta = None
    book_value_per_share = None
    try:
        info = yft.info or {}
        company_name = info.get("longName") or info.get("shortName") or symbol
        if current_price is None:
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")
            if current_price is not None:
                current_price = float(current_price)
        if market_cap is None:
            market_cap = info.get("marketCap")
            if market_cap is not None:
                market_cap = float(market_cap)
        pe_ratio_ttm = info.get("trailingPE")
        if pe_ratio_ttm is not None:
            pe_ratio_ttm = float(pe_ratio_ttm)
        g = info.get("earningsGrowth") or info.get("revenueGrowth")
        if g is not None:
            analyst_5yr_growth = float(g)
        so = info.get("sharesOutstanding")
        if so is not None:
            shares_outstanding = float(so)
        dy = info.get("dividendYield")
        if dy is not None:
            # yfinance 0.2.40+ returns dividend yield as a percentage value
            # (e.g. 2.44 means 2.44%, not 244%). Convert to decimal.
            dividend_yield = float(dy) / 100
        b = info.get("beta")
        if b is not None:
            beta = float(b)
        bvps = info.get("bookValue")
        if bvps is not None:
            book_value_per_share = float(bvps)
    except Exception:
        if not company_name:
            company_name = symbol

    # Fallback: derive shares outstanding from market cap / price.
    if shares_outstanding is None and market_cap and current_price:
        shares_outstanding = market_cap / current_price

    return Financials(
        ticker=symbol,
        revenue=revenue,
        net_income=net_income,
        eps=eps,
        equity=equity,
        shares=shares,
        operating_cash_flow=ocf,
        capex=capex,
        free_cash_flow=fcf,
        long_term_debt=long_term_debt,
        tax_rate=tax_rate,
        ebit=ebit,
        current_price=current_price,
        market_cap=market_cap,
        pe_ratio_ttm=pe_ratio_ttm,
        analyst_5yr_growth=analyst_5yr_growth,
        shares_outstanding=shares_outstanding,
        dividend_yield=dividend_yield,
        beta=beta,
        book_value_per_share=book_value_per_share,
        company_name=company_name or symbol,
        raw={"income": income, "balance": balance, "cash": cash},
    )
