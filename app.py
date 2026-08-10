from __future__ import annotations

import pandas as pd
import streamlit as st

from rule1 import (
    Big5Result,
    Financials,
    compute_big5,
    dcf_two_stage,
    debt_to_fcf,
    fetch,
    graham_formula,
    graham_number,
    payback_time,
    peg_fair_value,
    peter_lynch_fair,
    sticker_price,
)
from rule1.big5 import WINDOWS, PASS_THRESHOLD
from rule1.fetcher import FetchError
from rule1.intrinsic import MethodResult

st.set_page_config(page_title="Value Investing", page_icon=None, layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_fetch(symbol: str) -> Financials:
    return fetch(symbol)


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%"


def _fmt_money(v: float | None) -> str:
    if v is None:
        return "n/a"
    if abs(v) >= 1e9:
        return f"${v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.2f}M"
    return f"${v:,.2f}"


def _color_pass(val: float | None) -> str:
    if val is None:
        return "color: #888;"
    if val >= PASS_THRESHOLD:
        return "background-color: #1e4620; color: #b6f0b6;"
    return "background-color: #4b1e1e; color: #f0b6b6;"


def _render_big5_table(big5: Big5Result) -> None:
    df = big5.as_dataframe()
    display = df.copy()
    for w in WINDOWS:
        col = f"{w}yr"
        display[col] = display[col].map(_fmt_pct)
    display["pass"] = display["pass"].map(lambda p: "PASS" if p else "FAIL")
    display = display.rename(columns={"metric": "Metric", "pass": "Verdict"})

    def _style(row):
        original = df.loc[row.name]
        out = [""]  # Metric column
        for w in WINDOWS:
            out.append(_color_pass(original[f"{w}yr"]))
        out.append(
            "background-color: #1e4620; color: #b6f0b6;"
            if original["pass"]
            else "background-color: #4b1e1e; color: #f0b6b6;"
        )
        return out

    styled = display.style.apply(_style, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)


def _verdict_price(current: float | None, mos: float | None, sticker: float | None) -> tuple[str, str]:
    if current is None or mos is None or sticker is None:
        return ("Unknown", "gray")
    if current <= mos:
        return ("BUY — trading below MOS", "green")
    if current <= sticker:
        return ("WATCH — between MOS and sticker", "orange")
    return ("AVOID — above sticker price", "red")


def _big5_eps_growth(big5: Big5Result) -> float | None:
    """Pick the most conservative computable EPS growth for valuation."""
    computed = [v for v in big5.eps.values.values() if v is not None]
    if not computed:
        return None
    return min(computed)


st.markdown(
    """
    <div style="margin-bottom: 0.5rem;">
        <div style="font-size: 2.75rem; font-weight: 700; line-height: 1.1;">
            Value Investing
        </div>
        <div style="font-size: 0.85rem; color: #9aa0a6; margin-top: 0.15rem;">
            inspired by Rule #1 Investing by Phil Town
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Big 5 growth screening plus DCF, Peter Lynch Fair Value, Graham Number, "
    "Graham Formula, and PEG. Enter a ticker to compute all methods side-by-side."
)

with st.sidebar:
    st.header("Valuation knobs")
    dcf_discount = st.slider(
        "DCF discount rate", 0.06, 0.15, 0.10, 0.005,
        help="Required annual return. 10% = long-run S&P 500 average; 15% = Phil Town's aggressive rate.",
    )
    dcf_terminal = st.slider(
        "DCF terminal growth", 0.00, 0.04, 0.025, 0.005,
        help="Perpetual growth rate after the fade period. Should be <= long-run GDP growth (~2.5-3%).",
    )
    aaa_yield = st.slider(
        "AAA corporate bond yield (for Graham Formula)", 0.02, 0.10, 0.045, 0.005,
        help="Current AAA corporate bond yield. Used in Graham's revised 1974 formula.",
    )
    mos_pct = st.slider(
        "Margin of Safety (Lynch/Graham/PEG)", 0.10, 0.60, 0.25, 0.05,
        help="Discount applied to fair value to get a buy price. Phil Town's sticker uses a fixed 50%.",
    )

with st.form("analyze"):
    col_a, col_b, col_c = st.columns([2, 1, 5])
    with col_a:
        ticker = st.text_input("Ticker symbol", value="AAPL", max_chars=10).strip().upper()
    with col_b:
        # Blank markdown pushes the button down so it lines up with the input
        # (whose label adds ~28px of height above it).
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Analyze", use_container_width=True)

if not submitted and "last_ticker" not in st.session_state:
    st.info("Enter a US-listed ticker (e.g. AAPL, MSFT, KO) and click **Analyze**.")
    st.stop()

if submitted:
    st.session_state["last_ticker"] = ticker

symbol = st.session_state.get("last_ticker", ticker)

with st.spinner(f"Fetching 10-year financials for {symbol}..."):
    try:
        fin = _cached_fetch(symbol)
    except FetchError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error fetching {symbol}: {e}")
        st.stop()

st.subheader(f"{fin.company_name} ({fin.ticker})")
top1, top2, top3, top4, top5, top6 = st.columns(6)
top1.metric("Current Price", _fmt_money(fin.current_price))
top2.metric("Market Cap", _fmt_money(fin.market_cap))
top3.metric("TTM P/E", f"{fin.pe_ratio_ttm:.1f}" if fin.pe_ratio_ttm else "n/a")
top4.metric("Div Yield", _fmt_pct(fin.dividend_yield))
top5.metric("BVPS", _fmt_money(fin.book_value_per_share))
top6.metric("Years of data", str(fin.years_available))

# Data-source banner. EDGAR path is green (10+ yrs), yfinance-only is amber.
if fin.data_source == "edgar+yfinance":
    st.success(fin.data_source_note)
elif fin.data_source == "edgar":
    st.info(fin.data_source_note)
elif fin.data_source_note:
    st.warning(fin.data_source_note)

# EDGAR EPS is NOT split-adjusted (values are as-reported in each original 10-K),
# which can create phantom "collapses" in EPS growth around stock split years.
# Flag this to prevent the user from misreading the Big 5 EPS column.
if fin.data_source.startswith("edgar") and not fin.eps.empty and len(fin.eps) >= 3:
    ratios = fin.eps.diff() / fin.eps.shift(1).abs()
    if (ratios.abs() > 0.5).any():
        st.caption(
            ":warning: EDGAR reports EPS as-filed (not split-adjusted). "
            "Large year-over-year EPS jumps may reflect a stock split rather than "
            "actual earnings change. Cross-check with a split-adjusted source before "
            "acting on the EPS growth number."
        )

big5 = compute_big5(fin)

st.markdown("### The Big 5")
st.caption(
    "Phil Town's rule: every metric should compound at **10%+** across every window. "
    "ROIC is shown as a period-average return, not a growth rate."
)
_render_big5_table(big5)

overall_ok = big5.all_pass()
if overall_ok:
    st.success("All Big 5 pass in every available window. This is a Rule #1 candidate.")
else:
    failing = [
        m.label
        for m in (big5.roic, big5.sales, big5.eps, big5.equity, big5.fcf)
        if not m.passes
    ]
    st.warning("Failing the Big 5: " + ", ".join(failing))

st.markdown("### Valuation")

current_eps = float(fin.eps.iloc[-1]) if not fin.eps.empty else None
big5_eps_g = _big5_eps_growth(big5)
val = sticker_price(
    current_eps=current_eps,
    big5_eps_growth=big5_eps_g,
    analyst_growth=fin.analyst_5yr_growth,
    historical_pe=fin.pe_ratio_ttm,
)

if val is None:
    st.info(
        "Cannot compute Sticker Price — need positive current EPS and at least one "
        "growth estimate (Big 5 EPS growth or analyst 5yr growth)."
    )
else:
    v_verdict, v_color = _verdict_price(fin.current_price, val.mos_price, val.sticker_price)
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Sticker Price", _fmt_money(val.sticker_price))
    v2.metric("MOS Buy Price", _fmt_money(val.mos_price))
    v3.metric("Current Price", _fmt_money(fin.current_price))
    v4.metric("Verdict", v_verdict)

    with st.expander("Valuation assumptions"):
        st.write(
            {
                "current_eps": val.current_eps,
                "growth_rate": _fmt_pct(val.growth_rate),
                "growth_source": val.growth_source,
                "future_pe": round(val.future_pe, 2),
                "future_eps (yr 10)": round(val.future_eps, 2),
                "future_price": _fmt_money(val.future_price),
                "discount_rate": _fmt_pct(val.discount_rate),
                "horizon_years": val.horizon_years,
                "margin_of_safety": "50%",
            }
        )

st.markdown("### Other Intrinsic Value Methods")
st.caption(
    "Multiple valuation lenses on the same ticker. Each shows fair value, a margin-of-safety "
    "buy price, upside vs current, and a verdict. No single method is right — look for consensus."
)

_fcf_ttm_for_dcf = float(fin.free_cash_flow.iloc[-1]) if not fin.free_cash_flow.empty else None
_growth_for_intrinsic = big5_eps_g if big5_eps_g is not None else fin.analyst_5yr_growth

methods: list[MethodResult] = []

def _add(m: MethodResult | None):
    if m is not None:
        methods.append(m)

_add(
    dcf_two_stage(
        fcf_ttm=_fcf_ttm_for_dcf,
        shares_out=fin.shares_outstanding,
        current_price=fin.current_price,
        growth_rate=_growth_for_intrinsic,
        discount_rate=dcf_discount,
        terminal_growth=dcf_terminal,
    )
)
_add(
    peter_lynch_fair(
        current_eps=current_eps,
        growth_rate=_growth_for_intrinsic,
        dividend_yield=fin.dividend_yield,
        current_price=fin.current_price,
        mos=mos_pct,
    )
)
_add(
    graham_number(
        current_eps=current_eps,
        book_value_per_share=fin.book_value_per_share,
        current_price=fin.current_price,
        mos=mos_pct,
    )
)
_add(
    graham_formula(
        current_eps=current_eps,
        growth_rate=_growth_for_intrinsic,
        current_price=fin.current_price,
        aaa_bond_yield=aaa_yield,
        mos=mos_pct,
    )
)
_add(
    peg_fair_value(
        current_eps=current_eps,
        growth_rate=_growth_for_intrinsic,
        current_price=fin.current_price,
        mos=mos_pct,
    )
)

if not methods:
    st.info(
        "No intrinsic-value methods computable — need positive current EPS and at least "
        "one growth estimate (Big 5 EPS or analyst)."
    )
else:
    def _row(m: MethodResult) -> dict:
        return {
            "Method": m.name,
            "Fair Value": _fmt_money(m.fair_value),
            "MOS Buy": _fmt_money(m.mos_price),
            "Upside vs Current": _fmt_pct(m.upside_pct),
            "Verdict": m.verdict,
        }

    methods_df = pd.DataFrame([_row(m) for m in methods])

    def _style_row(row):
        v = row["Verdict"]
        if v == "BUY":
            return ["background-color: #1e4620; color: #b6f0b6;"] * len(row)
        if v == "WATCH":
            return ["background-color: #4a3a1e; color: #f0d8a6;"] * len(row)
        if v == "AVOID":
            return ["background-color: #4b1e1e; color: #f0b6b6;"] * len(row)
        return [""] * len(row)

    st.dataframe(
        methods_df.style.apply(_style_row, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    # Consensus verdict.
    buys = sum(1 for m in methods if m.verdict == "BUY")
    watches = sum(1 for m in methods if m.verdict == "WATCH")
    avoids = sum(1 for m in methods if m.verdict == "AVOID")
    total = buys + watches + avoids
    if total:
        if buys >= total / 2:
            st.success(f"Consensus: **BUY** ({buys}/{total} methods say below MOS)")
        elif avoids >= total / 2:
            st.error(f"Consensus: **AVOID** ({avoids}/{total} methods say above fair value)")
        else:
            st.warning(f"Consensus: **WATCH** ({buys} buy / {watches} watch / {avoids} avoid)")

    with st.expander("Per-method assumptions"):
        for m in methods:
            st.markdown(f"**{m.name}**")
            st.write(m.assumptions)

st.markdown("### Health checks")
fcf_ttm = float(fin.free_cash_flow.iloc[-1]) if not fin.free_cash_flow.empty else None
ltd_ttm = float(fin.long_term_debt.iloc[-1]) if not fin.long_term_debt.empty else None
big5_growth_for_payback = big5_eps_g if big5_eps_g is not None else fin.analyst_5yr_growth

pay = payback_time(fcf_ttm, big5_growth_for_payback, fin.market_cap)
dtf = debt_to_fcf(ltd_ttm, fcf_ttm)

h1, h2 = st.columns(2)
with h1:
    if pay is None:
        st.metric("Payback Time", "n/a")
        st.caption("Needs positive TTM FCF and a valid market cap.")
    else:
        st.metric("Payback Time", f"{pay:.0f} years", delta=("PASS" if pay <= 8 else "FAIL"))
        st.caption("Phil Town's rule: 8 years or fewer.")
with h2:
    if dtf is None:
        st.metric("Debt / FCF", "n/a")
        st.caption("Needs positive TTM FCF.")
    else:
        st.metric("Debt / FCF", f"{dtf:.2f}", delta=("PASS" if dtf < 3 else "FAIL"))
        st.caption("Phil Town's rule: less than 3 years to pay off all long-term debt.")

st.markdown("### Underlying data (annual)")
with st.expander("Show raw statement values used"):
    frames = {
        "Revenue": fin.revenue,
        "Net Income": fin.net_income,
        "EPS (Diluted)": fin.eps,
        "Equity": fin.equity,
        "Long-Term Debt": fin.long_term_debt,
        "Operating Cash Flow": fin.operating_cash_flow,
        "CapEx": fin.capex,
        "Free Cash Flow": fin.free_cash_flow,
        "ROIC (per year)": big5.roic_by_year,
    }
    combined = pd.DataFrame({k: v for k, v in frames.items() if not v.empty})
    combined.index.name = "Year"
    st.dataframe(combined.sort_index(ascending=False), use_container_width=True)

st.caption(
    "Data: Yahoo Finance via yfinance. Not investment advice. Cross-check with 10-K "
    "filings before any capital decision."
)
