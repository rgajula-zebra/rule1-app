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

st.set_page_config(page_title="MoatCheck", page_icon=None, layout="wide")


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


def _verdict_price(current: float | None, mos: float | None, sticker: float | None) -> tuple[str, str, str]:
    """Return (headline, detail, color) for the verdict cell.

    Tiers (Phil Town + one extra):
      - BARGAIN BUY: current price is <= 50% of MOS Buy Price (already-half of MOS)
      - BUY:        current <= MOS Buy Price
      - WATCH:      current between MOS and Sticker (Intrinsic)
      - AVOID:      current > Sticker (Intrinsic)
    """
    if current is None or mos is None or sticker is None or sticker <= 0 or mos <= 0:
        return ("Unknown", "", "gray")
    if current <= mos * 0.5:
        pct = (mos - current) / mos * 100
        return ("BARGAIN BUY", f"{pct:.0f}% below MOS Buy Price — rare deep discount", "bargain")
    if current <= mos:
        pct = (mos - current) / mos * 100
        return ("BUY", f"{pct:.0f}% below MOS Buy Price", "green")
    if current <= sticker:
        pct = (sticker - current) / sticker * 100
        return ("WATCH", f"between MOS and Intrinsic Value ({pct:.0f}% below Intrinsic)", "orange")
    pct = (current - sticker) / sticker * 100
    return ("AVOID", f"above Intrinsic Value by {pct:.0f}%", "red")


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
            MoatCheck<span style="font-weight: 400; color: #9aa0a6; font-size: 1.6rem; margin-left: 0.5rem;">— Value Investing Stock Scanner</span>
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

# Data-source note is shown in the footer only — not up top. Split-artifact
# flag is precomputed here and rendered alongside the footer note.
_show_split_warning = False
if fin.data_source.startswith("edgar") and not fin.eps.empty and len(fin.eps) >= 3:
    ratios = fin.eps.diff() / fin.eps.shift(1).abs()
    _show_split_warning = bool((ratios.abs() > 0.5).any())

big5 = compute_big5(fin)

st.markdown(
    '### Big 5 Growth Rates — <span style="color: #00E676;">Is this a Wonderful Company?</span>',
    unsafe_allow_html=True,
)
st.caption(
    "As per Buffett, a wonderful business has a durable competitive advantage — a "
    '"moat". Ideally, all five metrics below should show historical growth rates of '
    "10% or more per year over the 10, 5, 3, and 1-year windows."
)
st.caption("Note: ROIC is shown as a period-average return, not a CAGR.")
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

_COLOR_HEX = {
    "bargain": "#00E676",
    "green": "#4CAF50",
    "orange": "#FFA726",
    "red": "#EF5350",
    "gray": "#9AA0A6",
}


_VERDICT_TOOLTIPS = {
    "BARGAIN BUY": (
        "Current price is at or below 50% of the MOS Buy Price — a rare deep "
        "discount. In Rule #1 terms this is a 'back up the truck' setup: the "
        "formula says you get a huge margin of safety on top of an already "
        "conservative fair value. BUT — verify the company is still healthy. "
        "Bargain prices often reflect real risks the market is pricing in "
        "(pending lawsuit, industry decline, accounting concerns). Check "
        "recent news and the Big 5 trend before buying."
    ),
    "BUY": (
        "Current price is below the Margin of Safety (MOS) Buy Price. This "
        "means the formula says you can buy today with the safety cushion "
        "Phil Town / Lynch / Graham built into their method. This is the "
        "signal each method was designed to produce — but it's still just "
        "one formula. Cross-check against the other valuation methods in "
        "the table below, verify the Big 5 are strong, and confirm the "
        "business quality before acting."
    ),
    "WATCH": (
        "Current price is between the MOS Buy Price and the Intrinsic Value. "
        "The formula thinks the stock is reasonably valued but not offering "
        "a margin of safety. Add to your watchlist and wait for a pullback "
        "toward MOS — Phil Town's discipline is patience. If fundamentals "
        "improve (Big 5 accelerating) or the stock drops, this can quickly "
        "become a BUY."
    ),
    "AVOID": (
        "AVOID here doesn't mean the stock is a bad buy in absolute terms — "
        "it only means the current price is above the intrinsic value "
        "produced by this specific formula (Rule #1 / Peter Lynch / etc.). "
        "These formulas are deliberately conservative (Phil Town caps growth "
        "at 15%, uses a 15% discount rate, and applies a 50% margin of "
        "safety). A stock flagged AVOID may still be a good long-term buy "
        "if you believe the company will outgrow the conservative "
        "assumptions, if you require a lower return than 15%, or if "
        "qualitative factors (moat, management, industry) justify paying "
        "above the formula price. Use these numbers as guardrails, not gospel."
    ),
    "Unknown": (
        "Not enough data to compute a verdict — usually missing current price "
        "or a computable fair value. Check the Data footnote at the bottom "
        "of the page."
    ),
}

# Section-level tooltip (used above the Other Intrinsic Value Methods table,
# where a per-cell tooltip isn't possible). Covers all four verdict outcomes.
_VERDICT_TOOLTIP_SECTION = (
    "Verdicts compare current price against each formula's fair value. "
    "BARGAIN BUY = current is <= 50% of MOS Buy Price (deep discount). "
    "BUY = current is below the MOS Buy Price. "
    "WATCH = between MOS and fair value. "
    "AVOID = above fair value — but each formula is deliberately conservative, "
    "so AVOID doesn't necessarily mean 'bad company' — it means 'the formula's "
    "guardrails say wait'. Cross-check with the Big 5 and the other methods."
)


def _render_verdict_cell(col, verdict: str, detail: str, color: str) -> None:
    # &#9432; is the ⓘ (info) glyph. HTML `title` gives us a native browser tooltip
    # on hover — works on all platforms without needing any Streamlit component.
    tip_raw = _VERDICT_TOOLTIPS.get(verdict, _VERDICT_TOOLTIPS["Unknown"])
    tip = tip_raw.replace('"', "&quot;")
    col.markdown(
        f"""
        <div style="color: rgba(250,250,250,0.6); font-size: 0.875rem;">
            Verdict
            <span title="{tip}" style="cursor: help; color: #4CAF50; margin-left: 4px; font-size: 0.9rem;">&#9432;</span>
        </div>
        <div style="font-size: 2.0rem; font-weight: 400; line-height: 1.2; color: {_COLOR_HEX[color]};">
            {verdict}
        </div>
        <div style="font-size: 0.72rem; color: #9aa0a6; line-height: 1.3; margin-top: 0.2rem;">
            {detail}
        </div>
        """,
        unsafe_allow_html=True,
    )


if val is None:
    st.info(
        "Cannot compute Sticker Price — need positive current EPS and at least one "
        "growth estimate (Big 5 EPS growth or analyst 5yr growth)."
    )
else:
    st.markdown("**Phil Town Sticker Price (Rule #1)**")
    v_verdict, v_detail, v_color = _verdict_price(fin.current_price, val.mos_price, val.sticker_price)
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Sticker Price aka Intrinsic Value", _fmt_money(val.sticker_price))
    v2.metric("Margin of Safety (MOS) Buy Price", _fmt_money(val.mos_price))
    v3.metric("Current Price", _fmt_money(fin.current_price))
    _render_verdict_cell(v4, v_verdict, v_detail, v_color)

    with st.expander("How the Sticker Price was calculated"):
        _big5_g_txt = _fmt_pct(big5_eps_g)
        _analyst_g_txt = _fmt_pct(fin.analyst_5yr_growth)
        _hist_pe_txt = f"{fin.pe_ratio_ttm:.1f}" if fin.pe_ratio_ttm else "n/a"
        _default_pe = val.growth_rate * 100 * 2
        _mos_pct_int = int(round(100 * (1 - val.mos_price / val.sticker_price))) if val.sticker_price else 50

        st.markdown(
            f"""
**Step 1 — Pick the growth rate (the lower of two, capped at 15%)**
- Big 5 lowest EPS growth: **{_big5_g_txt}**
- Analyst 5yr growth: **{_analyst_g_txt}**
- Chosen: **{_fmt_pct(val.growth_rate)}** *(source: {val.growth_source}, capped at 15%)*

**Step 2 — Estimate future P/E (the lower of two)**
- Historical TTM P/E: **{_hist_pe_txt}**
- 2 × growth rate: **{_default_pe:.1f}**
- Chosen future P/E: **{val.future_pe:.2f}**

**Step 3 — Project EPS 10 years forward**
- Current EPS × (1 + growth)^10
- = {val.current_eps:.2f} × (1 + {val.growth_rate:.4f})^10
- = **{val.future_eps:.2f}**

**Step 4 — Future stock price**
- Future EPS × Future P/E
- = {val.future_eps:.2f} × {val.future_pe:.2f}
- = **{_fmt_money(val.future_price)}**

**Step 5 — Discount back to today at {_fmt_pct(val.discount_rate)} (Phil Town's required return)**
- Future price ÷ (1 + {val.discount_rate:.2f})^{val.horizon_years}
- = **{_fmt_money(val.sticker_price)}**  ← *Sticker Price*

**Step 6 — Apply {_mos_pct_int}% Margin of Safety**
- Sticker × {1 - _mos_pct_int/100:.2f}
- = **{_fmt_money(val.mos_price)}**  ← *MOS Buy Price*
""".strip()
        )

# --- Peter Lynch Fair Value (elevated to the main Valuation section) ---
_growth_for_lynch = big5_eps_g if big5_eps_g is not None else fin.analyst_5yr_growth
lynch = peter_lynch_fair(
    current_eps=current_eps,
    growth_rate=_growth_for_lynch,
    dividend_yield=fin.dividend_yield,
    current_price=fin.current_price,
    mos=mos_pct,
)

st.markdown("**Peter Lynch Fair Value**")
if lynch is None:
    st.info(
        "Cannot compute Peter Lynch Fair Value — need positive current EPS and a "
        "positive growth estimate."
    )
else:
    l_verdict, l_detail, l_color = _verdict_price(
        fin.current_price, lynch.mos_price, lynch.fair_value
    )
    l1, l2, l3, l4 = st.columns(4)
    l1.metric("Fair Value", _fmt_money(lynch.fair_value))
    l2.metric(f"MOS Buy Price ({int(mos_pct*100)}% off)", _fmt_money(lynch.mos_price))
    l3.metric("Current Price", _fmt_money(fin.current_price))
    _render_verdict_cell(l4, l_verdict, l_detail, l_color)

    with st.expander("How the Peter Lynch Fair Value was calculated"):
        _g_pct = lynch.assumptions["growth_rate"] * 100
        _d_pct = lynch.assumptions["dividend_yield"] * 100
        _fair_pe = lynch.assumptions["fair_pe_used"]
        _sum_uncapped = _g_pct + _d_pct
        _cap_note = "" if _sum_uncapped <= 30 else " (capped at 30 — Lynch was skeptical of anything higher)"
        st.markdown(
            f"""
**Step 1 — Fair PE = growth rate (%) + dividend yield (%)**
- Growth rate: **{_g_pct:.1f}%** *(the Big 5 lowest EPS growth, or analyst 5yr if Big 5 unavailable)*
- Dividend yield: **{_d_pct:.2f}%**
- Sum: {_g_pct:.1f} + {_d_pct:.2f} = **{_sum_uncapped:.2f}**
- Fair PE used: **{_fair_pe:.2f}**{_cap_note}

**Step 2 — Fair Value = Current EPS × Fair PE**
- {lynch.assumptions["current_eps"]:.2f} × {_fair_pe:.2f}
- = **{_fmt_money(lynch.fair_value)}**  ← *Peter Lynch Fair Value*

**Step 3 — Apply {int(mos_pct*100)}% Margin of Safety** *(set by sidebar slider)*
- Fair Value × {1 - mos_pct:.2f}
- = **{_fmt_money(lynch.mos_price)}**  ← *MOS Buy Price*

_Lynch's rule of thumb: PEG < 1 = cheap for its growth, PEG > 2 = overpriced._
""".strip()
        )

_tip_other = _VERDICT_TOOLTIP_SECTION.replace('"', "&quot;")
st.markdown(
    f"""
    <h3 style="margin-bottom: 0.25rem;">
        Other Intrinsic Value Methods
        <span title="{_tip_other}" style="cursor: help; color: #4CAF50; font-size: 0.9rem; margin-left: 6px;">&#9432;</span>
    </h3>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Multiple valuation lenses on the same ticker. Each shows fair value, a margin-of-safety "
    "buy price, upside vs current, and a verdict. No single method is right — look for consensus. "
    "Hover the ⓘ icon for how to interpret the verdicts."
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
# Peter Lynch Fair Value is now shown in the main Valuation section above,
# so we don't repeat it in this comparison table.
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
    def _verdict_with_pct(m: MethodResult) -> str:
        """Append a small "(±NN%)" hint to the verdict cell.

        Sign convention: positive % = current price is BELOW fair (upside),
        negative = current price is ABOVE fair (overvalued).
        """
        if m.upside_pct is None:
            return m.verdict
        pct = m.upside_pct * 100
        if m.verdict == "BUY":
            return f"{m.verdict} ({pct:+.0f}% below fair)"
        if m.verdict == "WATCH":
            return f"{m.verdict} ({pct:+.0f}% vs fair)"
        if m.verdict == "AVOID":
            return f"{m.verdict} ({-pct:.0f}% above fair)"
        return m.verdict

    def _row(m: MethodResult) -> dict:
        return {
            "Method": m.name,
            "Fair Value": _fmt_money(m.fair_value),
            "MOS Buy": _fmt_money(m.mos_price),
            "Upside vs Current": _fmt_pct(m.upside_pct),
            "Verdict": _verdict_with_pct(m),
        }

    methods_df = pd.DataFrame([_row(m) for m in methods])

    def _style_row(row):
        v = row["Verdict"]
        if v.startswith("BUY"):
            return ["background-color: #1e4620; color: #b6f0b6;"] * len(row)
        if v.startswith("WATCH"):
            return ["background-color: #4a3a1e; color: #f0d8a6;"] * len(row)
        if v.startswith("AVOID"):
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

st.divider()

# Tiny disclaimer-style footer. Uses raw HTML so we can shrink below Streamlit's
# default caption font (which is already small but not "fine print" small).
_split_line = (
    "EDGAR reports EPS as-filed (not split-adjusted); large year-over-year "
    "EPS jumps may reflect a stock split rather than an earnings change. "
    if _show_split_warning else ""
)
st.markdown(
    f"""
    <div style="color: #6c6f75; font-size: 0.72rem; line-height: 1.5; margin-top: 0.5rem;">
        <em>
        {fin.data_source_note}
        {_split_line}
        Not investment advice. Cross-check with 10-K filings before any capital decision.
        </em>
    </div>
    """,
    unsafe_allow_html=True,
)
