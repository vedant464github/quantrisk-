"""
QuantRisk dashboard. Reads ONLY from artifacts/ (produced by
build_artifacts.py) — never calls yfinance, never refits the HMM/GARCH,
never re-runs the optimizer. Re-run build_artifacts.py separately to
refresh the numbers shown here.
"""
import json
import pickle
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

REGIME_COLORS = {
    "Bull": "#2ca02c",
    "Neutral": "#7f7f7f",
    "High-Vol": "#ff7f0e",
    "Bear/Crisis": "#d62728",
}

st.set_page_config(page_title="QuantRisk", layout="wide")


@st.cache_data
def load_json(name):
    with open(ARTIFACTS_DIR / name) as f:
        return json.load(f)


@st.cache_data
def load_parquet(name):
    return pd.read_parquet(ARTIFACTS_DIR / name)


@st.cache_resource
def load_pickle(name):
    with open(ARTIFACTS_DIR / name, "rb") as f:
        return pickle.load(f)


meta = load_json("meta.json")
state_stats = load_json("state_stats.json")
risk_metrics = load_json("risk_metrics.json")
shrinkage = load_json("shrinkage_intensities.json")
portfolio_summary = load_parquet("portfolio_summary.parquet")
stress_summary = load_parquet("stress_test_summary.parquet")
portfolio_results = load_pickle("portfolio_results.pkl")
backtest_paths = load_pickle("backtest_paths.pkl")

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("QuantRisk — Regime-Aware Portfolio Risk Engine")
st.caption(
    f"10 Nifty constituents · {meta['start_date']} to {meta['end_date']} · "
    f"{meta['n_observations']:,} trading days · "
    f"last updated {meta['last_updated_utc'][:16].replace('T', ' ')} UTC"
)
st.markdown(
    "A 4-state HMM detects market regimes (Bull / Neutral / High-Vol / "
    "Bear-Crisis) from portfolio return and rolling volatility, then "
    "builds regime-conditional minimum-variance portfolios using "
    "Ledoit-Wolf shrunk covariance. Stress-tested against GFC, COVID, "
    "and the 2022 rate shock."
)
st.divider()

# ---------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------
st.header("1. Detected Market Regimes")

col1, col2 = st.columns([3, 2])
with col1:
    st.image(str(ARTIFACTS_DIR / "regime_plot.png"), use_container_width=True)

with col2:
    st.subheader("Regime characteristics")
    stats_rows = []
    for regime, s in state_stats.items():
        stats_rows.append({
            "Regime": regime,
            "Mean daily return": f"{s['mean_return']:.4%}",
            "Mean annualized vol": f"{s['mean_volatility']:.1%}",
            "Days observed": s["n_obs"],
        })
    stats_df = pd.DataFrame(stats_rows).set_index("Regime")
    # order rows Bull -> Neutral -> High-Vol -> Bear/Crisis for readability
    order = [r for r in REGIME_COLORS if r in stats_df.index]
    st.dataframe(stats_df.loc[order], use_container_width=True)
    st.caption(
        "Regimes are labeled by splitting HMM states into calm vs. "
        "stressed by mean volatility first, then ranking by return "
        "within each half — not by return alone, which would conflate "
        "a slow low-vol decline with a sharp high-vol crash."
    )

st.divider()

# ---------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------
st.header("2. Risk Metrics")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Sharpe Ratio", f"{risk_metrics['sharpe_ratio']:.3f}")
m2.metric("Sortino Ratio", f"{risk_metrics['sortino_ratio']:.3f}")
m3.metric("Max Drawdown", f"{risk_metrics['max_drawdown']:.1%}")
m4.metric("Beta vs Nifty 50", f"{risk_metrics['portfolio_beta_vs_nifty50']:.3f}")
st.caption(
    f"Max drawdown: peak {risk_metrics['max_drawdown_peak_date']} → "
    f"trough {risk_metrics['max_drawdown_trough_date']} (GFC). "
    f"Beta computed against Nifty 50 data from 2007-09 onward "
    f"(earliest available index history)."
)

st.subheader("GARCH(1,1) Volatility Model")
g = risk_metrics["garch"]
g1, g2, g3 = st.columns(3)
g1.metric("Persistence (α+β)", f"{g['persistence']:.4f}")
g2.metric("Fitted ν (Student's-t)", f"{g['fitted_nu']:.2f}")
g3.metric("5-day vol forecast", f"{g['vol_forecast_5d'][-1] * 100:.2f}% / day")

st.subheader("CVaR (95%, daily)")
c1, c2 = st.columns(2)
c1.metric("Historical CVaR", f"{risk_metrics['historical_cvar_95']:.2%}")
c2.metric("Parametric CVaR (GARCH + fitted-ν t-dist)", f"{risk_metrics['parametric_cvar_95']:.2%}")
st.caption(
    "Parametric CVaR is lower in magnitude than historical CVaR, "
    "consistent with GARCH indicating current volatility is calmer "
    "than the full-history average."
)

st.divider()

# ---------------------------------------------------------------------
# Regime-conditional portfolios
# ---------------------------------------------------------------------
st.header("3. Regime-Conditional Minimum-Variance Portfolios")

col1, col2 = st.columns([2, 3])
with col1:
    st.subheader("Min-achievable volatility by regime")
    display_summary = portfolio_summary.copy()
    display_summary["min_achievable_annual_vol"] = display_summary[
        "min_achievable_annual_vol"
    ].map("{:.1%}".format)
    order_with_static = [r for r in REGIME_COLORS if r in display_summary.index] + [
        "Static (full sample)"
    ]
    st.dataframe(display_summary.loc[order_with_static], use_container_width=True)
    st.caption(
        "Min-achievable volatility in Bear/Crisis is roughly 3.5x Bull — "
        "diversification breaks down exactly when it's needed most, as "
        "correlations spike during crises."
    )

with col2:
    st.subheader("Portfolio weights: Bull vs Bear/Crisis")
    fig, ax = plt.subplots(figsize=(7, 4))
    bull_w = portfolio_results["Bull"]["weights"]
    crisis_w = portfolio_results["Bear/Crisis"]["weights"]
    tickers = bull_w.index
    x = range(len(tickers))
    width = 0.35
    ax.bar([i - width / 2 for i in x], bull_w.values, width, label="Bull", color=REGIME_COLORS["Bull"])
    ax.bar([i + width / 2 for i in x], crisis_w.values, width, label="Bear/Crisis", color=REGIME_COLORS["Bear/Crisis"])
    ax.set_xticks(list(x))
    ax.set_xticklabels([t.replace(".NS", "") for t in tickers], rotation=45, ha="right")
    ax.set_ylabel("Weight")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    st.caption(
        "Crisis-regime weights concentrate in defensives (pharma) and IT "
        "services, while financials and cyclicals go to ~0 — a sector "
        "rotation the model found in the data, not diversification alone."
    )

with st.expander("Shrinkage intensities by regime"):
    shrink_df = pd.DataFrame.from_dict(shrinkage, orient="index", columns=["shrinkage_intensity"])
    shrink_df["shrinkage_intensity"] = shrink_df["shrinkage_intensity"].map("{:.3f}".format)
    st.dataframe(shrink_df, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------
st.header("4. Stress Test: Static vs Regime-Conditional")
st.markdown(
    "Realized performance during three historical crises, comparing a "
    "fixed static min-variance allocation against dynamically switching "
    "to that day's regime-specific min-variance weights."
)

stress_display = stress_summary.copy()
stress_display["total_return"] = stress_display["total_return"].map("{:.2%}".format)
stress_display["annualized_vol"] = stress_display["annualized_vol"].map("{:.2%}".format)
stress_display["max_drawdown"] = stress_display["max_drawdown"].map("{:.2%}".format)
stress_display["regime_switches"] = stress_display["regime_switches"].map(
    lambda x: "-" if pd.isna(x) else f"{int(x)}"
)
st.dataframe(stress_display, use_container_width=True)

st.caption(
    "GFC and COVID: regime-conditional improves return, vol, and "
    "drawdown, with very few switches (2-3), indicating a stable, "
    "high-conviction signal. 2022 rate shock: results roughly tie "
    "static allocation, with 14 switches over the year — the regime "
    "detector is less discriminating during grinding, moderate-severity "
    "corrections than during sharp, discrete crises."
)

st.divider()

# ---------------------------------------------------------------------
# Regime-switching value-path backtest
# ---------------------------------------------------------------------
st.header("5. Regime-Switching Backtest: Value Over Time")
st.warning(
    "**Not a live-tradable backtest.** Regime-specific weights are "
    "computed from each regime's full-history covariance, and regime "
    "labels come from a smoothing pass (Viterbi) over the entire fitted "
    "series — both use information not available to an investor in "
    "real time. This shows how much regime-awareness would have helped "
    "**in hindsight**, not what a live system would have achieved."
)

window_choice = st.selectbox("Crisis window", list(backtest_paths.keys()))
bt = backtest_paths[window_choice]

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(bt["regime_conditional_value"], label="Regime-Conditional", color="#1f77b4")
ax.plot(bt["equal_weight_value"], label="Equal-Weight Buy & Hold", color="#7f7f7f")
for d in bt["switch_dates"]:
    ax.axvline(d, color="red", alpha=0.25, linestyle="--", linewidth=1)
ax.set_title(f"{window_choice}: Portfolio Value")
ax.legend()
plt.tight_layout()
st.pyplot(fig)

final_rc = bt["regime_conditional_value"].iloc[-1]
final_ew = bt["equal_weight_value"].iloc[-1]
b1, b2 = st.columns(2)
b1.metric("Regime-Conditional final value", f"{final_rc:.3f}", f"{(final_rc - 1):.1%}")
b2.metric("Equal-Weight final value", f"{final_ew:.3f}", f"{(final_ew - 1):.1%}")
st.caption(
    "Benchmarked against naive equal-weight here (vs. static "
    "min-variance in the stress-test table above) — regime-conditional "
    "beats the weaker baseline more decisively than it beats the "
    "stronger one."
)

st.divider()
st.caption(
    "Built as a placement portfolio project. "
    "Source: github.com/<your-repo>"
)