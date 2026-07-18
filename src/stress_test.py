import numpy as np
import pandas as pd


CRISIS_WINDOWS = {
    "GFC (2008-09)": ("2008-01-01", "2009-03-31"),
    "COVID Crash (2020)": ("2020-02-01", "2020-04-30"),
    "2022 Rate Shock": ("2022-01-01", "2022-12-31"),
}


def compute_drawdown(cum_returns: pd.Series):
    """
    cum_returns: cumulative return series (e.g. (1+r).cumprod()), NOT raw returns.
    Returns (max_drawdown, peak_date, trough_date).
    """
    running_max = cum_returns.cummax()
    drawdown = cum_returns / running_max - 1
    trough_date = drawdown.idxmin()
    max_dd = drawdown.loc[trough_date]
    peak_date = cum_returns.loc[:trough_date].idxmax()
    return max_dd, peak_date, trough_date


def realized_stats(portfolio_returns: pd.Series):
    """
    portfolio_returns: daily simple returns of the portfolio (already
    weighted), for a single window. Returns dict of cumulative return,
    annualized vol (realized, not model-implied), and max drawdown.
    """
    cum_returns = (1 + portfolio_returns).cumprod()
    total_return = cum_returns.iloc[-1] - 1
    ann_vol = portfolio_returns.std() * np.sqrt(252)
    max_dd, peak_date, trough_date = compute_drawdown(cum_returns)

    return {
        "total_return": total_return,
        "annualized_vol": ann_vol,
        "max_drawdown": max_dd,
        "peak_date": peak_date,
        "trough_date": trough_date,
    }


def static_portfolio_returns(returns_df: pd.DataFrame, static_weights: pd.Series,
                              start: str, end: str):
    """
    Realized daily returns of a fixed weight vector held over [start, end].
    static_weights should come from results["Static (full sample)"]["weights"].
    """
    window_returns = returns_df.loc[start:end]
    w = static_weights.reindex(window_returns.columns)
    portfolio_returns = window_returns @ w
    return portfolio_returns


def regime_conditional_returns(returns_df: pd.DataFrame, regime_series: pd.Series,
                                regime_weights: dict, start: str, end: str):
    """
    Realized daily returns using each day's regime-specific min-variance
    weights. regime_weights: {regime_name: weights Series}, i.e.
    {r: results[r]["weights"] for r in results} excluding "Static (full sample)".

    A day whose regime has no corresponding weight vector (e.g. it fell
    below min_obs and was skipped in regime_covariances) is dropped from
    the realized series rather than silently assigned zero return, so
    gaps are visible instead of biasing the stats.
    """
    common_idx = returns_df.index.intersection(regime_series.index)
    window_idx = common_idx[(common_idx >= start) & (common_idx <= end)]

    daily_returns = []
    dropped_days = []
    for date in window_idx:
        regime = regime_series.loc[date]
        if regime not in regime_weights:
            dropped_days.append(date)
            continue
        w = regime_weights[regime].reindex(returns_df.columns)
        day_ret = returns_df.loc[date] @ w
        daily_returns.append((date, day_ret))

    if dropped_days:
        print(f"  NOTE: {len(dropped_days)} day(s) dropped — regime had no "
              f"weight vector (likely skipped for min_obs in regime_covariances)")

    portfolio_returns = pd.Series(
        [r for _, r in daily_returns],
        index=[d for d, _ in daily_returns],
        name="regime_conditional_return"
    )
    return portfolio_returns


def count_regime_switches(regime_series: pd.Series, start: str, end: str):
    """
    Number of days within the window where the regime differs from the
    prior day — proxy for how often the regime-conditional strategy would
    need to rebalance. Not a full transaction-cost model, but enough to
    flag if the strategy is switching so often that realistic costs would
    eat into the headline comparison.
    """
    window_regimes = regime_series.loc[start:end]
    switches = (window_regimes != window_regimes.shift()).sum() - 1
    return max(switches, 0)

def equal_weight_returns(returns_df: pd.DataFrame, start: str, end: str):
    """
    Realized daily returns of a naive equal-weight buy-and-hold portfolio
    over [start, end] — the benchmark for the regime-switching value path,
    as distinct from the static MIN-VARIANCE benchmark used elsewhere in
    stress_test.py. Equal-weight requires no optimization or covariance
    estimate at all, so it's a cleaner "did the regime signal do anything"
    baseline than static min-variance, which already embeds some
    diversification benefit of its own.
    """
    window_returns = returns_df.loc[start:end]
    n = window_returns.shape[1]
    w = pd.Series(1.0 / n, index=window_returns.columns)
    return window_returns @ w
 
 
def get_regime_switch_dates(regime_series: pd.Series, start: str, end: str):
    """
    Dates within [start, end] where the regime label differs from the
    prior day — used to mark rebalance points on the value-path chart.
    """
    window_regimes = regime_series.loc[start:end]
    changed = window_regimes.ne(window_regimes.shift())
    changed.iloc[0] = False  # first day isn't a "switch", it's the start
    return window_regimes.index[changed]
 
 
def regime_switching_backtest(returns_df: pd.DataFrame, regime_series: pd.Series,
                               results: dict, start: str, end: str, initial_value=1.0):
    """
    Simulates portfolio VALUE over time (not just summary stats) for two
    strategies over [start, end]:
      - "regime_conditional": rebalances to that day's regime's
        min-variance weights whenever the HMM-detected regime changes
      - "equal_weight": naive buy-and-hold benchmark, no rebalancing
 
    Returns a dict with both value paths (pd.Series indexed by date,
    starting at initial_value) and the list of regime-switch dates, ready
    to plot as a line chart with switch markers.
 
    IMPORTANT — two look-ahead sources, both inherited from upstream
    modules, neither introduced here:
      1. Weight look-ahead: each regime's min-variance weights come from
         regime_covariances(), fit on that regime's FULL-SAMPLE history
         (all 20 years), not just data available up to each rebalance
         date. A live investor on, say, 2020-03-09 would not have had
         access to the same regime-conditional weights computed here.
      2. Label look-ahead: regime_series comes from model.predict(), a
         Viterbi smoothing pass over the ENTIRE fitted series at once —
         the regime label on any given day can be informed by data from
         after that day. A live investor would face detection lag that
         this backtest does not model.
    This is therefore a "how much would regime-awareness have helped in
    hindsight" comparison, not a live-tradable strategy backtest. State
    this explicitly wherever the resulting chart is shown.
    """
    regime_weights = {r: v["weights"] for r, v in results.items()
                       if r != "Static (full sample)"}
 
    dynamic_ret = regime_conditional_returns(returns_df, regime_series,
                                              regime_weights, start, end)
    benchmark_ret = equal_weight_returns(returns_df, start, end)
    # align benchmark to the same (possibly gap-dropped) dates as dynamic_ret
    benchmark_ret = benchmark_ret.reindex(dynamic_ret.index)
 
    dynamic_value = initial_value * (1 + dynamic_ret).cumprod()
    benchmark_value = initial_value * (1 + benchmark_ret).cumprod()
 
    switch_dates = get_regime_switch_dates(regime_series, start, end)
    switch_dates = switch_dates.intersection(dynamic_value.index)
 
    return {
        "regime_conditional_value": dynamic_value,
        "equal_weight_value": benchmark_value,
        "switch_dates": switch_dates,
    }


def run_stress_test(returns_df: pd.DataFrame, regime_series: pd.Series, results: dict,
                     windows: dict = CRISIS_WINDOWS):
    """
    results: output of compare_regime_portfolios()[0] — dict of
    {regime_name: {"weights": Series, "min_variance_vol": float}},
    including the "Static (full sample)" key.

    Returns a DataFrame comparing static vs regime-conditional realized
    performance across each crisis window.
    """
    static_weights = results["Static (full sample)"]["weights"]
    regime_weights = {r: v["weights"] for r, v in results.items()
                       if r != "Static (full sample)"}

    rows = []
    for window_name, (start, end) in windows.items():
        print(f"\n--- {window_name} ({start} to {end}) ---")

        static_ret = static_portfolio_returns(returns_df, static_weights, start, end)
        static_stats = realized_stats(static_ret)

        dynamic_ret = regime_conditional_returns(returns_df, regime_series,
                                                   regime_weights, start, end)
        if len(dynamic_ret) == 0:
            print("  WARNING: no valid regime-labeled days in this window — skipping dynamic stats")
            continue
        dynamic_stats = realized_stats(dynamic_ret)

        n_switches = count_regime_switches(regime_series, start, end)

        rows.append({
            "window": window_name,
            "strategy": "Static",
            "total_return": static_stats["total_return"],
            "annualized_vol": static_stats["annualized_vol"],
            "max_drawdown": static_stats["max_drawdown"],
            "regime_switches": np.nan,
        })
        rows.append({
            "window": window_name,
            "strategy": "Regime-Conditional",
            "total_return": dynamic_stats["total_return"],
            "annualized_vol": dynamic_stats["annualized_vol"],
            "max_drawdown": dynamic_stats["max_drawdown"],
            "regime_switches": n_switches,
        })

        print(f"  Static:              return={static_stats['total_return']:.2%}  "
              f"vol={static_stats['annualized_vol']:.2%}  "
              f"max_dd={static_stats['max_drawdown']:.2%}")
        print(f"  Regime-Conditional:  return={dynamic_stats['total_return']:.2%}  "
              f"vol={dynamic_stats['annualized_vol']:.2%}  "
              f"max_dd={dynamic_stats['max_drawdown']:.2%}  "
              f"switches={n_switches}")

    summary = pd.DataFrame(rows).set_index(["window", "strategy"])
    return summary