"""
Runs the full QuantRisk pipeline end-to-end and saves every output the
dashboard needs to artifacts/. This is the ONLY script that touches
yfinance, fits the HMM, fits GARCH, or runs the optimizer — app.py only
reads what's saved here. Re-run this manually whenever you want to
refresh the numbers (new data, code changes, etc); the dashboard itself
never recomputes.

Usage:
    python build_artifacts.py
"""
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import fetch_prices, compute_returns, TICKERS
from src.regime import build_hmm_features, fit_hmm, label_regimes, plot_regimes
from src.risk import (
    sharpe_ratio, sortino_ratio, max_drawdown, fit_garch,
    forecast_volatility, historical_cvar, parametric_cvar,
    get_fitted_nu, calculate_beta,
)
from src.portfolio import compare_regime_portfolios
from src.stress_test import CRISIS_WINDOWS, run_stress_test, regime_switching_backtest


START_DATE = "2006-01-01"
END_DATE = None  # None = up to today

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    print(f"Artifacts will be saved to {ARTIFACTS_DIR}\n")

    # ---------------------------------------------------------------
    # 1. Data
    # ---------------------------------------------------------------
    print("=== Fetching prices ===")
    prices = fetch_prices(tickers=TICKERS, start=START_DATE, end=END_DATE)
    returns_df = compute_returns(prices, method="log")

    ew_return = returns_df.mean(axis=1)
    ew_price = (1 + ew_return).cumprod()

    prices.to_parquet(ARTIFACTS_DIR / "prices.parquet")
    returns_df.to_parquet(ARTIFACTS_DIR / "returns.parquet")
    ew_return.to_frame("ew_return").to_parquet(ARTIFACTS_DIR / "ew_return.parquet")

    # ---------------------------------------------------------------
    # 2. Regime detection
    # ---------------------------------------------------------------
    print("\n=== Fitting HMM / detecting regimes ===")
    features = build_hmm_features(ew_return)
    model, hidden_states = fit_hmm(features)
    regime_series, labels = label_regimes(model, hidden_states, features)

    regime_series.to_frame("regime").to_parquet(ARTIFACTS_DIR / "regime_series.parquet")

    state_stats = {}
    for state in range(model.n_components):
        mask = hidden_states == state
        state_stats[labels[state]] = {
            "mean_return": float(features["return"][mask].mean()),
            "mean_volatility": float(features["volatility"][mask].mean()),
            "n_obs": int(mask.sum()),
        }
    with open(ARTIFACTS_DIR / "state_stats.json", "w") as f:
        json.dump(state_stats, f, indent=2)

    # save the regime plot image straight into artifacts for the dashboard
    plot_regimes(ew_price, regime_series, save_path=ARTIFACTS_DIR / "regime_plot.png")

    # ---------------------------------------------------------------
    # 3. Risk metrics
    # ---------------------------------------------------------------
    print("\n=== Computing risk metrics ===")
    sharpe = sharpe_ratio(ew_return)
    sortino = sortino_ratio(ew_return)
    dd = max_drawdown(ew_price)

    garch_fitted = fit_garch(ew_return)
    omega = float(garch_fitted.params["omega"])
    alpha = float(garch_fitted.params["alpha[1]"])
    beta_garch = float(garch_fitted.params["beta[1]"])
    persistence = alpha + beta_garch

    vol_forecast_5d = forecast_volatility(garch_fitted, horizon=5)
    fitted_nu = get_fitted_nu(garch_fitted)

    hist_cvar = historical_cvar(ew_return)
    current_vol_forecast = float(vol_forecast_5d[0])
    param_cvar = parametric_cvar(
        mean_return=float(ew_return.mean()),
        volatility=current_vol_forecast,
        dist="t",
        nu=float(fitted_nu),
    )

    print("\n=== Fetching Nifty 50 benchmark for beta ===")
    # NOTE: extended to START_DATE (2006) for consistency with the rest of
    # the pipeline. The original notebook hardcoded start="2015-01-01" here
    # from before the 2006 extension — beta was silently computed only over
    # the 2015-2026 overlap. If you want to preserve that original 0.938
    # figure exactly, change START_DATE below back to "2015-01-01".
    bench_prices = fetch_prices(tickers=["^NSEI"], start=START_DATE, end=END_DATE)
    bench_returns = compute_returns(bench_prices, method="log")["^NSEI"]
    portfolio_beta = calculate_beta(ew_return, bench_returns)

    risk_metrics = {
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(dd["max_drawdown"]),
        "max_drawdown_peak_date": str(dd["peak_date"].date()),
        "max_drawdown_trough_date": str(dd["trough_date"].date()),
        "garch": {
            "omega": omega,
            "alpha": alpha,
            "beta": beta_garch,
            "persistence": persistence,
            "fitted_nu": float(fitted_nu),
            "vol_forecast_5d": vol_forecast_5d.tolist(),
        },
        "historical_cvar_95": float(hist_cvar),
        "parametric_cvar_95": float(param_cvar),
        "portfolio_beta_vs_nifty50": float(portfolio_beta),
    }
    with open(ARTIFACTS_DIR / "risk_metrics.json", "w") as f:
        json.dump(risk_metrics, f, indent=2)

    # drawdown series is needed for a chart but too big for json — parquet it
    dd["drawdown_series"].to_frame("drawdown").to_parquet(
        ARTIFACTS_DIR / "drawdown_series.parquet"
    )

    # ---------------------------------------------------------------
    # 4. Regime-conditional portfolios
    # ---------------------------------------------------------------
    print("\n=== Computing regime-conditional min-variance portfolios ===")
    results, summary, shrinkage_intensities = compare_regime_portfolios(
        returns_df, regime_series
    )
    with open(ARTIFACTS_DIR / "portfolio_results.pkl", "wb") as f:
        pickle.dump(results, f)
    summary.to_parquet(ARTIFACTS_DIR / "portfolio_summary.parquet")
    with open(ARTIFACTS_DIR / "shrinkage_intensities.json", "w") as f:
        json.dump({k: float(v) for k, v in shrinkage_intensities.items()}, f, indent=2)

    # ---------------------------------------------------------------
    # 5. Stress test (static vs regime-conditional, summary stats)
    # ---------------------------------------------------------------
    print("\n=== Running stress test ===")
    stress_summary = run_stress_test(returns_df, regime_series, results, CRISIS_WINDOWS)
    stress_summary.to_parquet(ARTIFACTS_DIR / "stress_test_summary.parquet")

    # ---------------------------------------------------------------
    # 6. Regime-switching value-path backtest (per crisis window)
    # ---------------------------------------------------------------
    print("\n=== Running regime-switching value-path backtests ===")
    backtest_paths = {}
    for window_name, (start, end) in CRISIS_WINDOWS.items():
        backtest_paths[window_name] = regime_switching_backtest(
            returns_df, regime_series, results, start=start, end=end
        )
    with open(ARTIFACTS_DIR / "backtest_paths.pkl", "wb") as f:
        pickle.dump(backtest_paths, f)

    # ---------------------------------------------------------------
    # 7. Metadata
    # ---------------------------------------------------------------
    meta = {
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        "tickers": TICKERS,
        "start_date": START_DATE,
        "end_date": str(prices.index.max().date()),
        "n_observations": int(len(returns_df)),
    }
    with open(ARTIFACTS_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n=== Done. All artifacts saved to {ARTIFACTS_DIR} ===")


if __name__ == "__main__":
    main()