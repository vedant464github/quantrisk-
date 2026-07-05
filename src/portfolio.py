import numpy as np
import pandas as pd
from scipy.optimize import minimize


from sklearn.covariance import LedoitWolf

def regime_covariances(returns_df, regime_series, min_obs=30, shrinkage=True):
    """
    Computes a separate covariance matrix per regime.

    shrinkage=True: uses Ledoit-Wolf shrinkage instead of the raw sample
    covariance. Matters most for regimes with few observations relative
    to the number of assets (e.g. Bear/Crisis with 105 obs for 10 assets,
    ~10:1 ratio) — raw sample covariance in that regime is noisy and prone
    to producing extreme corner-solution portfolios in the optimizer.
    Shrinkage pulls it toward a more stable, better-conditioned estimate.

    Regimes with plenty of data (e.g. Bull with 1098 obs) barely change
    under shrinkage — the sample covariance is already reliable there,
    so the shrinkage intensity the algorithm picks will be small.
    """
    common_idx = returns_df.index.intersection(regime_series.index)
    returns_aligned = returns_df.loc[common_idx]
    regime_aligned = regime_series.loc[common_idx]

    cov_matrices = {}
    shrinkage_intensities = {}

    for regime in regime_aligned.unique():
        mask = regime_aligned == regime
        n_obs = mask.sum()
        if n_obs < min_obs:
            print(f"Skipping '{regime}': only {n_obs} observations "
                  f"(need >= {min_obs}) — covariance estimate would be unstable")
            continue

        regime_returns = returns_aligned.loc[mask]

        if shrinkage:
            lw = LedoitWolf().fit(regime_returns.values)
            cov = pd.DataFrame(lw.covariance_ * 252,
                                index=regime_returns.columns,
                                columns=regime_returns.columns)
            shrinkage_intensities[regime] = lw.shrinkage_
            print(f"'{regime}': {n_obs} obs, shrinkage intensity={lw.shrinkage_:.3f}")
        else:
            cov = regime_returns.cov() * 252
            print(f"'{regime}': {n_obs} obs, covariance computed (no shrinkage)")

        cov_matrices[regime] = cov

    return cov_matrices, shrinkage_intensities


def min_variance_weights(cov_matrix, allow_short=False):
    """
    Solves for the minimum-variance portfolio: the weight allocation
    that minimizes portfolio variance = w^T @ Cov @ w, subject to
    weights summing to 1 (fully invested) and optionally no negative
    weights (no short-selling, allow_short=False is more realistic
    for a retail/most-institutional context).

    This is intentionally NOT max-Sharpe — mean-variance optimization
    that uses expected returns is extremely sensitive to estimation
    error in the mean (means are much harder to estimate reliably than
    covariances from historical data). Min-variance sidesteps that by
    ignoring expected returns entirely and just minimizing risk, which
    is more robust for a project where you don't want return forecasting
    noise to dominate the story.
    """
    n = cov_matrix.shape[0]

    def portfolio_variance(w):
        return w @ cov_matrix.values @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) if not allow_short else (-1, 1) for _ in range(n)]
    w0 = np.ones(n) / n  # start from equal weight

    result = minimize(portfolio_variance, w0, method="SLSQP",
                       bounds=bounds, constraints=constraints)

    if not result.success:
        print(f"WARNING: optimizer did not converge — {result.message}")

    weights = pd.Series(result.x, index=cov_matrix.columns)
    portfolio_vol = np.sqrt(portfolio_variance(result.x))
    return weights, portfolio_vol


def compare_regime_portfolios(returns_df, regime_series, min_obs=30, shrinkage=True):
    cov_matrices, shrinkage_intensities = regime_covariances(returns_df, regime_series, min_obs, shrinkage)

    static_cov = returns_df.cov() * 252
    cov_matrices["Static (full sample)"] = static_cov

    results = {}
    for regime, cov in cov_matrices.items():
        weights, vol = min_variance_weights(cov)
        results[regime] = {"weights": weights, "min_variance_vol": vol}

    summary = pd.DataFrame({
        regime: {"min_achievable_annual_vol": res["min_variance_vol"]}
        for regime, res in results.items()
    }).T

    return results, summary, shrinkage_intensities
