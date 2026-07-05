import numpy as np
import pandas as pd
from arch import arch_model

def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    returns: daily simple or log returns (pd.Series)
    risk_free_rate: ANNUAL risk-free rate (e.g. 0.07 for 7% — use the
        current Indian 10Y G-Sec yield or 91-day T-bill rate as a proxy)
    periods_per_year: 252 for daily trading data

    Intuition: mean daily excess return, annualized by multiplying by
    periods_per_year, divided by annualized volatility (daily std * sqrt(252)).
    The sqrt(252) scaling comes from variance scaling linearly with time
    under i.i.d. return assumptions — std scales with sqrt(time).
    """
    daily_rf = risk_free_rate / periods_per_year
    excess_returns = returns - daily_rf

    mean_excess = excess_returns.mean() * periods_per_year
    vol = excess_returns.std() * np.sqrt(periods_per_year)

    if vol == 0:
        return np.nan
    return mean_excess / vol


def max_drawdown(price_series):
    """
    Maximum peak-to-trough decline, as a fraction (e.g. -0.38 = -38%).

    Intuition: at every point in time, compare current price to the
    highest price seen so far (the "running peak" / cumulative max).
    The drawdown at time t is how far below that peak you currently are.
    Max drawdown is the worst (most negative) such value across the
    whole series — the deepest hole the portfolio ever fell into.

    Returns a dict with the drawdown value plus the dates of the peak
    and trough, since "how deep" alone is less useful than "how deep,
    and when" for stress-test narrative later.
    """
    running_max = price_series.cummax()
    drawdown = (price_series - running_max) / running_max

    trough_date = drawdown.idxmin()
    max_dd = drawdown.loc[trough_date]

    # peak is the running max value *before* the trough, so find the
    # date where price last equaled that running max prior to the trough
    peak_value = running_max.loc[trough_date]
    peak_date = price_series[:trough_date][price_series[:trough_date] == peak_value].index[-1]

    return {
        "max_drawdown": max_dd,
        "peak_date": peak_date,
        "trough_date": trough_date,
        "drawdown_series": drawdown,  
    }


def sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Like Sharpe, but only penalizes downside volatility, not upside.

    Intuition: Sharpe treats a big positive day the same as a big negative
    day when computing volatility (both increase std). Sortino argues
    upside swings aren't "risk" an investor cares about — only downside
    is. Computed identically to Sharpe except the denominator only uses
    the std of negative excess returns.
    """
    daily_rf = risk_free_rate / periods_per_year
    excess_returns = returns - daily_rf

    downside_returns = excess_returns[excess_returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(periods_per_year)

    mean_excess = excess_returns.mean() * periods_per_year

    if downside_vol == 0:
        return np.nan
    return mean_excess / downside_vol

def fit_garch(returns, p=1, q=1, dist="t"):
    """
    Fits a GARCH(p,q) model to a return series and forecasts next-day volatility.

    returns: daily returns as PERCENTAGES, not decimals (arch package
        convention — e.g. pass returns * 100, not raw 0.01-scale returns).
        This isn't optional; fitting on raw decimal returns causes the
        optimizer to struggle with numerical scale and often fails to
        converge or gives nonsensical parameter estimates.

    dist="t": use Student's t distribution for the innovations, not
        Normal. Equity returns have fatter tails than a Normal distribution
        predicts (extreme moves happen more often than Gaussian assumes) —
        using "t" instead of the arch default "normal" gives more realistic
        tail behavior, which matters directly for CVaR later since CVaR
        is specifically about tail risk.

    Returns the fitted model and a summary of ω, α, β so you can sanity
    check: α + β should be < 1 (stationarity condition — if it's >= 1,
    volatility shocks never decay, which is not realistic for real markets)
    and typically α + β lands somewhere in the 0.90-0.99 range for daily
    equity data (volatility is highly persistent but does mean-revert
    eventually).
    """
    returns_pct = returns * 100
    model = arch_model(returns_pct, vol="GARCH", p=p, q=q, dist=dist)
    fitted = model.fit(disp="off")

    params = fitted.params
    omega, alpha, beta = params["omega"], params[f"alpha[1]"], params[f"beta[1]"]
    persistence = alpha + beta

    print(f"GARCH({p},{q}) fitted — omega={omega:.4f}, alpha={alpha:.4f}, "
          f"beta={beta:.4f}, persistence (alpha+beta)={persistence:.4f}")
    if persistence >= 1:
        print("WARNING: alpha + beta >= 1 — model is non-stationary, "
              "volatility shocks don't decay. Check for data issues "
              "(outliers, wrong scale) or try a different distribution.")

    return fitted


def forecast_volatility(fitted_model, horizon=1):
    """
    Forecasts volatility (as a decimal, e.g. 0.015 = 1.5% daily vol)
    for `horizon` days ahead.

    Converts back from the percentage scale used internally by arch
    (see fit_garch) to decimal, and from variance to volatility (std)
    by taking the square root — the model forecasts variance internally,
    volatility is what you actually want for risk metrics.
    """
    forecast = fitted_model.forecast(horizon=horizon)
    variance_forecast = forecast.variance.values[-1]  # last row = most recent forecast
    volatility_forecast_pct = np.sqrt(variance_forecast)
    volatility_forecast = volatility_forecast_pct / 100  # back to decimal scale
    return volatility_forecast

def historical_cvar(returns, confidence_level=0.95):
    """
    Historical (non-parametric) CVaR: directly uses the empirical
    distribution of past returns, no distributional assumption.

    Sorts returns, finds the VaR cutoff (the (1-confidence_level)
    percentile), then averages everything worse than that cutoff.

    e.g. confidence_level=0.95 means: look at the worst 5% of days
    in history, and average their returns. That average IS the CVaR.

    Limitation: purely backward-looking — assumes the future's tail
    risk looks like the past's. Doesn't use GARCH at all.
    """
    var_cutoff = returns.quantile(1 - confidence_level)
    tail_losses = returns[returns <= var_cutoff]
    cvar = tail_losses.mean()
    return cvar


def parametric_cvar(mean_return, volatility, confidence_level=0.95, dist="normal", nu=5):
    """
    Parametric CVaR: assumes returns follow a distribution (Normal or
    Student's t) with a given mean and volatility, then computes CVaR
    analytically from that distribution's formula — rather than reading
    it off historical data.

    This is where GARCH plugs in: pass the GARCH-forecasted volatility
    as `volatility` instead of historical std, and you get a CVaR that
    reflects *current* market conditions (e.g. elevated if we're in a
    high-vol regime right now) rather than an average over all history.

    dist="t" uses Student's t (fatter tails, more realistic for equity
    returns) with nu degrees of freedom — lower nu = fatter tails.
    GARCH fits often report a fitted nu; you can pass that here for
    consistency instead of guessing nu=5.
    """
    from scipy import stats

    alpha = 1 - confidence_level

    if dist == "normal":
        z = stats.norm.ppf(alpha)
        cvar_z = -stats.norm.pdf(z) / alpha
        cvar = mean_return + volatility * cvar_z
    elif dist == "t":
        t_ppf = stats.t.ppf(alpha, nu)
        cvar_factor = -stats.t.pdf(t_ppf, nu) / alpha * (nu + t_ppf**2) / (nu - 1)
        cvar = mean_return + volatility * cvar_factor
    else:
        raise ValueError("dist must be 'normal' or 't'")

    return cvar

def get_fitted_nu(fitted_garch_model):
    """
    Extracts the fitted degrees-of-freedom (nu) from a GARCH model fit
    with dist='t'. Using the model's own fitted nu instead of a generic
    guess keeps the parametric CVaR consistent with what the GARCH fit
    actually found about this data's tail heaviness, rather than
    assuming an arbitrary fatness.
    """
    return fitted_garch_model.params["nu"]

def calculate_beta(portfolio_returns, benchmark_returns):
    """
    Beta: portfolio's sensitivity to benchmark (market) moves.
    beta = Cov(portfolio, benchmark) / Var(benchmark)

    Interpretation: beta=1 means portfolio moves 1:1 with the market on
    average. beta=1.3 means it amplifies market moves by 30% (more
    aggressive/volatile than the market). beta=0.7 means it dampens
    market moves (more defensive).

    Needs a benchmark series aligned to the same dates as your portfolio
    — Nifty 50 index itself (^NSEI on yfinance) is the natural choice
    here since your 10 stocks are all Nifty constituents.
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner").dropna()
    port_ret, bench_ret = aligned.iloc[:, 0], aligned.iloc[:, 1]

    covariance = port_ret.cov(bench_ret)
    benchmark_variance = bench_ret.var()

    return covariance / benchmark_variance