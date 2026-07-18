import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.patches as mpatches


def build_hmm_features(returns: pd.Series, vol_window=21):
    """
    Features for HMM: use return + rolling vol of a single asset or index.
    We use a single series (e.g. Nifty proxy or equal-weight portfolio return)
    because HMM works best on 1D or low-dim input.
    """
    roll_vol = returns.rolling(vol_window).std() * np.sqrt(252)
    features = pd.DataFrame({
        "return": returns,
        "volatility": roll_vol
    }).dropna()
    return features

import numpy as np

def fit_hmm(features, n_states=4, n_init=10, random_state=42,
            covariance_type="diag", min_covar=1e-3):
    """
    Standardizes features before fitting: return (~1e-2 scale) and rolling
    vol (different scale/units) bias EM's covariance estimation if left
    unscaled — this gets worse once you add more features later (skew,
    volume z-scores, etc).

    Runs n_init random restarts, keeps the highest log-likelihood fit.
    Baum-Welch finds a local optimum; a bad init can converge to a
    degenerate state (one state fits ~3 points, variance -> 0, covariance
    matrix goes singular = your LinAlgError).
    """
    X = features.values if hasattr(features, "values") else np.asarray(features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_model, best_score = None, -np.inf
    for seed in range(random_state, random_state + n_init):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            min_covar=min_covar,
            n_iter=1000,
            tol=1e-4,
            random_state=seed,
        )
        try:
            model.fit(X_scaled)
            score = model.score(X_scaled)
            if score > best_score:
                best_score, best_model = score, model
        except Exception:
            continue  # this seed collapsed into a singular covariance, skip it

    if best_model is None:
        raise RuntimeError("All HMM inits failed — check features for NaNs/near-constant columns")

    hidden_states = best_model.predict(X_scaled)
    best_model.scaler_ = scaler  # stash it — you'll need this to unscale means_ for labeling
    return best_model, hidden_states


def label_regimes(model, hidden_states, features):
    """
    Labels states via 2D (return, volatility) classification, not return
    rank alone. Ranking by return alone conflates two distinct "bad"
    regimes: a low-vol grinding decline (2008-style) can post a more
    negative mean return than a short, violent, high-vol crash (COVID-
    style), even though the latter is what a risk engine should flag as
    Bear/Crisis. Fix: split states into calm (bottom half by mean vol)
    vs stressed (top half by mean vol) first, then rank by return only
    within each half.
    """
    state_stats = {}
    for state in range(model.n_components):
        mask = hidden_states == state
        state_stats[state] = {
            "return": features["return"][mask].mean(),
            "vol": features["volatility"][mask].mean(),
        }

    states_by_vol = sorted(state_stats, key=lambda s: state_stats[s]["vol"])
    half = len(states_by_vol) // 2
    calm_states = states_by_vol[:half]
    stressed_states = states_by_vol[half:]

    calm_sorted = sorted(calm_states, key=lambda s: state_stats[s]["return"], reverse=True)
    stressed_sorted = sorted(stressed_states, key=lambda s: state_stats[s]["return"], reverse=True)

    labels = {}
    calm_names = ["Bull", "Neutral"]
    stressed_names = ["High-Vol", "Bear/Crisis"]
    for i, state in enumerate(calm_sorted):
        labels[state] = calm_names[i] if i < len(calm_names) else f"Calm-{i}"
    for i, state in enumerate(stressed_sorted):
        labels[state] = stressed_names[i] if i < len(stressed_names) else f"Stressed-{i}"

    print("State stats (return, vol):", state_stats)
    print("Labels:", labels)

    regime_series = pd.Series(
        [labels[s] for s in hidden_states],
        index=features.index,
        name="regime"
    )
    return regime_series, labels

REGIME_COLORS = {
    "Bull": "#2ca02c",
    "Neutral": "#7f7f7f",
    "High-Vol": "#ff7f0e",
    "Bear/Crisis": "#d62728",
}


def plot_regimes(price_series, regime_series, save_path=None):
    fig, ax = plt.subplots(figsize=(14, 6))

    # align both series to the same dates (regime_series is shorter due to
    # rolling-window warmup, so plot price only where we have a regime label)
    common_idx = price_series.index.intersection(regime_series.index)
    price_aligned = price_series.loc[common_idx]
    regime_aligned = regime_series.loc[common_idx]

    # shade the background by regime: find contiguous blocks of the same
    # regime and draw one axvspan per block, so adjacent same-regime days
    # merge into a single shaded region instead of one span per day
    regime_change = regime_aligned.ne(regime_aligned.shift()).cumsum()
    for _, block in regime_aligned.groupby(regime_change):
        label = block.iloc[0]
        start, end = block.index[0], block.index[-1]
        ax.axvspan(start, end, color=REGIME_COLORS.get(label, "#cccccc"),
                   alpha=0.25, lw=0)

    ax.plot(price_aligned.index, price_aligned.values, color="black",
            linewidth=1, label="Equal-weight portfolio price")

    # build legend manually since axvspan doesn't auto-register labels
    legend_handles = [mpatches.Patch(color=c, alpha=0.25, label=l)
                       for l, c in REGIME_COLORS.items()
                       if l in regime_aligned.unique()]
    legend_handles.append(plt.Line2D([0], [0], color="black", lw=1,
                                       label="Equal-weight portfolio price"))
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)

    ax.set_title("Detected Market Regimes vs Portfolio Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Price")
    plt.tight_layout()

    if save_path is None:
        save_path = Path(__file__).resolve().parent.parent / "data" / "regime_plot.png"
    else:
        save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Plot saved to {save_path}")