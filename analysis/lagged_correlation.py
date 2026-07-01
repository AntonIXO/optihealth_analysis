import logging
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests

_SLEEP_PARTS = {
    "sleep_duration_light",
    "sleep_duration_deep",
    "sleep_duration_rem",
    "sleep_duration_awake",
}
_DEFAULT_EXCLUDE = [
    ["workout_distance", "workout_duration"],
]


def _is_trivial(a, b, exclude_pairs):
    s = {a, b}
    if "sleep_duration_total" in s and (a in _SLEEP_PARTS or b in _SLEEP_PARTS):
        return True
    for pair in exclude_pairs:
        if set(pair) == s:
            return True
    return False


def run_lagged_correlation_discovery(daily_df, parameters):
    """
    Directed next-day (lag) correlation discovery: X(t) -> Y(t + lag_days).

    Captures overnight / carry-over physiology that same-day correlation is blind
    to (e.g. HRV today -> lower stress tomorrow). For every ORDERED pair (x, y),
    correlate x shifted forward by lag_days against y, on a continuous daily grid
    so the shift is a true calendar day. Same FDR + threshold gating as the
    same-day discovery. Returns a LIST of insight dicts (dir-aware), capped top_k.
    """
    method = parameters.get("method", "spearman")
    lag_days = int(parameters.get("lag_days", 1))
    min_points = parameters.get("min_data_points", 25)
    min_abs_r = parameters.get("min_abs_r", 0.35)
    max_p = parameters.get("max_p", 0.05)
    max_q = parameters.get("max_q", 0.10)
    top_k = parameters.get("top_k", 15)
    exclude_pairs = parameters.get("exclude_pairs", _DEFAULT_EXCLUDE)

    if daily_df is None or daily_df.empty or daily_df.shape[1] < 2:
        return None

    corr_fn = spearmanr if method == "spearman" else pearsonr

    df = daily_df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # Continuous daily grid so shift(lag) is a real calendar-day lag, not a
    # "next observed row" lag.
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_idx)
    cols = list(df.columns)

    candidates = []  # (x, y, n, r, p)
    for x in cols:
        x_lag = df[x].shift(lag_days)
        for y in cols:
            if x == y or _is_trivial(x, y, exclude_pairs):
                continue
            sub = pd.concat([x_lag.rename("x"), df[y].rename("y")], axis=1).dropna()
            n = len(sub)
            if n < min_points:
                continue
            r, p = corr_fn(sub["x"], sub["y"])
            if r is None or np.isnan(r):
                continue
            candidates.append((x, y, n, r, p))

    if not candidates:
        return None

    p_values = [c[4] for c in candidates]
    _, q_values, _, _ = multipletests(p_values, alpha=max_q, method="fdr_bh")

    survivors = []
    for (x, y, n, r, p), q in zip(candidates, q_values):
        if abs(r) >= min_abs_r and p < max_p and q < max_q:
            survivors.append((x, y, n, r, p, q))

    if not survivors:
        return None

    survivors.sort(key=lambda t: abs(t[3]), reverse=True)
    survivors = survivors[:top_k]

    day_word = "the next day" if lag_days == 1 else f"{lag_days} days later"
    insights = []
    for x, y, n, r, p, q in survivors:
        direction = "higher" if r > 0 else "lower"
        insights.append({
            "type": "lagged_correlation",
            "title": f"{x.replace('_', ' ').title()} Today Precedes {y.replace('_', ' ').title()} {('Tomorrow' if lag_days == 1 else 'Later')}",
            "summary": (
                f"Days with higher {x.replace('_', ' ')} tend to be followed by "
                f"{direction} {y.replace('_', ' ')} {day_word} "
                f"(r={r:.2f}, lag={lag_days}d, n={n})."
            ),
            "evidence": {
                "predictor_metric": x,
                "outcome_metric": y,
                "lag_days": lag_days,
                "method": method,
                "correlation_coefficient": round(float(r), 3),
                "p_value": round(float(p), 5),
                "q_value_fdr": round(float(q), 5),
                "data_points_used": int(n),
            },
        })

    logging.info(
        "lagged_correlation '%s': %d significant directed links (of %d tested, lag=%dd)",
        parameters.get("name", "lag_discovery"), len(insights), len(candidates), lag_days,
    )
    return insights
