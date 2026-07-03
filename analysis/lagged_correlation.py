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
    Directed multi-lag correlation discovery: X(t) -> Y(t + L) for L in lag_days.

    Captures carry-over physiology that same-day correlation is blind to, at
    MULTIPLE horizons (e.g. HRV -> stress next day, but also training -> deep
    sleep a week later via super-compensation). For every ORDERED pair (x, y)
    and every lag L, correlate x shifted forward by L days against y on a
    continuous daily grid (so the shift is a true calendar-day lag).

    Statistical honesty:
      - ONE Benjamini-Hochberg FDR family across ALL (pair x lag) tests, so
        scanning several lags does not inflate false positives.
      - Gate: |r| >= min_abs_r AND p < max_p AND q < max_q.
      - Per directed pair, keep only the BEST lag (largest |r|) to avoid near-
        duplicate insights for adjacent horizons; report top_k overall.

    `lag_days` accepts an int or a list (default [1, 2, 3, 7]).
    Returns a LIST of insight dicts.
    """
    method = parameters.get("method", "spearman")
    lag_param = parameters.get("lag_days", [1, 2, 3, 7])
    lags = [int(lag_param)] if isinstance(lag_param, (int, float)) else [int(l) for l in lag_param]
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
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_idx)
    cols = list(df.columns)

    candidates = []  # (x, y, lag, n, r, p)
    for lag in lags:
        shifted = {c: df[c].shift(lag) for c in cols}
        for x in cols:
            x_lag = shifted[x]
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
                candidates.append((x, y, lag, n, r, p))

    if not candidates:
        return None

    # ONE FDR family across every (pair x lag) test.
    p_values = [c[5] for c in candidates]
    _, q_values, _, _ = multipletests(p_values, alpha=max_q, method="fdr_bh")

    survivors = []
    for (x, y, lag, n, r, p), q in zip(candidates, q_values):
        if abs(r) >= min_abs_r and p < max_p and q < max_q:
            survivors.append((x, y, lag, n, r, p, q))

    if not survivors:
        return None

    # Keep the strongest lag per directed (x, y) pair.
    best_per_pair = {}
    for s in survivors:
        key = (s[0], s[1])
        if key not in best_per_pair or abs(s[4]) > abs(best_per_pair[key][4]):
            best_per_pair[key] = s
    survivors = sorted(best_per_pair.values(), key=lambda t: abs(t[4]), reverse=True)[:top_k]

    insights = []
    for x, y, lag, n, r, p, q in survivors:
        direction = "higher" if r > 0 else "lower"
        when = "the next day" if lag == 1 else f"{lag} days later"
        when_title = "Tomorrow" if lag == 1 else f"{lag} Days Later"
        insights.append({
            "type": "lagged_correlation",
            "title": f"{x.replace('_', ' ').title()} Today Precedes {y.replace('_', ' ').title()} {when_title}",
            "summary": (
                f"Days with higher {x.replace('_', ' ')} tend to be followed by "
                f"{direction} {y.replace('_', ' ')} {when} "
                f"(r={r:.2f}, lag={lag}d, n={n})."
            ),
            "evidence": {
                "predictor_metric": x,
                "outcome_metric": y,
                "lag_days": lag,
                "method": method,
                "correlation_coefficient": round(float(r), 3),
                "p_value": round(float(p), 5),
                "q_value_fdr": round(float(q), 5),
                "data_points_used": int(n),
            },
        })

    logging.info(
        "lagged_correlation '%s': %d links (of %d tested across lags %s)",
        parameters.get("name", "lag_discovery"), len(insights), len(candidates), lags,
    )
    return insights
