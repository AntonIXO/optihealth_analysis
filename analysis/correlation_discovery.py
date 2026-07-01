import itertools
import logging
import numpy as np
from scipy.stats import spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests

# Definitional / tautological pairs that should never be reported as "insights".
# sleep_duration_total = light + deep + rem + awake, so those correlate trivially.
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


def run_correlation_discovery(daily_df, parameters):
    """
    Unsupervised same-day correlation discovery across ALL metric pairs.

    Instead of testing a hand-picked list, scan every pair with enough overlapping
    days, then keep only statistically defensible links:
      - |r| >= min_abs_r
      - p  <  max_p
      - Benjamini-Hochberg FDR q < max_q  (controls false positives across the
        whole family of tested pairs)
    Returns a LIST of insight dicts (one per surviving pair), ranked by |r| and
    capped at top_k. main_worker already handles list results (extend).
    """
    method = parameters.get("method", "spearman")
    min_points = parameters.get("min_data_points", 25)
    min_abs_r = parameters.get("min_abs_r", 0.35)
    max_p = parameters.get("max_p", 0.05)
    max_q = parameters.get("max_q", 0.10)
    top_k = parameters.get("top_k", 15)
    exclude_pairs = parameters.get("exclude_pairs", _DEFAULT_EXCLUDE)

    if daily_df is None or daily_df.empty or daily_df.shape[1] < 2:
        return None

    corr_fn = spearmanr if method == "spearman" else pearsonr
    cols = list(daily_df.columns)

    candidates = []  # (a, b, n, r, p)
    for a, b in itertools.combinations(cols, 2):
        if a == b or _is_trivial(a, b, exclude_pairs):
            continue
        sub = daily_df[[a, b]].dropna()
        n = len(sub)
        if n < min_points:
            continue
        r, p = corr_fn(sub[a], sub[b])
        if r is None or np.isnan(r):
            continue
        candidates.append((a, b, n, r, p))

    if not candidates:
        return None

    # Family-wide FDR correction over every pair we actually tested.
    p_values = [c[4] for c in candidates]
    _, q_values, _, _ = multipletests(p_values, alpha=max_q, method="fdr_bh")

    survivors = []
    for (a, b, n, r, p), q in zip(candidates, q_values):
        if abs(r) >= min_abs_r and p < max_p and q < max_q:
            survivors.append((a, b, n, r, p, q))

    if not survivors:
        return None

    survivors.sort(key=lambda x: abs(x[3]), reverse=True)
    survivors = survivors[:top_k]

    insights = []
    for a, b, n, r, p, q in survivors:
        relationship = "positively" if r > 0 else "negatively"
        strength = "strongly" if abs(r) > 0.6 else "moderately"
        insights.append({
            "type": "correlation",
            "title": f"Link Between {a.replace('_', ' ').title()} and {b.replace('_', ' ').title()}",
            "summary": (
                f"Your {a.replace('_', ' ').title()} is {strength} {relationship} "
                f"correlated with your {b.replace('_', ' ').title()} "
                f"(r={r:.2f}, n={n} days)."
            ),
            "evidence": {
                "metric_a": a,
                "metric_b": b,
                "method": method,
                "correlation_coefficient": round(float(r), 3),
                "p_value": round(float(p), 5),
                "q_value_fdr": round(float(q), 5),
                "data_points_used": int(n),
            },
        })

    logging.info(
        "correlation_discovery '%s': %d significant pairs (of %d tested)",
        parameters.get("name", "discovery"), len(insights), len(candidates),
    )
    return insights
