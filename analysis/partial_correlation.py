import logging
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


def _residualize(y, x):
    """Return residuals of y regressed on x (with intercept). y, x are 1D arrays."""
    X = np.c_[np.ones(len(x)), x]
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def _partial_spearman(df, a, b, control):
    """
    Rank-based partial correlation of a and b controlling for `control`:
    rank-transform all three, residualize a and b on control, correlate the
    residuals. Returns (partial_r, partial_p, n).
    """
    sub = df[[a, b, control]].dropna()
    n = len(sub)
    if n < 5:
        return None, None, n
    R = sub.rank()
    ra = _residualize(R[a].values, R[control].values)
    rb = _residualize(R[b].values, R[control].values)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None, None, n
    r, p = pearsonr(ra, rb)
    return r, p, n


def run_partial_correlation_analysis(daily_df, parameters):
    """
    For a curated set of (a, b, control) triples, report whether the raw a~b
    link SURVIVES or COLLAPSES once an obvious confound is held constant.

    This separates genuinely independent associations from confound-driven ones,
    which is exactly the "is this correlation real or spurious?" question a naive
    correlation scan cannot answer. Returns a LIST of insight dicts.
    """
    triples = parameters.get('triples', [])
    min_points = parameters.get('min_data_points', 25)
    min_abs_r = parameters.get('min_abs_r', 0.30)
    max_raw_p = parameters.get('max_raw_p', 0.05)
    top_k = parameters.get('top_k', 15)

    if daily_df is None or daily_df.empty or not triples:
        return None

    results = []
    for t in triples:
        a, b, control = t.get('a'), t.get('b'), t.get('control')
        if not all([a, b, control]):
            continue
        if not all(m in daily_df.columns for m in (a, b, control)):
            continue

        raw_sub = daily_df[[a, b]].dropna()
        if len(raw_sub) < min_points:
            continue
        raw_r, raw_p = spearmanr(raw_sub[a], raw_sub[b])
        if raw_r is None or np.isnan(raw_r):
            continue
        # Only bother reporting links that look real to begin with.
        if abs(raw_r) < min_abs_r or raw_p >= max_raw_p:
            continue

        pr, pp, n = _partial_spearman(daily_df, a, b, control)
        if pr is None or np.isnan(pr):
            continue

        survives = abs(pr) >= min_abs_r and (pp is not None and pp < max_raw_p)
        results.append((a, b, control, n, raw_r, raw_p, pr, pp, survives))

    if not results:
        return None

    # Rank by how much the control changed things (|raw| first, informative either way)
    results.sort(key=lambda x: abs(x[4]), reverse=True)
    results = results[:top_k]

    insights = []
    for a, b, control, n, raw_r, raw_p, pr, pp, survives in results:
        a_h, b_h, c_h = (m.replace('_', ' ').title() for m in (a, b, control))
        if survives:
            verdict = (
                f"The link between {a_h} and {b_h} holds up even after controlling "
                f"for {c_h} (raw r={raw_r:.2f} -> partial r={pr:.2f}), so it is likely "
                f"a genuine relationship rather than a side effect of {c_h}."
            )
            title = f"{a_h} & {b_h}: A Robust Link"
        else:
            verdict = (
                f"The apparent link between {a_h} and {b_h} (raw r={raw_r:.2f}) largely "
                f"disappears once {c_h} is held constant (partial r={pr:.2f}); it is "
                f"likely driven by {c_h} rather than a direct relationship."
            )
            title = f"{a_h} & {b_h}: Explained by {c_h}"

        insights.append({
            "type": "partial_correlation",
            "title": title,
            "summary": verdict,
            "evidence": {
                "metric_a": a,
                "metric_b": b,
                "controlled_for": control,
                "raw_correlation": round(float(raw_r), 3),
                "raw_p_value": round(float(raw_p), 5),
                "partial_correlation": round(float(pr), 3),
                "partial_p_value": (round(float(pp), 5) if pp is not None else None),
                "survives_control": bool(survives),
                "data_points_used": int(n),
            },
        })

    logging.info(
        "partial_correlation '%s': %d verdicts (of %d triples)",
        parameters.get('name', 'partial'), len(insights), len(triples),
    )
    return insights
