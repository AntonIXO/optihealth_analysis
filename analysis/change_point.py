"""
Change-point / regime-shift detection (classical, dependency-light).

Answers "WHEN did my body's baseline shift?" -- e.g. resting HR stepped up
~4 bpm in late January, or deep sleep collapsed in October. Single metrics
drift and jump in level; correlation/forecasting are blind to a discrete
baseline shift. This module dates those shifts and quantifies them.

Method: Taylor's CUSUM + bootstrap.
  1. CUSUM of deviations from the mean; the extreme of the cumulative sum marks
     the most likely change-point.
  2. Bootstrap significance: permute the series many times; the confidence that
     a change exists is the fraction of permutations whose CUSUM range is
     smaller than observed. No distributional assumption.
  3. Binary segmentation: recurse into the two sub-segments to find multiple
     change-points, each independently bootstrap-gated.
Gates: bootstrap confidence >= min_confidence, both segments >= min_segment_len,
|level shift| >= min_shift_sd, and FDR (Benjamini-Hochberg) across metrics on
the top-level change of each metric. Reports the most recent / largest shifts.

This is the classical statistical counterpart to the learned optiHealth-EiV
temporal encoder: it needs no training and yields dated, interpretable regime
boundaries that a downstream agent (or EiV training) can consume.
"""
import logging
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

_RNG = np.random.default_rng(42)


def _cusum_confidence(x, n_boot):
    """Return (change_idx, confidence, cusum_range) for one series via CUSUM+bootstrap."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    mu = x.mean()
    S = np.concatenate([[0.0], np.cumsum(x - mu)])
    s_range = S.max() - S.min()
    if s_range == 0:
        return None, 0.0, 0.0
    smaller = 0
    for _ in range(n_boot):
        xp = _RNG.permutation(x)
        Sp = np.cumsum(xp - mu)
        if (Sp.max() - Sp.min()) < s_range:
            smaller += 1
    confidence = smaller / n_boot
    change_idx = int(np.argmax(np.abs(S)))  # extreme cumulative deviation
    return change_idx, confidence, s_range


def _segment(x, dates, min_seg, min_conf, n_boot, depth, max_depth, out):
    n = len(x)
    if n < 2 * min_seg or depth > max_depth:
        return
    idx, conf, _ = _cusum_confidence(x, n_boot)
    if idx is None or idx < min_seg or idx > n - min_seg:
        return
    if conf < min_conf:
        return
    left, right = x[:idx], x[idx:]
    sd = np.asarray(x, float).std() or 1.0
    out.append({
        "date": dates[idx],
        "confidence": conf,
        "mean_before": float(left.mean()),
        "mean_after": float(right.mean()),
        "shift_sd": float((right.mean() - left.mean()) / sd),
        "n_before": int(len(left)),
        "n_after": int(len(right)),
        "depth": depth,
    })
    _segment(x[:idx], dates[:idx], min_seg, min_conf, n_boot, depth + 1, max_depth, out)
    _segment(x[idx:], dates[idx:], min_seg, min_conf, n_boot, depth + 1, max_depth, out)


def run_change_point_analysis(daily_df, parameters):
    """Returns a LIST of `regime_change` insight dicts (or None)."""
    min_obs = parameters.get("min_obs_per_metric", 80)
    min_seg = parameters.get("min_segment_len", 21)
    min_conf = parameters.get("min_confidence", 0.95)
    min_shift_sd = parameters.get("min_shift_sd", 0.6)
    n_boot = parameters.get("n_bootstrap", 1000)
    max_depth = parameters.get("max_depth", 3)
    top_k = parameters.get("top_k", 12)
    max_q = parameters.get("max_q", 0.10)
    metrics = parameters.get("metrics")  # None => all well-covered

    if daily_df is None or daily_df.empty:
        return None

    df = daily_df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    cov = df.notna().sum()
    cols = metrics if metrics else [c for c in df.columns if cov.get(c, 0) >= min_obs]
    if not cols:
        return None

    detected = []      # all change-points across metrics
    top_pvals = []     # one representative p per metric (strongest change) for FDR
    per_metric_top = {}
    for m in cols:
        s = df[m].dropna()
        if len(s) < 2 * min_seg:
            continue
        out = []
        _segment(s.values, list(s.index), min_seg, min_conf, n_boot, 0, max_depth, out)
        # keep only shifts large enough to matter
        out = [c for c in out if abs(c["shift_sd"]) >= min_shift_sd]
        if not out:
            continue
        for c in out:
            c["metric"] = m
        detected.extend(out)
        strongest = max(out, key=lambda c: c["confidence"])
        per_metric_top[m] = strongest
        top_pvals.append(max(1e-4, 1.0 - strongest["confidence"]))

    if not detected:
        return None

    # FDR across metrics (on each metric's strongest change); annotate every
    # change of a metric with that metric's q-value.
    metric_names = list(per_metric_top.keys())
    q_by_metric = {}
    if top_pvals:
        q_vals = multipletests(top_pvals, alpha=max_q, method="fdr_bh")[1]
        q_by_metric = {mn: float(q) for mn, q in zip(metric_names, q_vals)}

    # keep changes whose metric passes FDR
    survivors = [c for c in detected if q_by_metric.get(c["metric"], 1.0) < max_q]
    if not survivors:
        return None

    # rank by absolute shift, cap
    survivors.sort(key=lambda c: abs(c["shift_sd"]), reverse=True)
    survivors = survivors[:top_k]

    insights = []
    for c in survivors:
        m = c["metric"]
        mh = m.replace("_", " ")
        direction = "increased" if c["shift_sd"] > 0 else "decreased"
        insights.append({
            "type": "regime_change",
            "title": f"Your {mh.title()} Shifted Around {c['date'].date()}",
            "summary": (
                f"Your {mh} {direction} around {c['date'].date()}, from about "
                f"{c['mean_before']:.1f} to {c['mean_after']:.1f} "
                f"({c['shift_sd']:+.2f} SD; {c['n_before']} days before vs "
                f"{c['n_after']} after, confidence {c['confidence']:.0%}). "
                f"This was a sustained baseline shift, not a one-off day."
            ),
            "evidence": {
                "metric": m,
                "change_date": str(c["date"].date()),
                "mean_before": round(c["mean_before"], 3),
                "mean_after": round(c["mean_after"], 3),
                "shift_sd": round(c["shift_sd"], 3),
                "confidence": round(c["confidence"], 4),
                "q_value_fdr": round(q_by_metric.get(m, 1.0), 5),
                "n_before": c["n_before"],
                "n_after": c["n_after"],
            },
        })

    logging.info(
        "change_point '%s': %d regime shifts (of %d detected across %d metrics)",
        parameters.get("name", "change_point"), len(insights), len(detected), len(cols),
    )
    return insights
