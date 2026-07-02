import logging
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def run_weekly_rhythm_analysis(daily_df, parameters):
    """
    Cyclical-time analysis (whitepaper Ch. 13): does a metric vary systematically
    by day-of-week / weekend vs weekday?

    Linear time treats every day the same. But behaviour is cyclical: weekends,
    work rhythms, and weekly routines leave a fingerprint. For each metric with
    enough coverage we run:
      - Kruskal-Wallis across the 7 weekdays (is there ANY weekday structure?)
      - Mann-Whitney weekend vs weekday (a concrete, interpretable contrast)
    FDR-corrected across metrics. Emits an insight per metric that passes,
    naming the highest/lowest weekday and the weekend shift in SD units.

    Returns a LIST of insight dicts (may be empty -> None, which is correct and
    honest when the user simply has no weekly structure).
    """
    min_points = parameters.get("min_data_points", 60)
    min_per_group = parameters.get("min_per_weekday", 5)
    max_p = parameters.get("max_p", 0.05)
    max_q = parameters.get("max_q", 0.10)
    top_k = parameters.get("top_k", 10)
    metrics = parameters.get("metrics")  # None => all numeric columns

    if daily_df is None or daily_df.empty:
        return None

    df = daily_df.copy()
    df.index = pd.to_datetime(df.index)
    cols = metrics if metrics else list(df.columns)

    rows = []  # (metric, n, p_kruskal, best_day, worst_day, weekend_shift_sd, p_weekend)
    for m in cols:
        if m not in df.columns:
            continue
        s = df[m].dropna()
        if len(s) < min_points:
            continue
        wd = s.index.dayofweek
        groups = [s[wd == i].values for i in range(7) if (wd == i).sum() >= min_per_group]
        if len(groups) < 5:  # need most weekdays represented
            continue
        try:
            _, p_kw = kruskal(*groups)
        except Exception:
            continue
        if p_kw is None or np.isnan(p_kw):
            continue

        means = {i: s[wd == i].mean() for i in range(7) if (wd == i).sum() >= min_per_group}
        best_day = _WEEKDAYS[max(means, key=means.get)]
        worst_day = _WEEKDAYS[min(means, key=means.get)]

        we, wk = s[wd >= 5], s[wd < 5]
        sd = s.std() or 1.0
        shift = np.nan
        p_we = np.nan
        if len(we) >= 8 and len(wk) >= 8:
            try:
                _, p_we = mannwhitneyu(we, wk)
                shift = (we.mean() - wk.mean()) / sd
            except Exception:
                pass
        rows.append((m, len(s), p_kw, best_day, worst_day, shift, p_we))

    if not rows:
        return None

    p_all = [r[2] for r in rows]
    _, q_all, _, _ = multipletests(p_all, alpha=max_q, method="fdr_bh")

    survivors = [
        (r, q) for r, q in zip(rows, q_all)
        if r[2] < max_p and q < max_q
    ]
    if not survivors:
        return None

    survivors.sort(key=lambda rq: rq[0][2])  # by p ascending
    survivors = survivors[:top_k]

    insights = []
    for (m, n, p_kw, best_day, worst_day, shift, p_we), q in survivors:
        mh = m.replace("_", " ")
        we_txt = ""
        if not np.isnan(shift):
            dir_word = "higher" if shift > 0 else "lower"
            we_txt = f" Your weekend {mh} runs {abs(shift):.2f} SD {dir_word} than weekdays."
        insights.append({
            "type": "weekly_rhythm",
            "title": f"Your {mh.title()} Follows a Weekly Rhythm",
            "summary": (
                f"Your {mh} varies by day of week (highest on {best_day}, lowest on "
                f"{worst_day}; Kruskal-Wallis p={p_kw:.3f}, n={n} days).{we_txt}"
            ),
            "evidence": {
                "metric": m,
                "p_kruskal": round(float(p_kw), 5),
                "q_value_fdr": round(float(q), 5),
                "highest_weekday": best_day,
                "lowest_weekday": worst_day,
                "weekend_shift_sd": (round(float(shift), 3) if not np.isnan(shift) else None),
                "p_weekend_vs_weekday": (round(float(p_we), 5) if not np.isnan(p_we) else None),
                "data_points_used": int(n),
            },
        })

    logging.info(
        "weekly_rhythm '%s': %d metrics with weekly structure (of %d tested)",
        parameters.get("name", "weekly"), len(insights), len(rows),
    )
    return insights
