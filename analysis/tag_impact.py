"""
Tag-impact analysis: treat each user day-tag (e.g. "sleep-cube", "sauna",
"fasting") as an intervention/condition and measure its effect on health
metrics -- both same-day and next-day.

Two comparisons per (tag, metric, horizon):
  1. NAIVE: Mann-Whitney U of tagged vs untagged days + rank-biserial effect
     size. Fast but confounded (tagged days may differ systematically).
  2. VECTOR-MATCHED (whitepaper Ch. 8.3.2, N-of-1 propensity-style): for every
     tagged day, find its nearest UNTAGGED day by holistic Day-Vector distance
     computed EXCLUDING the outcome metric, then paired Wilcoxon on the matched
     outcome differences. This is the honest answer to "does the cube actually
     improve my sleep, controlling for how similar the days were otherwise?".

FDR across the whole (tag x metric x horizon) family. Emits a `tag_impact`
insight per surviving effect, reporting whether it survives matching.
"""
import logging
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon, rankdata
from statsmodels.stats.multitest import multipletests


def _rank_biserial(a, b):
    """Effect size for Mann-Whitney: 2*AUC-1 in [-1, 1]."""
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return np.nan
    all_v = np.concatenate([a, b])
    ranks = rankdata(all_v)
    ra = ranks[:na].sum()
    ua = ra - na * (na + 1) / 2.0
    return 2.0 * (ua / (na * nb)) - 1.0


def _build_vectors(daily_df, min_obs, min_dims):
    df = daily_df.copy()
    df.index = pd.to_datetime(df.index)
    cov = df.notna().sum()
    feats = [c for c in df.columns if cov.get(c, 0) >= min_obs]
    if len(feats) < 3:
        return None, None
    Z = (df[feats] - df[feats].mean()) / df[feats].std(ddof=0).replace(0, np.nan)
    Z = Z[Z.notna().sum(axis=1) >= min_dims]
    return Z, feats


def run_tag_impact_analysis(daily_df, tags_df, parameters):
    """
    daily_df: index=date, columns=metrics.
    tags_df:  long frame with columns [day, tag_name] (from analysis_day_tags).
    Returns a LIST of insight dicts.
    """
    min_tagged = parameters.get("min_tagged_days", 5)
    min_untagged = parameters.get("min_untagged_days", 5)
    horizons = parameters.get("horizons", [0, 1])  # 0 = same day, 1 = next day
    max_p = parameters.get("max_p", 0.05)
    max_q = parameters.get("max_q", 0.10)
    top_k = parameters.get("top_k", 15)
    min_effect = parameters.get("min_abs_effect", 0.3)  # |rank-biserial|
    min_obs = parameters.get("min_obs_per_metric", 60)
    min_dims = parameters.get("min_dims_per_day", 5)

    if daily_df is None or daily_df.empty or tags_df is None or len(tags_df) == 0:
        return None

    df = daily_df.copy()
    df.index = pd.to_datetime(df.index)
    tags_df = tags_df.copy()
    tags_df["day"] = pd.to_datetime(tags_df["day"])

    metrics = [c for c in df.columns if df[c].notna().sum() >= min_obs]
    if not metrics:
        return None

    # Vector space for matched comparison (built once).
    Z, feats = _build_vectors(df, min_obs, min_dims)

    candidates = []  # dict per (tag, metric, horizon)
    for tag_name, g in tags_df.groupby("tag_name"):
        tagged_days = pd.DatetimeIndex(sorted(set(g["day"])))
        if len(tagged_days) < min_tagged:
            continue
        for metric in metrics:
            for h in horizons:
                # outcome measured h days AFTER the tagged day
                out = df[metric]
                out_shifted = out.copy()
                out_shifted.index = out_shifted.index - pd.Timedelta(days=h)
                # value attributed to day d = metric at d+h
                tagged_vals = out_shifted.reindex(tagged_days).dropna()
                untagged_index = df.index.difference(tagged_days)
                untagged_vals = out_shifted.reindex(untagged_index).dropna()
                if len(tagged_vals) < min_tagged or len(untagged_vals) < min_untagged:
                    continue
                try:
                    _, p = mannwhitneyu(tagged_vals.values, untagged_vals.values, alternative="two-sided")
                except Exception:
                    continue
                eff = _rank_biserial(tagged_vals.values, untagged_vals.values)
                if np.isnan(p) or np.isnan(eff):
                    continue
                candidates.append({
                    "tag": tag_name, "metric": metric, "horizon": h,
                    "n_tagged": int(len(tagged_vals)), "n_untagged": int(len(untagged_vals)),
                    "mean_tagged": float(tagged_vals.mean()),
                    "mean_untagged": float(untagged_vals.mean()),
                    "effect": float(eff), "p": float(p),
                    "tagged_days": tagged_days,
                })

    if not candidates:
        return None

    # FDR across the whole family.
    q = multipletests([c["p"] for c in candidates], alpha=max_q, method="fdr_bh")[1]
    for c, qv in zip(candidates, q):
        c["q"] = float(qv)

    survivors = [c for c in candidates
                 if abs(c["effect"]) >= min_effect and c["p"] < max_p and c["q"] < max_q]
    if not survivors:
        return None
    survivors.sort(key=lambda c: abs(c["effect"]), reverse=True)
    survivors = survivors[:top_k]

    insights = []
    for c in survivors:
        metric, tag, h = c["metric"], c["tag"], c["horizon"]
        mh = metric.replace("_", " ")
        horizon_txt = "on tagged days" if h == 0 else f"{h} day(s) after"
        direction = "higher" if c["mean_tagged"] > c["mean_untagged"] else "lower"
        delta = c["mean_tagged"] - c["mean_untagged"]

        # ---- vector-matched (PSM-lite) confirmation ----
        matched = None
        if Z is not None and metric in df.columns:
            try:
                feats_ex = [f for f in feats if f != metric]
                if len(feats_ex) >= 3:
                    Zex = Z[feats_ex].fillna(0.0)
                    tagged_in = [d for d in c["tagged_days"] if d in Zex.index]
                    untagged_pool = Zex.index.difference(pd.DatetimeIndex(c["tagged_days"]))
                    Zpool = Zex.reindex(untagged_pool).dropna(how="all")
                    diffs = []
                    used_match = set()
                    for d in tagged_in:
                        td = d + pd.Timedelta(days=h)
                        if td not in df.index or pd.isna(df.at[td, metric]):
                            continue
                        tv = df.at[td, metric]
                        dist = np.linalg.norm(Zpool.values - Zex.loc[d].values, axis=1)
                        order = np.argsort(dist)
                        picked = None
                        for oi in order:
                            cand_day = Zpool.index[oi]
                            if cand_day in used_match:
                                continue
                            cd = cand_day + pd.Timedelta(days=h)
                            if cd in df.index and not pd.isna(df.at[cd, metric]):
                                picked = cd; used_match.add(cand_day); break
                        if picked is not None:
                            diffs.append(tv - df.at[picked, metric])
                    if len(diffs) >= max(3, min_tagged - 1):
                        try:
                            _, mp = wilcoxon(diffs)
                            matched = {
                                "n_pairs": len(diffs),
                                "mean_diff": round(float(np.mean(diffs)), 3),
                                "wilcoxon_p": round(float(mp), 5),
                                "survives_matching": bool(mp < max_p and np.sign(np.mean(diffs)) == np.sign(delta)),
                            }
                        except Exception:
                            matched = None
            except Exception as e:
                logging.error("tag_impact matching failed (%s/%s): %s", tag, metric, e)

        match_txt = ""
        if matched is not None:
            if matched["survives_matching"]:
                match_txt = (f" This holds up against matched comparison "
                             f"({matched['n_pairs']} similar-day pairs, p={matched['wilcoxon_p']}).")
            else:
                match_txt = (f" However, it does NOT survive matched comparison "
                             f"({matched['n_pairs']} pairs, p={matched['wilcoxon_p']}), so it may reflect "
                             f"which days you tagged rather than the tag's effect.")

        insights.append({
            "type": "tag_impact",
            "title": f"'{tag}' and Your {mh.title()}",
            "summary": (
                f"Your {mh} tends to be {direction} {horizon_txt} '{tag}' days "
                f"(by ~{abs(delta):.2f}; effect size {c['effect']:+.2f}, "
                f"n={c['n_tagged']} tagged vs {c['n_untagged']} untagged)."
                + match_txt
            ),
            "evidence": {
                "tag": tag,
                "metric": metric,
                "horizon_days": h,
                "rank_biserial_effect": round(c["effect"], 3),
                "p_value": round(c["p"], 5),
                "q_value_fdr": round(c["q"], 5),
                "mean_tagged": round(c["mean_tagged"], 3),
                "mean_untagged": round(c["mean_untagged"], 3),
                "n_tagged": c["n_tagged"],
                "n_untagged": c["n_untagged"],
                "matched_comparison": matched,
            },
        })

    logging.info(
        "tag_impact '%s': %d effects (of %d tested)",
        parameters.get("name", "tag_impact"), len(insights), len(candidates),
    )
    return insights
