"""
The "Everything is a Vector" analysis module (classical implementation of the
whitepaper's Ch. 8 paradigm).

Each day is turned into a single point in a high-dimensional space: z-scored
health metrics plus cyclical day-of-week encoding. Geometric relationships in
that space then power three insight families that per-metric analysis cannot
produce:

  1. vector_anomaly  -- robust multi-variate outlier days (Mahalanobis distance
     with Ledoit-Wolf shrinkage covariance). Flags days that are holistically
     unusual even when no single metric is out of range.
  2. day_type        -- KMeans archetypes on the day vector; k chosen by
     silhouette, gated on separation (and outcome ANOVA when an outcome is set).
  3. similar_days    -- for the most recent complete day, its nearest historical
     neighbours ("when have I felt like this before?").

This is the classical / statistical counterpart to the learned Temporal Encoder
(optiHealth-EiV); it needs no training and runs on whatever data exists.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import f_oneway

try:
    from sklearn.covariance import LedoitWolf
    _HAVE_LW = True
except Exception:  # pragma: no cover
    _HAVE_LW = False


def _build_day_vectors(daily_df, min_obs_per_metric, min_dims_per_day, add_cyclical):
    """Return (V_raw, Vz_imputed, feature_cols).

    V_raw: z-scored features with NaNs kept (for human-readable drivers).
    Vz_imputed: same, NaN->0 (=mean in z-space) for geometry; optional sin/cos DoW.
    """
    df = daily_df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    coverage = df.notna().sum()
    feats = [c for c in df.columns if coverage.get(c, 0) >= min_obs_per_metric]
    if len(feats) < 3:
        return None, None, None

    Z = (df[feats] - df[feats].mean()) / df[feats].std(ddof=0).replace(0, np.nan)
    filled = Z.notna().sum(axis=1)
    Z = Z[filled >= min_dims_per_day]
    if len(Z) < 20:
        return None, None, None

    Vi = Z.fillna(0.0)
    if add_cyclical:
        dow = Vi.index.dayofweek.values
        Vi = Vi.assign(
            _dow_sin=np.sin(2 * np.pi * dow / 7.0),
            _dow_cos=np.cos(2 * np.pi * dow / 7.0),
        )
    return Z, Vi, feats


def _drivers(zrow, k=3):
    s = zrow.dropna()
    if s.empty:
        return []
    top = s.abs().sort_values(ascending=False).head(k)
    return [(name, float(s[name])) for name in top.index]


def run_day_vector_analysis(daily_df, parameters):
    """Returns a LIST of insight dicts across the three vector families."""
    min_obs = parameters.get("min_obs_per_metric", 100)
    min_dims = parameters.get("min_dims_per_day", 6)
    add_cyclical = parameters.get("add_cyclical", True)
    anomaly_pct = parameters.get("anomaly_percentile", 97.5)
    max_anomalies = parameters.get("max_anomalies", 5)
    recent_days = parameters.get("recent_anomaly_window_days", 120)
    k_range = parameters.get("k_range", [2, 3, 4, 5, 6])
    min_silhouette = parameters.get("min_silhouette", 0.15)
    outcome_metric = parameters.get("outcome_metric")  # optional
    max_anova_p = parameters.get("max_anova_p", 0.05)
    n_similar = parameters.get("n_similar", 5)

    if daily_df is None or daily_df.empty:
        return None

    Z, Vi, feats = _build_day_vectors(daily_df, min_obs, min_dims, add_cyclical)
    if Vi is None:
        logging.info("day_vector '%s': not enough coverage to build vectors", parameters.get("name"))
        return None

    X = Vi.values
    insights = []

    # ---- 1. Multi-variate anomaly (Mahalanobis, shrinkage covariance) ----
    try:
        if _HAVE_LW:
            cov = LedoitWolf().fit(X).covariance_
        else:
            cov = np.cov(X, rowvar=False)
        inv = np.linalg.pinv(cov)
        mu = X.mean(axis=0)
        diff = X - mu
        md = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv, diff))
        thr = np.percentile(md, anomaly_pct)
        order = np.argsort(md)[::-1]
        cutoff = pd.Timestamp(Vi.index.max()) - pd.Timedelta(days=recent_days)
        shown = 0
        for idx in order:
            if md[idx] < thr:
                break
            day = Vi.index[idx]
            if day < cutoff:
                continue  # only surface reasonably recent anomalies
            drv = _drivers(Z.loc[day])
            drv_txt = ", ".join(f"{n.replace('_',' ')} {v:+.1f}SD" for n, v in drv)
            insights.append({
                "type": "vector_anomaly",
                "title": f"Holistically Unusual Day: {day.date()}",
                "summary": (
                    f"{day.date()} stands out as multi-variate unusual for you "
                    f"(Mahalanobis distance {md[idx]:.1f}). Main drivers: {drv_txt}."
                ),
                "evidence": {
                    "date": str(day.date()),
                    "mahalanobis_distance": round(float(md[idx]), 3),
                    "threshold": round(float(thr), 3),
                    "drivers": [{"metric": n, "z": round(v, 2)} for n, v in drv],
                },
            })
            shown += 1
            if shown >= max_anomalies:
                break
    except Exception as e:
        logging.error("day_vector anomaly failed for '%s': %s", parameters.get("name"), e)

    # ---- 2. Day-type clustering on the vector ----
    try:
        best = None  # (k, silhouette, labels)
        for k in k_range:
            if k >= len(X):
                continue
            km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
            if len(set(km.labels_)) < 2:
                continue
            sil = silhouette_score(X, km.labels_)
            if best is None or sil > best[1]:
                best = (k, sil, km.labels_)
        if best and best[1] >= min_silhouette:
            k, sil, labels = best
            clustered = Z.copy()
            clustered["_cluster"] = labels
            # profile each cluster by its most distinctive z-features
            profiles = {}
            for cid in range(k):
                mask = labels == cid
                prof = clustered.loc[mask, feats].mean().dropna()
                top = prof.abs().sort_values(ascending=False).head(3)
                desc = ", ".join(
                    f"{'high' if prof[n] > 0 else 'low'} {n.replace('_',' ')}"
                    for n in top.index
                )
                profiles[cid] = {
                    "size_percent": round(100.0 * mask.sum() / len(labels), 1),
                    "description": desc or "average",
                }
            # optional outcome separation gate -- only applied when the outcome
            # metric is well-covered across the clustered days (else the ANOVA is
            # underpowered and would wrongly suppress a real, descriptive day-type
            # structure). Day-types are descriptive first; outcome separation is a
            # bonus, not a prerequisite, when the outcome is sparse.
            outcome_ok = True
            outcome_evidence = {}
            if outcome_metric and outcome_metric in daily_df.columns:
                oc = daily_df[outcome_metric].reindex(Z.index)
                min_outcome_cov = parameters.get("min_outcome_coverage", 0.5)
                if oc.notna().mean() >= min_outcome_cov:
                    groups = [oc[labels == cid].dropna().values for cid in range(k)]
                    groups = [g for g in groups if len(g) >= 3]
                    if len(groups) >= 2:
                        try:
                            _, ap = f_oneway(*groups)
                            outcome_ok = ap is not None and ap < max_anova_p
                            outcome_evidence = {"outcome_metric": outcome_metric,
                                                "anova_p": round(float(ap), 5)}
                        except Exception:
                            outcome_ok = True
            if outcome_ok:
                insights.append({
                    "type": "day_type",
                    "title": f"You Have {k} Distinct 'Day Types'",
                    "summary": (
                        f"Clustering your holistic day vectors reveals {k} recurring "
                        f"archetypes (separation silhouette={sil:.2f}). "
                        + "; ".join(
                            f"Type {cid}: {p['description']} ({p['size_percent']}%)"
                            for cid, p in profiles.items()
                        ) + "."
                    ),
                    "evidence": {
                        "n_clusters": k,
                        "silhouette": round(float(sil), 3),
                        "profiles": profiles,
                        **outcome_evidence,
                    },
                })
    except Exception as e:
        logging.error("day_vector clustering failed for '%s': %s", parameters.get("name"), e)

    # ---- 3. Similar days to the most recent complete day ----
    try:
        anchor = Vi.index.max()
        av = Vi.loc[anchor].values
        d = np.linalg.norm(X - av, axis=1)
        sim_idx = np.argsort(d)
        neighbours = [Vi.index[i] for i in sim_idx if Vi.index[i] != anchor][:n_similar]
        if neighbours:
            drv = _drivers(Z.loc[anchor])
            drv_txt = ", ".join(f"{n.replace('_',' ')} {v:+.1f}SD" for n, v in drv)
            insights.append({
                "type": "similar_days",
                "title": f"Days Like {anchor.date()}",
                "summary": (
                    f"Your most recent profiled day ({anchor.date()}; notable: {drv_txt}) "
                    f"is holistically most similar to: "
                    + ", ".join(str(n.date()) for n in neighbours) + "."
                ),
                "evidence": {
                    "anchor_day": str(anchor.date()),
                    "nearest_days": [str(n.date()) for n in neighbours],
                    "anchor_drivers": [{"metric": n, "z": round(v, 2)} for n, v in drv],
                },
            })
    except Exception as e:
        logging.error("day_vector similar-days failed for '%s': %s", parameters.get("name"), e)

    if not insights:
        return None

    logging.info(
        "day_vector '%s': %d insights (%d dims, %d day-vectors)",
        parameters.get("name", "day_vector"), len(insights), X.shape[1], X.shape[0],
    )
    return insights
