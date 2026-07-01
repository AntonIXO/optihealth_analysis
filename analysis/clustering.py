from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import f_oneway
import logging


def _get_cluster_profiles(df_with_clusters, metrics_to_cluster, outcome_metric):
    """Analyzes the characteristics of each cluster."""
    profiles = {}
    cluster_summary = df_with_clusters.groupby('cluster')[metrics_to_cluster + [outcome_metric]].mean()
    cluster_sizes = df_with_clusters['cluster'].value_counts(normalize=True).sort_index()

    for i in range(len(cluster_summary)):
        profile = cluster_summary.iloc[i].to_dict()
        profile['size_percent'] = round(cluster_sizes.iloc[i] * 100, 1)
        descriptions = []
        for metric in metrics_to_cluster:
            others_mean = cluster_summary[metric].drop(i).mean()
            this_mean = profile[metric]
            if this_mean > others_mean * 1.1:
                descriptions.append(f"High {metric.replace('_', ' ')}")
            elif this_mean < others_mean * 0.9:
                descriptions.append(f"Low {metric.replace('_', ' ')}")
        profile['description'] = ", ".join(descriptions) if descriptions else "Average"
        profiles[i] = profile
    return profiles


def run_day_clustering_analysis(daily_df, parameters):
    """
    Performs unsupervised clustering to identify 'Day Types'.

    KMeans ALWAYS returns clusters, so "you have N day types" is only a real
    insight if (a) the clusters are actually separated in feature space and
    (b) they differ in the outcome metric. We therefore gate on:
      - silhouette score >= min_silhouette   (clusters are cohesive/separated)
      - one-way ANOVA p < max_anova_p         (clusters differ in the outcome)
    Otherwise we emit nothing rather than dress up noise as structure.
    """
    metrics = parameters.get('metrics_to_cluster', [])
    outcome_metric = parameters.get('outcome_metric')
    n_clusters = parameters.get('n_clusters', 3)
    min_days = parameters.get('min_days', 30)
    min_silhouette = parameters.get('min_silhouette', 0.15)
    max_anova_p = parameters.get('max_anova_p', 0.05)

    required_cols = metrics + [outcome_metric]
    if not all(col in daily_df.columns for col in required_cols):
        return None

    analysis_df = daily_df[required_cols].dropna()
    if len(analysis_df) < min_days or len(analysis_df) <= n_clusters:
        return None

    feature_df = analysis_df[metrics]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(scaled_features)
    analysis_df = analysis_df.copy()
    analysis_df['cluster'] = kmeans.labels_

    # Gate 1: are the clusters actually separated?
    if len(set(kmeans.labels_)) < 2:
        return None
    try:
        sil = float(silhouette_score(scaled_features, kmeans.labels_))
    except Exception as e:
        logging.error(f"clustering silhouette failed for '{parameters.get('name')}': {e}")
        return None
    if sil < min_silhouette:
        logging.info(
            "clustering '%s': clusters not separated (silhouette=%.3f < %.3f) -> no insight",
            parameters.get('name'), sil, min_silhouette,
        )
        return None

    # Gate 2: do the clusters differ in the outcome metric?
    groups = [g[outcome_metric].values for _, g in analysis_df.groupby('cluster') if len(g) >= 2]
    if len(groups) < 2:
        return None
    try:
        _, anova_p = f_oneway(*groups)
    except Exception as e:
        logging.error(f"clustering ANOVA failed for '{parameters.get('name')}': {e}")
        return None
    if anova_p is None or anova_p >= max_anova_p:
        logging.info(
            "clustering '%s': clusters do not separate outcome (ANOVA p=%.3f) -> no insight",
            parameters.get('name'), (anova_p if anova_p is not None else float('nan')),
        )
        return None

    profiles = _get_cluster_profiles(analysis_df, metrics, outcome_metric)
    if not profiles:
        return None

    best_cluster_id = max(profiles, key=lambda k: profiles[k][outcome_metric])
    best_profile = profiles[best_cluster_id]

    insight = {
        "type": "clustering",
        "title": f"You Have {n_clusters} Distinct 'Day Types'",
        "summary": (
            f"Your days with the highest '{outcome_metric.replace('_', ' ')}' are typically "
            f"'{best_profile['description']}' days, occurring {best_profile['size_percent']}% of the time."
        ),
        "evidence": {
            "outcome_metric": outcome_metric,
            "silhouette": round(sil, 3),
            "anova_p": round(float(anova_p), 5),
            "cluster_profiles": {
                f"Cluster {cid} ({p['description']})": {
                    'size_percent': p['size_percent'],
                    'avg_outcome': round(p[outcome_metric], 2),
                } for cid, p in profiles.items()
            },
        },
    }
    logging.info(
        "clustering '%s': validated (silhouette=%.3f, ANOVA p=%.4f)",
        parameters.get('name'), sil, anova_p,
    )
    return insight
