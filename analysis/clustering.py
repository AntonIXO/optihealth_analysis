from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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
    """Performs unsupervised clustering to identify 'Day Types'."""
    metrics = parameters.get('metrics_to_cluster', [])
    outcome_metric = parameters.get('outcome_metric')
    n_clusters = parameters.get('n_clusters', 3)
    min_days = parameters.get('min_days', 30)

    required_cols = metrics + [outcome_metric]
    if not all(col in daily_df.columns for col in required_cols):
        return None

    analysis_df = daily_df[required_cols].dropna()
    if len(analysis_df) < min_days:
        return None

    feature_df = analysis_df[metrics]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(scaled_features)
    analysis_df['cluster'] = kmeans.labels_

    profiles = _get_cluster_profiles(analysis_df, metrics, outcome_metric)
    if not profiles: return None

    best_cluster_id = max(profiles, key=lambda k: profiles[k][outcome_metric])
    best_profile = profiles[best_cluster_id]

    insight = {
        "type": "clustering",
        "title": f"You Have {n_clusters} Distinct 'Day Types'",
        "summary": f"Your days with the highest '{outcome_metric.replace('_', ' ')}' are typically '{best_profile['description']}' days, occurring {best_profile['size_percent']}% of the time.",
        "evidence": {
            "outcome_metric": outcome_metric,
            "cluster_profiles": {f"Cluster {cid} ({p['description']})": {'size_percent': p['size_percent'], 'avg_outcome': round(p[outcome_metric], 2)} for cid, p in profiles.items()}
        }
    }
    logging.info(f"Found significant Day Type clusters for '{parameters['name']}'")
    return insight