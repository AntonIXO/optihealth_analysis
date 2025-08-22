import numpy as np
from scipy.stats import pearsonr, spearmanr
import logging

def run_correlation_analysis(daily_df, parameters):
    """Calculates Pearson or Spearman correlation between two metrics."""
    metric_a = parameters.get('metric_a')
    metric_b = parameters.get('metric_b')
    method = parameters.get('method', 'pearson')
    min_points = parameters.get('min_data_points', 20)
    threshold = parameters.get('significance_threshold', 0.4)

    if metric_a not in daily_df.columns or metric_b not in daily_df.columns:
        return None

    analysis_df = daily_df[[metric_a, metric_b]].dropna()
    if len(analysis_df) < min_points:
        return None

    x = analysis_df[metric_a]
    y = analysis_df[metric_b]

    if method == 'pearson':
        corr, p_value = pearsonr(x, y)
    elif method == 'spearman':
        corr, p_value = spearmanr(x, y)
    else:
        return None

    if abs(corr) >= threshold and not np.isnan(corr):
        relationship = "positively" if corr > 0 else "negatively"
        strength = "strongly" if abs(corr) > 0.6 else "moderately"
        insight = {
            "type": "correlation",
            "title": f"Link Between {metric_a.replace('_', ' ').title()} and {metric_b.replace('_', ' ').title()}",
            "summary": f"Your {metric_a.replace('_', ' ').title()} seems to be {strength} {relationship} correlated with your {metric_b.replace('_', ' ').title()}.",
            "evidence": {
                "correlation_coefficient": round(corr, 3),
                "p_value": round(p_value, 5),
                "data_points_used": len(analysis_df),
            }
        }
        logging.info(f"Found significant correlation for '{parameters['name']}': corr={corr:.3f}")
        return insight
    return None