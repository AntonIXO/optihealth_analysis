from scipy.stats import ttest_ind, mannwhitneyu
import logging
import pandas as pd

def run_comparative_analysis(daily_df, events_df, parameters):
    """Compares a metric between two groups: days after an event vs. other days."""
    event_name = parameters.get('event_name')
    metric = parameters.get('metric')
    window = pd.Timedelta(days=parameters.get('time_window_days', 1))
    test_type = parameters.get('analysis_type', 'mannwhitneyu')
    min_size = parameters.get('min_group_size', 5)
    p_value_threshold = parameters.get('significance_threshold', 0.05)

    if metric not in daily_df.columns or events_df.empty:
        return None

    target_events = events_df[events_df['event_name'] == event_name]
    if target_events.empty:
        return None

    event_dates = target_events['start_timestamp'].dt.date
    post_event_dates = {date + i * pd.Timedelta(days=1) for date in event_dates for i in range(1, window.days + 1)}
    
    daily_df.index = pd.to_datetime(daily_df.index).date
    group_a = daily_df[daily_df.index.isin(post_event_dates)][metric].dropna()
    group_b = daily_df[~daily_df.index.isin(post_event_dates)][metric].dropna()

    if len(group_a) < min_size or len(group_b) < min_size:
        return None

    if test_type == 'ttest':
        stat, p_value = ttest_ind(group_a, group_b, equal_var=False)
    else:
        stat, p_value = mannwhitneyu(group_a, group_b)

    if p_value < p_value_threshold:
        mean_a = group_a.mean()
        mean_b = group_b.mean()
        diff = mean_a - mean_b
        direction = "higher" if diff > 0 else "lower"
        insight = {
            "type": "comparative",
            "title": f"Impact of '{event_name}' on {metric.replace('_', ' ').title()}",
            "summary": f"Your {metric.replace('_', ' ').title()} is significantly {direction} (by ~{abs(diff):.2f}) on days following a '{event_name}' event.",
            "evidence": {
                "p_value": round(p_value, 5),
                "group_a_mean": round(mean_a, 2),
                "group_b_mean": round(mean_b, 2),
            }
        }
        logging.info(f"Found significant comparison for '{parameters['name']}': p_value={p_value:.5f}")
        return insight
    return None