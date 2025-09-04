import pandas as pd
import logging
from database import fetch_user_data_points, fetch_user_events

def load_and_prepare_data(user_id, days=90):
    """
    Loads all relevant data for a user and prepares it for analysis.
    NOTE: The 'conn' parameter is no longer needed as the database functions now use the global engine.

    Returns:
        A tuple containing:
        - daily_summary_df (pd.DataFrame): Data pivoted with metrics as columns and daily dates as index.
        - events_df (pd.DataFrame): Raw events data.
    """
    logging.info(f"Loading data for user_id: {user_id}")

    # Fetch raw data points and events
    data_points_df = fetch_user_data_points(user_id, days)
    events_df = fetch_user_events(user_id, days)

    if data_points_df.empty:
        logging.warning(f"No numeric data points found for user {user_id} in the last {days} days.")
        return pd.DataFrame(), events_df

    # Pivot data_points to have metrics as columns, indexed by day
    # We take the mean value for metrics that might be logged multiple times a day
    data_points_df['date'] = data_points_df['timestamp'].dt.date
    daily_summary_df = data_points_df.pivot_table(
        index='date',
        columns='metric_name',
        values='value_numeric',
        aggfunc='mean'
    )

    logging.info(f"Prepared daily summary with {daily_summary_df.shape[0]} days of data and {daily_summary_df.shape[1]} metrics.")

    return daily_summary_df, events_df
