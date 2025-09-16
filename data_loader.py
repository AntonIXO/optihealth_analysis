import pandas as pd
import logging
from database import (
    fetch_user_data_points, 
    fetch_user_events, 
    fetch_user_supplement_component_logs
)

def load_and_prepare_data(user_id, days=90):
    """
    Loads all relevant data for a user and prepares it for analysis.
    
    Returns:
        A tuple containing:
        - daily_summary_df (pd.DataFrame)
        - events_df (pd.DataFrame)
        - daily_supplements_df (pd.DataFrame): A new daily summary of supplement components.
    """
    logging.info(f"Loading data for user_id: {user_id}")

    # Fetch all data streams
    data_points_df = fetch_user_data_points(user_id, days)
    events_df = fetch_user_events(user_id, days)
    # Use the new function to fetch component data
    supplement_components_df = fetch_user_supplement_component_logs(user_id, days)

    # --- Prepare daily summary for biometric data ---
    if data_points_df.empty:
        logging.warning(f"No numeric data points found for user {user_id} in the last {days} days.")
        daily_summary_df = pd.DataFrame()
    else:
        data_points_df['date'] = data_points_df['timestamp'].dt.date
        daily_summary_df = data_points_df.pivot_table(
            index='date',
            columns='metric_name',
            values='value_numeric',
            aggfunc='mean'
        )
        logging.info(f"Prepared daily summary with {daily_summary_df.shape[0]} days of data and {daily_summary_df.shape[1]} metrics.")

    # --- Prepare daily summary for supplement components ---
    if supplement_components_df.empty:
        logging.info(f"No supplement logs found for user {user_id} in the last {days} days.")
        daily_supplements_df = pd.DataFrame()
    else:
        # Pivot the component data to have one column per component
        daily_supplements_df = supplement_components_df.pivot_table(
            index='date',
            columns='component_name',
            values='total_daily_amount',
            aggfunc='sum' # Should already be summed, but this is safe
        ).fillna(0) # Fill non-taken days with 0
        logging.info(f"Prepared daily supplement summary with {daily_supplements_df.shape[0]} days and {daily_supplements_df.shape[1]} unique components.")


    return daily_summary_df, events_df, daily_supplements_df

