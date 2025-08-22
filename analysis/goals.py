import pandas as pd
import logging

def _calculate_streak(daily_adherence_series):
    """
    Calculates the current streak of consecutive True values ending on the most recent day.
    
    Args:
        daily_adherence_series (pd.Series): A boolean series indexed by date,
                                            where True means the goal was met.
    
    Returns:
        int: The length of the current streak.
    """
    if not isinstance(daily_adherence_series.index, pd.DatetimeIndex):
        daily_adherence_series.index = pd.to_datetime(daily_adherence_series.index)

    # Sort by date to ensure correctness
    s = daily_adherence_series.sort_index()
    
    # If the most recent day was a failure, the current streak is 0
    if not s.iloc[-1]:
        return 0
        
    # Calculate consecutive groups of True values
    # A cumulative sum over the inverted series creates groups of consecutive Trues
    streaks = s.cumsum()[~s].diff().fillna(s.cumsum())
    
    # The current streak is the length of the last group of Trues
    current_streak = (s.index.to_series().groupby(streaks).cumcount() + 1)[s].iloc[-1]
    
    return int(current_streak)


def run_goal_adherence_analysis(daily_df, parameters, user_goals):
    """
    Analyzes user's daily data against their defined goals to find streaks.

    Args:
        daily_df (pd.DataFrame): DataFrame with metrics as columns and date as index.
        parameters (dict): A dictionary containing analysis parameters from the config.
        user_goals (list): A list of goal dictionaries fetched from the database.

    Returns:
        list: A list of formatted insight dictionaries, one for each significant streak.
    """
    if not user_goals:
        logging.info("Goals: No active goals found for this user.")
        return []

    min_streak = parameters.get('min_streak_for_insight', 3)
    insights = []

    for goal in user_goals:
        metric = goal.get('metric_name')
        target = goal.get('target_value')
        op = goal.get('operator')
        goal_name = goal.get('goal_name')

        if metric not in daily_df.columns:
            continue

        # Create a boolean series indicating if the goal was met each day
        metric_series = daily_df[metric].dropna()
        if metric_series.empty:
            continue

        if op == '>=':
            adherence = metric_series >= target
        elif op == '<=':
            adherence = metric_series <= target
        elif op == '>':
            adherence = metric_series > target
        elif op == '<':
            adherence = metric_series < target
        else:
            continue

        # Calculate the current streak
        current_streak = _calculate_streak(adherence)

        # Generate an insight if the streak is significant
        if current_streak >= min_streak:
            insight = {
                "type": "goal_streak",
                "title": f"You're on a {current_streak}-Day Streak!",
                "summary": f"You've successfully met your '{goal_name}' goal for {current_streak} days in a row. Keep up the great momentum!",
                "evidence": {
                    "goal_name": goal_name,
                    "metric": metric,
                    "streak_length": current_streak,
                    "goal_condition": f"{op} {target}",
                    "most_recent_value": round(metric_series.iloc[-1], 2)
                }
            }
            insights.append(insight)
            logging.info(f"Found significant streak of {current_streak} days for goal '{goal_name}'")

    return insights