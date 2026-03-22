import pandas as pd
import logging
from database import fetch_user_goals # Make sure to import the new function

def _calculate_streak(daily_adherence: pd.Series) -> int:
    """Calculates the current streak of True values ending today."""
    if daily_adherence.empty or not daily_adherence.iloc[-1]:
        return 0
    
    # Invert the series (so False is a "break") and find the first break from the end
    breaks = (~daily_adherence).cumsum()
    # The streak is the length of the last group of non-breaks
    current_streak = len(daily_adherence) - breaks.idxmax() if breaks.max() > 0 and not daily_adherence.loc[breaks.idxmax()] else len(daily_adherence)
    
    # A simpler way if the above is complex for some pandas versions:
    streak = 0
    for met_goal in reversed(daily_adherence.values):
        if met_goal:
            streak += 1
        else:
            break
    return streak


def run_goal_adherence_analysis(daily_df, user_id, parameters):
    """
    Analyzes user's daily data against their active goals to find streaks.
    """
    min_streak = parameters.get('min_streak_for_insight', 3)
    
    # 1. Fetch user's active goals
    goals_df = fetch_user_goals(user_id)
    if goals_df.empty:
        return None # Return a list if multiple insights can be generated

    insights = []

    # 2. Evaluate each goal
    for _, goal in goals_df.iterrows():
        metric = goal['metric_name']
        target = goal['target_value']
        operator = goal['operator']

        if metric not in daily_df.columns:
            continue

        # Create a boolean series for goal adherence
        metric_series = daily_df[metric].dropna()
        if operator == '>=':
            adherence = metric_series >= target
        elif operator == '<=':
            adherence = metric_series <= target
        elif operator == '>':
            adherence = metric_series > target
        elif operator == '<':
            adherence = metric_series < target
        else:
            continue
        
        # 3. Calculate streak
        current_streak = _calculate_streak(adherence)

        # 4. Generate insight if streak is significant
        if current_streak >= min_streak:
            insight = {
                "type": "goal_streak",
                "title": f"You're on a {current_streak}-Day Streak!",
                "summary": f"Great job! You've successfully met your goal of '{goal['goal_name']}' for {current_streak} days in a row.",
                "evidence": {
                    "goal_name": goal['goal_name'],
                    "metric": metric,
                    "target": f"{operator} {target}",
                    "current_streak": current_streak,
                    "days_checked": len(metric_series)
                }
            }
            insights.append(insight)
    
    return insights if insights else None
