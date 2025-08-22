import pandas as pd
import logging
from prophet import Prophet

def run_forecasting_and_anomaly_analysis(daily_df, parameters):
    """
    Uses Prophet to forecast a metric's trend and detect anomalies.

    Args:
        daily_df (pd.DataFrame): DataFrame with metrics as columns and date as index.
        parameters (dict): A dictionary containing analysis parameters from the config.

    Returns:
        dict: A formatted insight dictionary if a significant trend or anomaly is found, otherwise None.
    """
    metric = parameters.get('metric_to_forecast')
    forecast_days = parameters.get('days_to_forecast', 14)
    min_days = parameters.get('min_days', 60)
    desired_trend = parameters.get('desired_trend', 'increasing')
    # interval_width is the confidence interval. 0.99 means we are 99% confident
    # that the true value falls within the yhat_lower and yhat_upper bounds.
    sensitivity = parameters.get('anomaly_sensitivity', 0.99)

    # 1. Data Preparation
    if metric not in daily_df.columns:
        return None

    # Prophet requires specific column names: 'ds' for date and 'y' for value
    prophet_df = daily_df[[metric]].dropna().reset_index()
    prophet_df = prophet_df.rename(columns={'index': 'ds', metric: 'y'})
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

    if len(prophet_df) < min_days:
        return None

    # 2. Model Training and Forecasting
    # Suppress verbose Prophet output
    m = Prophet(interval_width=sensitivity, daily_seasonality=True)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=forecast_days)
    forecast = m.predict(future)

    # 3. Anomaly Detection
    # Merge forecast with actuals
    results = pd.merge(prophet_df, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
    # An anomaly is where the actual value ('y') is outside the confidence interval
    results['anomaly'] = (results['y'] < results['yhat_lower']) | (results['y'] > results['yhat_upper'])
    
    anomalies = results[results['anomaly']]
    if not anomalies.empty:
        # For simplicity, we'll create one insight for the most recent anomaly if found.
        latest_anomaly = anomalies.sort_values('ds', ascending=False).iloc[0]
        direction = "higher" if latest_anomaly['y'] > latest_anomaly['yhat'] else "lower"
        
        insight = {
            "type": "anomaly_detection",
            "title": f"Unusual {metric.replace('_', ' ').title()} Detected",
            "summary": f"Your {metric.replace('_', ' ')} on {latest_anomaly['ds'].strftime('%B %d')} was statistically unusual, being significantly {direction} than your typical range.",
            "evidence": {
                "metric": metric,
                "date": latest_anomaly['ds'].strftime('%Y-%m-%d'),
                "actual_value": round(latest_anomaly['y'], 2),
                "expected_range": [round(latest_anomaly['yhat_lower'], 2), round(latest_anomaly['yhat_upper'], 2)],
            }
        }
        logging.info(f"Found anomaly for '{parameters['name']}' on {latest_anomaly['ds'].strftime('%Y-%m-%d')}")
        # We return the anomaly insight and stop, as it's often more timely/important than a trend.
        return insight

    # 4. Trend Analysis
    # Compare the trend at the start vs. the end of the forecast period
    start_trend = forecast['trend'].iloc[len(prophet_df) - 1] # Last actual data point
    end_trend = forecast['trend'].iloc[-1] # End of forecast
    
    trend_diff = end_trend - start_trend
    
    # Check if the trend is significant and matches the desired direction
    is_significant_trend = False
    if desired_trend == 'increasing' and trend_diff > 0:
        is_significant_trend = True
        trend_direction = "positive"
        change_desc = f"projected to increase by ~{abs(trend_diff):.2f}"
    elif desired_trend == 'decreasing' and trend_diff < 0:
        is_significant_trend = True
        trend_direction = "positive"
        change_desc = f"projected to decrease by ~{abs(trend_diff):.2f}"
    elif trend_diff != 0: # A non-zero trend in the wrong direction
        trend_direction = "negative"
        change_desc = f"projected to {'increase' if trend_diff > 0 else 'decrease'} by ~{abs(trend_diff):.2f}"
    else: # Flat trend
        return None

    if is_significant_trend:
        insight = {
            "type": "trend_forecast",
            "title": f"{trend_direction.title()} Trend in {metric.replace('_', ' ').title()}",
            "summary": f"Based on your recent data, your {metric.replace('_', ' ')} is on a {trend_direction} trajectory and is {change_desc} over the next {forecast_days} days. Keep up the great work!",
            "evidence": {
                "metric": metric,
                "forecast_period_days": forecast_days,
                "projected_change": round(trend_diff, 2),
                "current_value": round(prophet_df['y'].iloc[-1], 2),
                "projected_end_value": round(forecast['yhat'].iloc[-1], 2)
            }
        }
        logging.info(f"Found significant {trend_direction} trend for '{parameters['name']}'")
        return insight

    return None