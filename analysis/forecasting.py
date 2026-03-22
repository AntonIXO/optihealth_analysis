import pandas as pd
import logging
from prophet import Prophet

def run_forecasting_and_anomaly_analysis(daily_df, parameters):
    """
    Uses Prophet to forecast a metric's trend and detect anomalies.
    """
    metric = parameters.get('metric_to_forecast')
    forecast_days = parameters.get('days_to_forecast', 14)
    min_days = parameters.get('min_days', 60)
    desired_trend = parameters.get('desired_trend', 'increasing')
    sensitivity = parameters.get('anomaly_sensitivity', 0.99)

    # 1. Data Preparation
    if metric not in daily_df.columns:
        return None

    # --- Start of Fix ---
    # Create the DataFrame for Prophet directly with the correct column names ('ds', 'y').
    # This avoids the error-prone rename step.
    prophet_df = pd.DataFrame({
        'ds': daily_df.index,
        'y': daily_df[metric]
    }).dropna()
    # --- End of Fix ---

    if len(prophet_df) < min_days:
        return None

    # 2. Model Training and Forecasting
    m = Prophet(interval_width=sensitivity, daily_seasonality=True)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=forecast_days)
    forecast = m.predict(future)

    # 3. Anomaly Detection
    # Merge the original data with the forecast results
    results = pd.merge(prophet_df, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
    results['anomaly'] = (results['y'] < results['yhat_lower']) | (results['y'] > results['yhat_upper'])
    
    anomalies = results[results['anomaly']]
    if not anomalies.empty:
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
        return insight

    # 4. Trend Analysis
    # Get the trend value at the end of the historical data and at the end of the forecast
    start_trend = forecast['trend'].iloc[len(prophet_df) - 1]
    end_trend = forecast['trend'].iloc[-1]
    trend_diff = end_trend - start_trend
    
    is_significant_trend = False
    trend_direction = ""
    change_desc = ""

    if desired_trend == 'increasing' and trend_diff > 0:
        is_significant_trend = True
        trend_direction = "positive"
        change_desc = f"projected to increase by ~{abs(trend_diff):.2f}"
    elif desired_trend == 'decreasing' and trend_diff < 0:
        is_significant_trend = True
        trend_direction = "positive" # A decreasing resting HR is a positive trend
        change_desc = f"projected to decrease by ~{abs(trend_diff):.2f}"
    else:
        # If the trend is flat or moving in the non-desired direction, we don't generate an insight.
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
