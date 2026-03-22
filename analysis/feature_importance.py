import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import logging

def run_feature_importance_analysis(daily_df, parameters):
    """
    Uses an XGBoost model to find the most influential factors for a target metric.
    """
    target = parameters.get('target_metric')
    features = parameters.get('feature_metrics', [])
    min_days = parameters.get('min_days', 45)
    top_n = parameters.get('top_n_features', 3)

    # 1. Data Preparation
    # Ensure target is not in features
    features = [f for f in features if f != target]
    
    # Filter for available columns in user's datalog
    available_features = [f for f in features if f in daily_df.columns]
    if not available_features or target not in daily_df.columns:
        return None

    analysis_df = daily_df[[target] + available_features].dropna()
    if len(analysis_df) < min_days:
        return None

    X = analysis_df[available_features]
    y = analysis_df[target]

    # 2. Model Training
    # Split data for a basic validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 3. Feature Importance Extraction
    importances = model.feature_importances_
    feature_names = X.columns
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # 4. Insight Generation
    if importance_df.empty:
        return None

    top_features = importance_df.head(top_n)
    top_feature_list = top_features['feature'].str.replace('_', ' ').str.title().tolist()

    # Create a readable list for the summary
    if len(top_feature_list) > 1:
        top_features_str = ", ".join(top_feature_list[:-1]) + f", and {top_feature_list[-1]}"
    else:
        top_features_str = top_feature_list[0]

    insight = {
        "type": "feature_importance",
        "title": f"What Drives Your {target.replace('_', ' ').title()}?",
        "summary": f"For you, the most powerful predictors of {target.replace('_', ' ')} are: {top_features_str}.",
        "evidence": {
            "target_metric": target,
            "model_type": "XGBoost Regressor",
            "top_factors": top_features.to_dict('records'),
            "data_points_used": len(analysis_df)
        }
    }
    logging.info(f"Found top predictors for '{parameters['name']}'")
    return insight