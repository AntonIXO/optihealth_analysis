import xgboost as xgb
import numpy as np
from sklearn.model_selection import cross_val_score
import pandas as pd
import logging


def run_feature_importance_analysis(daily_df, parameters):
    """
    Uses an XGBoost model to find the most influential factors for a target metric.

    IMPORTANT: feature importances are only meaningful if the model actually
    predicts the target out-of-sample. On small n (tens of rows) XGBoost with
    100 trees will happily overfit and report confident-looking importances that
    are pure noise. We therefore gate on a cross-validated R^2: if the model does
    not generalize (mean CV R^2 < min_cv_r2), we emit NOTHING.
    """
    target = parameters.get('target_metric')
    features = parameters.get('feature_metrics', [])
    min_days = parameters.get('min_days', 45)
    top_n = parameters.get('top_n_features', 3)
    min_cv_r2 = parameters.get('min_cv_r2', 0.05)

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

    # 2. Cross-validated generalization check (the honesty gate).
    # cv folds scale with n but stay in [3, 5]; need at least ~3 samples/fold.
    n = len(analysis_df)
    cv = max(3, min(5, n // 10))
    if n < cv * 3:
        return None

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    try:
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    except Exception as e:
        logging.error(f"feature_importance CV failed for '{parameters.get('name')}': {e}")
        return None
    mean_cv_r2 = float(np.mean(cv_scores))

    if mean_cv_r2 < min_cv_r2:
        logging.info(
            "feature_importance '%s': model does not generalize (CV R2=%.3f < %.3f) -> no insight",
            parameters.get('name'), mean_cv_r2, min_cv_r2,
        )
        return None

    # 3. Fit on full data (model earned the right to be interpreted) + importances
    model.fit(X, y)
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances,
    }).sort_values('importance', ascending=False)

    if importance_df.empty:
        return None

    top_features = importance_df.head(top_n)
    top_feature_list = top_features['feature'].str.replace('_', ' ').str.title().tolist()

    if len(top_feature_list) > 1:
        top_features_str = ", ".join(top_feature_list[:-1]) + f", and {top_feature_list[-1]}"
    else:
        top_features_str = top_feature_list[0]

    variance_pct = round(max(0.0, mean_cv_r2) * 100, 1)
    insight = {
        "type": "feature_importance",
        "title": f"What Drives Your {target.replace('_', ' ').title()}?",
        "summary": (
            f"For you, the most powerful predictors of {target.replace('_', ' ')} are: "
            f"{top_features_str}. This model explains ~{variance_pct}% of the "
            f"variance out-of-sample."
        ),
        "evidence": {
            "target_metric": target,
            "model_type": "XGBoost Regressor",
            "cv_r2": round(mean_cv_r2, 3),
            "cv_folds": cv,
            "top_factors": top_features.to_dict('records'),
            "data_points_used": len(analysis_df),
        },
    }
    logging.info(
        "feature_importance '%s': validated (CV R2=%.3f), top predictors reported",
        parameters.get('name'), mean_cv_r2,
    )
    return insight
