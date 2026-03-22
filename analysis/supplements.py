import pandas as pd
import logging
import statsmodels.api as sm

def _run_regression_for_component(df, component_name, outcome_metric, confounding_metrics):
    """
    Runs a multiple regression for a single supplement component.
    """
    # Create the binary flag for the component (1 if taken, 0 if not)
    # We use a small threshold to account for any tiny leftover values
    df[component_name] = (df[component_name] > 0.001).astype(int)

    # Prepare data for the model
    # Ensure outcome_metric is the first column for easy access later
    model_df = df[[outcome_metric, component_name] + confounding_metrics].dropna()
    
    # Check if we have enough data and variance to run a model
    if len(model_df) < 20 or model_df[component_name].nunique() < 2:
        return None # Not enough data or only one group (e.g., always taken)

    # Define dependent (y) and independent (X) variables
    y = model_df[outcome_metric]
    X = model_df[[component_name] + confounding_metrics]
    X = sm.add_constant(X) # Add an intercept

    # Fit the Ordinary Least Squares (OLS) model
    model = sm.OLS(y, X).fit()
    
    # Extract results specifically for the component
    p_value = model.pvalues.get(component_name)
    coefficient = model.params.get(component_name)
    
    if p_value is not None and coefficient is not None:
        return {
            'p_value': p_value,
            'coefficient': coefficient,
            'component_name': component_name,
            'outcome_metric': outcome_metric,
            'n_obs': int(model.nobs)
        }
    return None

def find_best_supplement_impact(daily_summary_df, daily_supplements_df, parameters):
    """
    Iterates through all logged supplement components to find the most statistically
    significant impact on a list of outcome metrics.
    """
    outcome_metrics = parameters.get('outcome_metrics', [])
    confounding_metrics = parameters.get('confounding_metrics', [])
    p_value_threshold = parameters.get('p_value_threshold', 0.1)

    # --- Data Preparation ---
    if daily_summary_df.empty or daily_supplements_df.empty:
        return None
        
    # Combine biometric and supplement data. Use an outer join to keep all days.
    # Convert index to datetime if it's not already
    daily_summary_df.index = pd.to_datetime(daily_summary_df.index)
    daily_supplements_df.index = pd.to_datetime(daily_supplements_df.index)
    
    # Use a time lag for supplements. We assume the supplement taken today affects tomorrow's metrics.
    shifted_supplements = daily_supplements_df.shift(1, freq='D').add_suffix('_yesterday')
    
    combined_df = pd.merge(daily_summary_df, shifted_supplements, left_index=True, right_index=True, how='inner')

    all_results = []
    
    # Identify which supplement components are available for analysis
    available_components = [col.replace('_yesterday', '') for col in shifted_supplements.columns]

    # --- Iterative Analysis ---
    for component in available_components:
        component_col_shifted = f"{component}_yesterday"
        for outcome in outcome_metrics:
            # Check if all necessary columns exist in the combined frame
            required_cols = [outcome] + confounding_metrics + [component_col_shifted]
            if not all(col in combined_df.columns for col in required_cols):
                continue

            result = _run_regression_for_component(
                combined_df, 
                component_col_shifted, 
                outcome, 
                confounding_metrics
            )
            if result:
                all_results.append(result)

    if not all_results:
        return None

    # --- Find and Format the Best Result ---
    # Find the result with the lowest p-value
    best_result = min(all_results, key=lambda x: x['p_value'])

    if best_result['p_value'] < p_value_threshold:
        coef = best_result['coefficient']
        direction = "positive" if coef > 0 else "negative"
        outcome_metric = best_result['outcome_metric']
        component_name = best_result['component_name'].replace('_yesterday', '')

        title = f"Potential Link Found: {component_name} & {outcome_metric.replace('_', ' ').title()}"
        summary = (
            f"After accounting for lifestyle factors like {', '.join(confounding_metrics).replace('_', ' ')}, "
            f"taking {component_name} was associated with a statistically significant {direction} change "
            f"in your next-day {outcome_metric.replace('_', ' ')}. "
            f"(Change: {coef:.2f} units, p={best_result['p_value']:.3f})"
        )

        insight = {
            "type": "supplement_impact_discovery",
            "title": title,
            "summary": summary,
            "evidence": best_result
        }
        logging.info(f"Found significant supplement impact for '{parameters['name']}': {component_name} on {outcome_metric}")
        return insight

    return None

