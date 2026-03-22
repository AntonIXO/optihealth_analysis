import pandas as pd
import numpy as np
import yaml
import logging
import importlib
from datetime import date, timedelta

# Configure logging to show test results clearly
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Mock Data Generation ---

def generate_mock_data(days=90):
    """
    Generates a realistic, patterned dataset for testing all analysis modules.
    """
    logging.info(f"Generating mock data for {days} days...")
    
    # Create a date range for the last 90 days
    today = date.today()
    date_range = pd.to_datetime([today - timedelta(days=i) for i in range(days)]).sort_values()

    # --- Create Core Patterns ---
    np.random.seed(42)
    # Pattern 1: Base activity (steps)
    base_steps = np.random.randint(4000, 15000, size=days)
    
    # Pattern 2: Sleep is somewhat random but influenced by very high activity
    base_sleep = np.random.uniform(380, 500, size=days)
    base_sleep[base_steps > 12000] -= 30 # High activity days can reduce sleep
    
    # Pattern 3: Resting HR is strongly negatively correlated with steps
    base_hr = 65 - (base_steps - np.mean(base_steps)) / 1000 + np.random.normal(0, 1, size=days)
    
    # Pattern 4: HRV has a clear upward trend + one anomaly
    base_hrv = np.linspace(35, 50, days) + np.random.normal(0, 2, size=days)
    base_hrv[days // 2] = 20 # Inject a clear anomaly mid-way through
    
    # Pattern 5: Sleep Score is a function of steps and sleep duration
    sleep_score = 50 + (base_sleep - np.mean(base_sleep)) / 10 + (base_steps - np.mean(base_steps)) / 1000
    sleep_score = np.clip(sleep_score, 40, 95)

    # --- Assemble Daily Summary DataFrame ---
    daily_summary_df = pd.DataFrame({
        'activity_steps': base_steps,
        'sleep_duration_total': base_sleep,
        'hr_resting': base_hr,
        'hrv_rmssd': base_hrv,
        'sleep_score': sleep_score,
        'sleep_duration_deep': base_sleep / 4, # Simplified relationship
        'nutrition_calories': np.random.randint(1800, 3000, size=days),
        'respiratory_rate': np.random.uniform(12, 18, size=days),
        'body_temperature': np.random.uniform(36.5, 37.2, size=days)
    }, index=date_range)
    
    # Introduce some missing data to test robustness
    for col in daily_summary_df.columns:
        mask = np.random.rand(days) < 0.10 # 10% missing data
        daily_summary_df.loc[mask, col] = np.nan

    # --- Assemble Events DataFrame ---
    event_dates = daily_summary_df.sample(n=20, random_state=42).index
    events_df = pd.DataFrame({
        'event_name': 'Running',
        'start_timestamp': event_dates
    })
    
    # Make the "comparative analysis" pattern explicit
    # Lower resting HR on the day after a run
    for dt in event_dates:
        next_day = dt + pd.Timedelta(days=1)
        if next_day in daily_summary_df.index:
            daily_summary_df.loc[next_day, 'hr_resting'] -= 5

    logging.info("Mock data generation complete.")
    return daily_summary_df.sort_index(), events_df.sort_values('start_timestamp')


# --- Test Runner Logic ---

def main():
    """
    Main function to run tests on all analysis modules.
    """
    logging.info("--- Starting OptiHealth Analysis Module Test Runner ---")
    
    # 1. Load Analysis Configuration
    try:
        with open('analysis_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        logging.info("Successfully loaded analysis_config.yaml")
    except FileNotFoundError:
        logging.error("FATAL: analysis_config.yaml not found. Aborting.")
        return
        
    # 2. Generate Mock Data
    daily_df, events_df = generate_mock_data()
    
    # 3. Iterate and Test Each Analysis
    for i, analysis_def in enumerate(config.get('analyses', [])):
        name = analysis_def.get('name', f'Unnamed Analysis {i+1}')
        module_name = analysis_def.get('module')
        function_name = analysis_def.get('function')
        
        print("\n" + "="*80)
        logging.info(f"RUNNING TEST FOR: [{name}]")
        print("="*80)

        if not module_name or not function_name:
            logging.error(f"Skipping '{name}' due to missing 'module' or 'function' in config.")
            continue
            
        try:
            # Dynamically import the required analysis function
            analysis_module = importlib.import_module(f"analysis.{module_name}")
            analysis_function = getattr(analysis_module, function_name)
            
            # Prepare arguments
            params = analysis_def.get('parameters', {})
            params['name'] = name # Pass name for logging within the module
            
            # Call the analysis function with the appropriate data
            result = None
            if module_name in ['correlation', 'clustering', 'feature_importance', 'forecasting']:
                result = analysis_function(daily_df.copy(), params)
            elif module_name == 'comparative':
                result = analysis_function(daily_df.copy(), events_df.copy(), params)
            
            # 4. Validate and Report Result
            if result:
                logging.info(f"✅ PASSED: Insight successfully generated for '{name}'.")
                print(f"   Insight Type: {result.get('type')}")
                print(f"   Title: {result.get('title')}")
                print(f"   Summary: {result.get('summary')}")
            else:
                logging.warning(f"❌ FAILED or SKIPPED: No significant insight generated for '{name}'. This may be expected if data patterns aren't strong enough.")

        except ImportError:
            logging.error(f"❌ FAILED: Could not import module 'analysis.{module_name}'. Check file name.")
        except AttributeError:
            logging.error(f"❌ FAILED: Function '{function_name}' not found in 'analysis.{module_name}'.")
        except Exception as e:
            logging.error(f"❌ FAILED: An unexpected error occurred during execution of '{name}': {e}", exc_info=True)
            
    print("\n" + "="*80)
    logging.info("--- Test run complete. ---")

if __name__ == "__main__":
    main()
