import yaml
import time
import logging
import importlib
from data_loader import load_and_prepare_data
from database import fetch_pending_job, update_job_status, store_insights

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

POLL_INTERVAL_SECONDS = 2

def load_analysis_config():
    """Loads the analysis definitions from the YAML file."""
    with open('analysis_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def run_insight_engine(user_id, job_id):
    """Orchestrates the analysis process for a single job."""
    logging.info(f"Starting insight engine for job {job_id}, user {user_id}")

    # 1. Load all data streams for the user
    daily_summary_df, events_df, daily_supplements_df = load_and_prepare_data(user_id)

    # 2. Load analysis configuration
    config = load_analysis_config()
    significant_insights = []

    # 3. Iteratively run each defined analysis
    for analysis_def in config.get('analyses', []):
        if not analysis_def.get('enabled', False):
            continue

        try:
            module_name = analysis_def['module']
            function_name = analysis_def['function']
            
            # Dynamically import the analysis module and get the function
            analysis_module = importlib.import_module(f"analysis.{module_name}")
            analysis_function = getattr(analysis_module, function_name)
            
            # Prepare parameters for the function
            params_with_name = analysis_def.get('parameters', {}).copy()
            params_with_name['name'] = analysis_def['name'] # Pass name for logging

            # Call the correct function with the right data arguments
            if module_name == 'supplements':
                result = analysis_function(daily_summary_df, daily_supplements_df, params_with_name)
            elif module_name == 'comparative':
                result = analysis_function(daily_summary_df, events_df, params_with_name)
            else: # For correlation, clustering, feature_importance, forecasting, goals
                result = analysis_function(daily_summary_df, params_with_name)

            # 4. Collect significant results.
            # Modules may return a single insight dict OR a list of insight dicts
            # (correlation_discovery, lagged_correlation, goals). Normalise both.
            if result:
                results = result if isinstance(result, list) else [result]
                for item in results:
                    item['user_id'] = user_id
                    significant_insights.append(item)

        except Exception as e:
            logging.error(f"Error running analysis '{analysis_def['name']}' for user {user_id}: {e}", exc_info=True)
            continue
    
    # 5. Store all significant findings
    if significant_insights:
        store_insights(significant_insights)
    else:
        logging.info(f"No new significant insights found for user {user_id}.")


def main():
    """Main worker loop to poll for and process jobs."""
    logging.info("Starting OptiHealth Analysis Worker...")
    while True:
        job = None
        try:
            job = fetch_pending_job()
            if job:
                logging.info(f"Picked up job {job['id']} for user {job['user_id']}")
                run_insight_engine(job['user_id'], job['id'])
                update_job_status(job['id'], 'completed')
            else:
                logging.info(f"No pending jobs found. Sleeping for {POLL_INTERVAL_SECONDS} seconds.")
                time.sleep(POLL_INTERVAL_SECONDS)

        except Exception as e:
            job_id = job['id'] if job else 'N/A'
            logging.error(f"An unexpected error occurred while processing job {job_id}: {e}", exc_info=True)
            if job:
                update_job_status(job['id'], 'failed')
            time.sleep(60) # Wait longer after a major error

if __name__ == "__main__":
    main()
