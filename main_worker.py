import yaml
import time
import logging
import importlib
import os # <-- Import os
from database import get_raw_connection, fetch_pending_job, update_job_status, store_insights
from data_loader import load_and_prepare_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

POLL_INTERVAL_SECONDS = 5

def load_analysis_config():
    """Loads the analysis definitions from the YAML file."""
    with open('analysis_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def run_insight_engine(user_id, job_id):
    """Orchestrates the analysis process for a single job."""
    logging.info(f"Starting insight engine for job {job_id}, user {user_id}")

    # 1. Load data for the user
    daily_summary_df, events_df = load_and_prepare_data(user_id)
    if daily_summary_df.empty and events_df.empty:
        logging.warning(f"No data to analyze for user {user_id}. Skipping.")
        return

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
            
            analysis_module = importlib.import_module(f"analysis.{module_name}")
            analysis_function = getattr(analysis_module, function_name)
            
            params_with_name = analysis_def['parameters'].copy()
            params_with_name['name'] = analysis_def['name']

            # Pass dataframes and other required context to analysis functions
            if module_name in ['correlation', 'clustering', 'feature_importance', 'forecasting']:
                result = analysis_function(daily_summary_df, params_with_name)
            elif module_name == 'comparative':
                result = analysis_function(daily_summary_df, events_df, params_with_name)
            elif module_name == 'goals':
                result = analysis_function(daily_summary_df, user_id, params_with_name)
            else:
                logging.warning(f"Analysis module '{module_name}' not recognized. Skipping.")
                continue

            if result:
                # Handle single insight or list of insights
                if isinstance(result, list):
                    for r in result:
                        r['user_id'] = user_id
                        significant_insights.append(r)
                else:
                    result['user_id'] = user_id
                    significant_insights.append(result)

        except Exception as e:
            logging.error(f"Error running analysis '{analysis_def['name']}' for user {user_id}: {e}", exc_info=True)
            continue
    
    return significant_insights

def main():
    """Main worker loop to poll for and process jobs."""
    logging.info("Starting OptiHealth Analysis Worker...")
    if not os.getenv("DATABASE_URL"):
        logging.error("DATABASE_URL environment variable not set. Exiting.")
        return

    while True:
        conn = get_raw_connection()
        if not conn:
            logging.error("Database connection failed. Retrying in 60 seconds...")
            time.sleep(60)
            continue
        
        job = None
        try:
            job = fetch_pending_job(conn)
            if job:
                logging.info(f"Picked up job {job['id']} for user {job['user_id']}")
                insights = run_insight_engine(job['user_id'], job['id'])
                if insights:
                    store_insights(conn, insights)
                else:
                    logging.info(f"No new significant insights found for user {job['user_id']}.")
                update_job_status(conn, job['id'], 'completed')
            else:
                logging.info(f"No pending jobs found. Sleeping for {POLL_INTERVAL_SECONDS} seconds.")
                time.sleep(POLL_INTERVAL_SECONDS)

        except Exception as e:
            logging.error(f"An unexpected error occurred while processing job {job['id'] if job else 'N/A'}: {e}", exc_info=True)
            if job:
                update_job_status(conn, job['id'], 'failed')
        finally:
            if conn:
                conn.close()

if __name__ == "__main__":
    main()
