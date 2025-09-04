import yaml
import time
import importlib
import logging
from database import get_db_connection, fetch_pending_job, update_job_status, store_insights, fetch_user_goals
from data_loader import load_and_prepare_data

POLL_INTERVAL_SECONDS = 5

def load_analysis_config():
    """Loads the analysis definitions from the YAML file."""
    with open('analysis_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def run_insight_engine(conn, job):
    """Orchestrates the analysis process for a single job."""
    user_id = job['user_id']
    logging.info(f"Starting insight engine for job {job['id']}, user {user_id}")

    daily_summary_df, events_df = load_and_prepare_data(conn, user_id)
    if daily_summary_df.empty:
        logging.warning(f"No data to analyze for user {user_id}. Skipping.")
        return
    
    # Fetch user goals at the start of the engine run
    user_goals = fetch_user_goals(conn, user_id)

    config = load_analysis_config()
    significant_insights = []

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

            result = None
            if module_name in ['correlation', 'clustering', 'feature_importance', 'forecasting']:
                result = analysis_function(daily_summary_df, params_with_name)
            elif module_name == 'comparative':
                result = analysis_function(daily_summary_df, events_df, params_with_name)
            elif module_name == 'goals':
                # The goals module gets the user_goals as an extra argument
                result = analysis_function(daily_summary_df, params_with_name, user_goals)
            else:
                logging.warning(f"Analysis module '{module_name}' not recognized.")

            if result:
                if isinstance(result, list): # If the module returns a list of insights
                    for res in result:
                        res['user_id'] = user_id
                    significant_insights.extend(result)
                else: # If it returns a single insight dictionary
                    result['user_id'] = user_id
                significant_insights.append(result)

        except Exception as e:
            logging.error(f"Error running analysis '{analysis_def['name']}' for user {user_id}: {e}", exc_info=True)
            continue
    
    if significant_insights:
        store_insights(conn, significant_insights)
    else:
        logging.info(f"No new significant insights found for user {user_id}.j")

def main():
    """Main worker loop to poll for and process jobs."""
    logging.info("Starting OptiHealth Analysis Worker...")
    while True:
        conn = get_db_connection()
        if not conn:
            time.sleep(60)
            continue
        
        job = None
        try:
            job = fetch_pending_job(conn)
            if job:
                logging.info(f"Picked up job {job['id']} for user {job['user_id']}")
                run_insight_engine(conn, job)
                update_job_status(conn, job['id'], 'completed')
            else:
                logging.info(f"No pending jobs found. Sleeping for {POLL_INTERVAL_SECONDS} seconds.")
                time.sleep(POLL_INTERVAL_SECONDS)
        except Exception as e:
            logging.error(f"An unexpected error occurred while processing job {job['id'] if job else 'N/A'}: {e}")
            if job:
                update_job_status(conn, job['id'], 'failed')
        finally:
            if conn:
                conn.close()

if __name__ == "__main__":
    main()
