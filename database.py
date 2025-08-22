import os
import psycopg2
import psycopg2.extras
import pandas as pd
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    """Establishes a connection to the database."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except psycopg2.OperationalError as e:
        logging.error(f"Could not connect to the database: {e}")
        return None

def fetch_pending_job(conn):
    """Fetches a single pending job and marks it as in_progress."""
    job = None
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        try:
            cur.execute("""
                UPDATE analysis_jobs
                SET status = 'in_progress', updated_at = now()
                WHERE id = (
                    SELECT id
                    FROM analysis_jobs
                    WHERE status = 'pending'
                    ORDER BY created_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                RETURNING id, user_id;
            """)
            job = cur.fetchone()
            conn.commit()
        except Exception as e:
            logging.error(f"Error fetching pending job: {e}")
            conn.rollback()
    return job

def fetch_user_data_points(conn, user_id, days=90):
    """Fetches the last N days of data_points for a user."""
    query = """
        SELECT
            dp.timestamp,
            md.metric_name,
            dp.value_numeric
        FROM public.data_points dp
        JOIN public.metric_definitions md ON dp.metric_id = md.id
        WHERE dp.user_id = %(user_id)s
          AND dp.timestamp >= now() - interval '%(days)s days'
          AND dp.value_numeric IS NOT NULL;
    """
    params = {'user_id': user_id, 'days': days}
    try:
        df = pd.read_sql(query, conn, params=params)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        logging.error(f"Error fetching data points for user {user_id}: {e}")
        return pd.DataFrame()

def fetch_user_events(conn, user_id, days=90):
    """Fetches the last N days of events for a user."""
    query = """
        SELECT
            event_name,
            start_timestamp,
            end_timestamp,
            properties
        FROM public.events
        WHERE user_id = %(user_id)s
          AND start_timestamp >= now() - interval '%(days)s days';
    """
    params = {'user_id': user_id, 'days': days}
    try:
        df = pd.read_sql(query, conn, params=params)
        df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])
        return df
    except Exception as e:
        logging.error(f"Error fetching events for user {user_id}: {e}")
        return pd.DataFrame()

def update_job_status(conn, job_id, status):
    """Updates the status of a job (completed or failed)."""
    with conn.cursor() as cur:
        try:
            cur.execute("""
                UPDATE analysis_jobs
                SET status = %(status)s, updated_at = now()
                WHERE id = %(job_id)s;
            """, {'job_id': job_id, 'status': status})
            conn.commit()
            logging.info(f"Updated job {job_id} to status '{status}'.")
        except Exception as e:
            logging.error(f"Error updating job {job_id} status: {e}")
            conn.rollback()

def store_insights(conn, insights_data):
    """Stores a batch of generated insights into the database."""
    if not insights_data:
        return
    with conn.cursor() as cur:
        try:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO public.insights (user_id, insight_type, title, summary, result_data)
                VALUES %s;
                """,
                [(
                    d['user_id'],
                    d['type'],
                    d['title'],
                    d['summary'],
                    psycopg2.extras.Json(d['evidence'])
                ) for d in insights_data]
            )
            conn.commit()
            logging.info(f"Successfully stored {len(insights_data)} insights.")
        except Exception as e:
            logging.error(f"Error storing insights: {e}")
            conn.rollback()
            
def fetch_user_goals(conn, user_id):
    """Fetches all active goals for a given user."""
    query = """
        SELECT
            ug.id as goal_id,
            md.metric_name,
            ug.target_value,
            ug.operator,
            ug.goal_name
        FROM public.user_goals ug
        JOIN public.metric_definitions md ON ug.metric_id = md.id
        WHERE ug.user_id = %(user_id)s
          AND ug.is_active = true;
    """
    params = {'user_id': user_id}
    try:
        df = pd.read_sql(query, conn, params=params)
        return df.to_dict('records')
    except Exception as e:
        logging.error(f"Error fetching goals for user {user_id}: {e}")
        return []