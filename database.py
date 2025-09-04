import os
import psycopg2
import psycopg2.extras
import pandas as pd
from dotenv import load_dotenv
import logging
from sqlalchemy import create_engine, text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# --- SQLAlchemy Engine ---
# Create a single, global SQLAlchemy engine. This manages a pool of connections.
try:
    db_engine = create_engine(DATABASE_URL)
    logging.info("Database engine created successfully.")
except Exception as e:
    logging.error(f"Failed to create database engine: {e}")
    db_engine = None

def get_raw_connection():
    """Gets a single raw DBAPI connection from the engine's pool."""
    if not db_engine:
        logging.error("Database engine is not available.")
        return None
    try:
        return db_engine.raw_connection()
    except Exception as e:
        logging.error(f"Could not get a raw connection from the engine: {e}")
        return None

# --- Pandas Functions (Now use the engine directly) ---

def fetch_metric_definitions():
    """Fetches all metric definitions and returns a name -> id mapping."""
    try:
        df = pd.read_sql("SELECT id, metric_name FROM public.metric_definitions", db_engine)
        return pd.Series(df.id.values, index=df.metric_name).to_dict()
    except Exception as e:
        logging.error(f"Error fetching metric definitions: {e}")
        return {}

def fetch_user_data_points(user_id, days=90):
    """Fetches the last N days of data_points for a user."""
    query = text("""
        SELECT
            dp.timestamp,
            md.metric_name,
            dp.value_numeric
        FROM public.data_points dp
        JOIN public.metric_definitions md ON dp.metric_id = md.id
        WHERE dp.user_id = :user_id
          AND dp.timestamp >= now() - interval ':days days'
          AND dp.value_numeric IS NOT NULL;
    """)
    params = {'user_id': user_id, 'days': days}
    try:
        with db_engine.connect() as connection:
             df = pd.read_sql(query, connection, params=params)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        logging.error(f"Error fetching data points for user {user_id}: {e}")
        return pd.DataFrame()

def fetch_user_events(user_id, days=90):
    """Fetches the last N days of events for a user."""
    query = text("""
        SELECT
            event_name,
            start_timestamp,
            end_timestamp,
            properties
        FROM public.events
        WHERE user_id = :user_id
          AND start_timestamp >= now() - interval ':days days';
    """)
    params = {'user_id': user_id, 'days': days}
    try:
        with db_engine.connect() as connection:
            df = pd.read_sql(query, connection, params=params)
        df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])
        return df
    except Exception as e:
        logging.error(f"Error fetching events for user {user_id}: {e}")
        return pd.DataFrame()

def fetch_user_goals(user_id):
    """Fetches all active goals for a given user."""
    query = text("""
        SELECT
            g.goal_name,
            md.metric_name,
            g.target_value,
            g.operator
        FROM public.user_goals g
        JOIN public.metric_definitions md ON g.metric_id = md.id
        WHERE g.user_id = :user_id AND g.is_active = TRUE;
    """)
    params = {'user_id': user_id}
    try:
        with db_engine.connect() as connection:
            df = pd.read_sql(query, connection, params=params)
        return df
    except Exception as e:
        logging.error(f"Error fetching goals for user {user_id}: {e}")
        return pd.DataFrame()

# --- Raw Psycopg2 Functions (For transactions and updates) ---

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
                VALUES %s
                ON CONFLICT (user_id, insight_type, generated_at) DO NOTHING;
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
