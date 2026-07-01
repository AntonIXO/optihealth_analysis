import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logging
import json
from datetime import datetime, timedelta, timezone

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Database Connection Setup with SQLAlchemy ---
db_engine = None

def get_db_engine():
    """Creates and returns a SQLAlchemy engine, reusing it if it exists."""
    global db_engine
    if db_engine is None:
        try:
            load_dotenv()
            database_url = os.getenv("DATABASE_URL")
            if not database_url:
                logging.error("DATABASE_URL not set in environment variables.")
                return None
            db_engine = create_engine(database_url)
            logging.info("Database engine created successfully.")
        except Exception as e:
            logging.error(f"Failed to create database engine: {e}")
            return None
    return db_engine

# --- Job Management Functions ---
def fetch_pending_job():
    """Fetches a single pending job and marks it as in_progress."""
    engine = get_db_engine()
    if not engine:
        return None
        
    job = None
    update_query = text("""
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
    try:
        with engine.connect() as conn:
            with conn.begin(): # Start a transaction
                result = conn.execute(update_query).fetchone()
            if result:
                job = result._asdict() # Convert to dict-like object
    except Exception as e:
        logging.error(f"Error fetching pending job: {e}")
    return job

def update_job_status(job_id, status):
    """Updates the status of a job (completed or failed)."""
    engine = get_db_engine()
    if not engine:
        return

    query = text("""
        UPDATE analysis_jobs
        SET status = :status, updated_at = now()
        WHERE id = :job_id;
    """)
    try:
        with engine.connect() as conn:
            with conn.begin():
                conn.execute(query, {'status': status, 'job_id': job_id})
        logging.info(f"Updated job {job_id} to status '{status}'.")
    except Exception as e:
        logging.error(f"Error updating job {job_id} status: {e}")

def store_insights(insights_data):
    """Stores a batch of generated insights into the database."""
    if not insights_data:
        return

    engine = get_db_engine()
    if not engine:
        return

    # The table has UNIQUE(user_id, insight_type, generated_at). A single batch
    # now legitimately contains MANY insights of the same type (e.g. dozens of
    # discovered correlations), so a constant NOW() would collapse them via
    # ON CONFLICT. Assign each row a distinct generated_at (microsecond-stepped
    # from a single base) so every distinct insight is persisted, while repeated
    # identical batches across worker runs still differ in time (as before).
    insert_stmt = text("""
        INSERT INTO public.insights (user_id, insight_type, title, summary, result_data, generated_at)
        VALUES (:user_id, :insight_type, :title, :summary, :result_data, :generated_at)
        ON CONFLICT (user_id, insight_type, generated_at) DO NOTHING;
    """)

    base = datetime.now(timezone.utc)
    formatted_data = [
        {
            'user_id': d['user_id'],
            'insight_type': d['type'],
            'title': d['title'],
            'summary': d['summary'],
            'result_data': json.dumps(d['evidence']),
            'generated_at': base + timedelta(microseconds=i),
        } for i, d in enumerate(insights_data)
    ]

    try:
        with engine.connect() as conn:
            with conn.begin():
                conn.execute(insert_stmt, formatted_data)
        logging.info(f"Successfully stored {len(insights_data)} insights.")
    except Exception as e:
        logging.error(f"Error storing insights: {e}")

# --- Data Fetching Functions ---
def fetch_user_data_points(user_id, days=365):
    """Fetches the last N days of data_points for a user."""
    engine = get_db_engine()
    if not engine:
        return pd.DataFrame()
        
    query = text("""
        SELECT
            dp.timestamp,
            md.metric_name,
            dp.value_numeric
        FROM public.data_points dp
        JOIN public.metric_definitions md ON dp.metric_id = md.id
        WHERE dp.user_id = :user_id
          AND dp.timestamp >= NOW() - MAKE_INTERVAL(days => :days)
          AND dp.value_numeric IS NOT NULL;
    """)
    params = {'user_id': str(user_id), 'days': days}
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params=params)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        logging.error(f"Error fetching data points for user {user_id}: {e}")
        return pd.DataFrame()

def fetch_user_events(user_id, days=365):
    """Fetches the last N days of events for a user."""
    engine = get_db_engine()
    if not engine:
        return pd.DataFrame()
        
    query = text("""
        SELECT
            event_name,
            start_timestamp,
            end_timestamp,
            properties
        FROM public.events
        WHERE user_id = :user_id
          AND start_timestamp >= NOW() - MAKE_INTERVAL(days => :days);
    """)
    params = {'user_id': str(user_id), 'days': days}
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params=params)
        df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])
        return df
    except Exception as e:
        logging.error(f"Error fetching events for user {user_id}: {e}")
        return pd.DataFrame()

def fetch_user_goals(user_id):
    """Fetches all active goals for a user."""
    engine = get_db_engine()
    if not engine:
        return pd.DataFrame()
        
    query = text("""
        SELECT 
            metric_name, 
            target_value, 
            comparison_operator 
        FROM public.user_goals
        WHERE user_id = :user_id AND is_active = true;
    """)
    params = {'user_id': str(user_id)}
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params=params)
        return df
    except Exception as e:
        # Corrected a typo here from user_goid to user_id
        logging.error(f"Error fetching goals for user {user_id}: {e}")
        return pd.DataFrame()

def fetch_user_supplement_component_logs(user_id, days=365):
    """
    Fetches the total daily dosage per supplement SUBSTANCE for a user.

    Reads the analysis-optimized view public.analysis_daily_supplement_intake
    (created by the 20260630160000 migration), which rolls every supplement_log
    up to the substance grain (e.g. Magnesium Citrate + Magnesium L-Threonate ->
    "Magnesium") and sums calculated_dosage_mg per user/day/substance. This
    replaces the old 3-table join against supplement_products /
    product_component_link / supplement_components, which no longer exist after
    the web app migrated to the "Chapter 15" ontology (that join raised
    UndefinedTable on every run).

    The return contract (columns: date, component_name, total_daily_amount) is
    unchanged, so data_loader.load_and_prepare_data and analysis/supplements.py
    need no changes.
    """
    engine = get_db_engine()
    if not engine:
        return pd.DataFrame()

    query = text("""
        SELECT
            day              AS date,
            substance_name   AS component_name,
            total_dosage_mg  AS total_daily_amount
        FROM public.analysis_daily_supplement_intake
        WHERE user_id = :user_id
          AND day >= (NOW() - MAKE_INTERVAL(days => :days))::date
        ORDER BY
            day,
            substance_name;
    """)
    params = {'user_id': str(user_id), 'days': days}
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params=params)
        # The date is already a DATE in the view; ensure it's the correct type.
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        logging.error(f"Error fetching supplement substance logs for user {user_id}: {e}")
        return pd.DataFrame()
