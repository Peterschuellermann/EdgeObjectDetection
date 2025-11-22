import sqlite3
import pandas as pd
import os
from datetime import datetime

DB_PATH = "detections.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database with the detections table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            file_name TEXT NOT NULL,
            label TEXT NOT NULL,
            score REAL NOT NULL,
            geometry_wkt TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()

def insert_detection(run_id, file_name, label, score, geometry_wkt):
    """Insert a single detection record."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO detections (run_id, timestamp, file_name, label, score, geometry_wkt)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (run_id, datetime.now(), file_name, label, score, geometry_wkt))
    
    conn.commit()
    conn.close()

def get_detections_df():
    """Retrieve all detections as a pandas DataFrame."""
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM detections", conn)
        return df
    finally:
        conn.close()
