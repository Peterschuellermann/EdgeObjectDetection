import sqlite3
import pandas as pd
import os
from datetime import datetime
from .utils import parse_filename_datetime

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
            geometry_wkt TEXT NOT NULL,
            latitude REAL,
            longitude REAL,
            ais_matched INTEGER DEFAULT 0,
            ais_mmsi TEXT,
            ais_vessel_name TEXT,
            ais_vessel_type TEXT,
            ais_distance_m REAL
        )
    ''')
    
    # Migrate existing tables to add AIS columns if they don't exist
    migrate_db_schema(cursor)
    
    conn.commit()
    conn.close()


def migrate_db_schema(cursor):
    """Add AIS columns to existing detections table if they don't exist."""
    # Check if columns exist by trying to select them
    cursor.execute("PRAGMA table_info(detections)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if 'latitude' not in columns:
        cursor.execute('ALTER TABLE detections ADD COLUMN latitude REAL')
    if 'longitude' not in columns:
        cursor.execute('ALTER TABLE detections ADD COLUMN longitude REAL')
    if 'ais_matched' not in columns:
        cursor.execute('ALTER TABLE detections ADD COLUMN ais_matched INTEGER DEFAULT 0')
    if 'ais_mmsi' not in columns:
        cursor.execute('ALTER TABLE detections ADD COLUMN ais_mmsi TEXT')
    if 'ais_vessel_name' not in columns:
        cursor.execute('ALTER TABLE detections ADD COLUMN ais_vessel_name TEXT')
    if 'ais_vessel_type' not in columns:
        cursor.execute('ALTER TABLE detections ADD COLUMN ais_vessel_type TEXT')
    if 'ais_distance_m' not in columns:
        cursor.execute('ALTER TABLE detections ADD COLUMN ais_distance_m REAL')

def insert_detection(run_id, file_name, label, score, geometry_wkt, 
                     latitude=None, longitude=None, ais_matched=False,
                     ais_mmsi=None, ais_vessel_name=None, ais_vessel_type=None,
                     ais_distance_m=None):
    """Insert a single detection record."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO detections (
            run_id, timestamp, file_name, label, score, geometry_wkt,
            latitude, longitude, ais_matched, ais_mmsi, ais_vessel_name,
            ais_vessel_type, ais_distance_m
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        run_id, datetime.now(), file_name, label, score, geometry_wkt,
        latitude, longitude, 1 if ais_matched else 0,
        ais_mmsi, ais_vessel_name, ais_vessel_type, ais_distance_m
    ))
    
    conn.commit()
    conn.close()


def update_detection_ais(id, ais_matched, ais_mmsi=None, ais_vessel_name=None,
                         ais_vessel_type=None, ais_distance_m=None):
    """Update AIS match information for a detection."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE detections
        SET ais_matched = ?, ais_mmsi = ?, ais_vessel_name = ?,
            ais_vessel_type = ?, ais_distance_m = ?
        WHERE id = ?
    ''', (1 if ais_matched else 0, ais_mmsi, ais_vessel_name,
          ais_vessel_type, ais_distance_m, id))
    
    conn.commit()
    conn.close()


def get_detections_by_run_id(run_id):
    """Get all detections for a specific run_id."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM detections WHERE run_id = ?
        ORDER BY timestamp DESC
    ''', (run_id,))
    
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_detections_df():
    """Retrieve all detections as a pandas DataFrame."""
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM detections", conn)
        return df
    finally:
        conn.close()

def get_unique_days():
    """
    Extract unique days from file names in the database.
    
    Returns:
        list: Sorted list of unique day strings in YYYY-MM-DD format
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT file_name FROM detections")
        rows = cursor.fetchall()
        
        days = set()
        for row in rows:
            filename = row[0]
            start_dt, end_dt = parse_filename_datetime(filename)
            if start_dt is not None:
                day_str = start_dt.strftime('%Y-%m-%d')
                days.add(day_str)
        
        # Return sorted list
        return sorted(list(days))
    finally:
        conn.close()
