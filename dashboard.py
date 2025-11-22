import streamlit as st
import pandas as pd
import plotly.express as px
import os
from src.database import get_detections_df, init_db, get_unique_days
from src.visualization import get_map_files
from src.config import MAPS_OUTPUT_DIR
from src.utils import parse_filename_datetime

# Page Config
st.set_page_config(
    page_title="Object Detection Dashboard",
    layout="wide"
)

st.title("Edge Object Detection Analytics")

# Initialize DB (safe to call multiple times)
init_db()

# Load Data
if st.button("Refresh Data"):
    st.rerun()

try:
    df = get_detections_df()
except Exception as e:
    st.error(f"Error connecting to database: {e}")
    st.stop()

if df.empty:
    st.info("No detection data found. Run main.py to generate results.")
    st.stop()

# Sidebar Filters
st.sidebar.header("Filters")

# Run ID Filter
all_runs = df['run_id'].unique()
selected_runs = st.sidebar.multiselect("Select Run ID", all_runs, default=all_runs)

if selected_runs:
    df = df[df['run_id'].isin(selected_runs)]

# Label Filter
all_labels = df['label'].unique()
selected_labels = st.sidebar.multiselect("Select Labels", all_labels, default=all_labels)

if selected_labels:
    df = df[df['label'].isin(selected_labels)]

# Score Filter
min_score, max_score = float(df['score'].min()), float(df['score'].max())
score_range = st.sidebar.slider("Confidence Score", min_score, max_score, (min_score, max_score))
df = df[(df['score'] >= score_range[0]) & (df['score'] <= score_range[1])]

# Metrics
st.subheader("Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Total Detections", len(df))
c2.metric("Unique Images", df['file_name'].nunique())
c3.metric("Classes Detected", df['label'].nunique())

# Charts
c1, c2 = st.columns(2)

with c1:
    st.subheader("Class Distribution")
    if not df.empty:
        fig_bar = px.bar(
            df['label'].value_counts().reset_index(),
            x='label',
            y='count',
            labels={'label': 'Object Class', 'count': 'Count'},
            title="Detections by Class"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

with c2:
    st.subheader("Confidence Score Distribution")
    if not df.empty:
        fig_hist = px.histogram(
            df,
            x='score',
            nbins=20,
            title="Confidence Score Histogram",
            labels={'score': 'Confidence Score'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# Day-Specific Maps Section
st.subheader("Day-Specific Maps")
try:
    # Get unique days from database
    unique_days = get_unique_days()
    
    if not unique_days:
        st.info("No days found in the database. Run main.py to generate detection data.")
    else:
        # Get available map files
        map_files = get_map_files()
        
        if not map_files:
            st.warning("No day-specific maps found. Run main.py to generate maps.")
        else:
            # Create dropdown selector
            available_days = [day for day in unique_days if day in map_files]
            
            if not available_days:
                st.warning("No maps available for the days in the database. Run main.py to generate maps.")
            else:
                selected_day = st.selectbox(
                    "Select Day",
                    options=available_days,
                    index=0 if available_days else None
                )
                
                if selected_day and selected_day in map_files:
                    map_file_path = map_files[selected_day]
                    
                    # Check if file exists
                    if os.path.exists(map_file_path):
                        # Read and display the HTML map
                        with open(map_file_path, 'r', encoding='utf-8') as f:
                            map_html = f.read()
                        
                        st.components.v1.html(map_html, height=600, scrolling=True)
                        
                        # Optionally filter detection table for this day
                        if st.checkbox("Filter detection table to show only this day", value=False):
                            # Filter dataframe by day
                            day_filenames = set()
                            for filename in df['file_name'].unique():
                                start_dt, end_dt = parse_filename_datetime(filename)
                                if start_dt is not None:
                                    day_str = start_dt.strftime('%Y-%m-%d')
                                    if day_str == selected_day:
                                        day_filenames.add(filename)
                            
                            if day_filenames:
                                df_filtered = df[df['file_name'].isin(day_filenames)]
                                st.subheader(f"Detections for {selected_day}")
                                st.dataframe(
                                    df_filtered[['timestamp', 'run_id', 'file_name', 'label', 'score', 'geometry_wkt']].sort_values(by='timestamp', ascending=False),
                                    use_container_width=True
                                )
                            else:
                                st.info(f"No detections found for {selected_day}")
                    else:
                        st.error(f"Map file not found: {map_file_path}")
                else:
                    st.info("Please select a day to view its map.")
except Exception as e:
    st.error(f"Error loading day-specific maps: {e}")
    import traceback
    st.code(traceback.format_exc())

# Data Table
st.subheader("Detailed Results")
st.dataframe(
    df[['timestamp', 'run_id', 'file_name', 'label', 'score', 'geometry_wkt']].sort_values(by='timestamp', ascending=False),
    use_container_width=True
)
