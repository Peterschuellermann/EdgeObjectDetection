import streamlit as st
import pandas as pd
import plotly.express as px
from src.database import get_detections_df, init_db

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

# Data Table
st.subheader("Detailed Results")
st.dataframe(
    df[['timestamp', 'run_id', 'file_name', 'label', 'score', 'geometry_wkt']].sort_values(by='timestamp', ascending=False),
    use_container_width=True
)
