import streamlit as st
import pandas as pd
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db, get_app
from streamlit_autorefresh import st_autorefresh
import altair as alt
import json
from firebase_admin import credentials
# --- CONFIGURATION ---
# Initialize Firebase only once
try:
    get_app()
except ValueError:


    

    firebase_cred_dict = dict(st.secrets["firebase"])
    cred = credentials.Certificate(firebase_cred_dict)
    if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://autonomous-hvac-default-rtdb.firebaseio.com'
    })
    

# --- STREAMLIT UI CONFIG ---
st.set_page_config(
    page_title="HVAC Dashboard",
    layout="wide"
)
st.title("📊 HVAC Real-Time Dashboard")

# Auto-refresh every 1 second
auto_refresh = st_autorefresh(interval=1000, key="datarefresh")

# --- DATA LOADERS ---

def load_yolo_depth():
    ref = db.reference('/yolo-depth')
    raw = ref.get() or {}
    rows = []
    for key, entry in raw.items():
        ts = entry.get("timestamp")
        try:
            ts_dt = datetime.fromtimestamp(float(ts))
        except Exception:
            ts_dt = datetime.fromisoformat(ts) if isinstance(ts, str) else None
        rows.append({
            'Record ID': key,
            'Timestamp': ts_dt,
            'People Count': entry.get('People count') or entry.get('count'),
            'Fake Count': entry.get('Fake count') or entry.get('fake_count')
        })
    return pd.DataFrame(rows).sort_values('Timestamp', ascending=False)


def load_people_counts():
    ref = db.reference('/people_counts')
    raw = ref.get() or {}
    rows = []
    for key, entry in raw.items():
        ts = entry.get("timestamp")
        try:
            ts_dt = datetime.fromtimestamp(float(ts))
        except Exception:
            ts_dt = datetime.fromisoformat(ts) if isinstance(ts, str) else None
        dets = entry.get('detections', {})
        rows.append({
            'Record ID': key,
            'Timestamp': ts_dt,
            'Count': entry.get('count'),
            'Average Confidence': entry.get('average_confidence'),
            'Detections': len(dets)
        })
    return pd.DataFrame(rows).sort_values('Timestamp', ascending=False)

# --- MAIN UI TABS ---
yolo_tab, people_tab = st.tabs(["YOLO Depth", "People Counts"])

with yolo_tab:
    st.subheader("YOLO Depth Stream")
    df_yolo = load_yolo_depth()
    if df_yolo.empty:
        st.write("No data under `/yolo-depth`.")
    else:
        st.dataframe(df_yolo, use_container_width=True, height=400)
        # Plot only points, no connecting lines
        melt1 = df_yolo.reset_index(drop=True).melt(
            id_vars=['Timestamp'],
            value_vars=['People Count', 'Fake Count'],
            var_name='Series', value_name='Value'
        )
        chart1 = alt.Chart(melt1).mark_point(size=60).encode(
            x='Timestamp:T',
            y='Value:Q',
            color='Series:N'
        ).properties(width='container', height=400)
        st.altair_chart(chart1, use_container_width=True)

with people_tab:
    st.subheader("People Counts Stream")
    df_pc = load_people_counts()
    if df_pc.empty:
        st.write("No data under `/people_counts`.")
    else:
        # Full table
        st.dataframe(df_pc, use_container_width=True, height=400)
        # Scatter plot for Count
        st.subheader("Count Over Time")
        chart_count = alt.Chart(df_pc).mark_point(size=60, color='orange').encode(
            x='Timestamp:T',
            y='Count:Q'
        ).properties(width='container', height=300)
        st.altair_chart(chart_count, use_container_width=True)

        # Separate scatter plot for Average Confidence
        st.subheader("Average Confidence Over Time")
        chart_conf = alt.Chart(df_pc).mark_point(size=60, color='green').encode(
            x='Timestamp:T',
            y='Average Confidence:Q'
        ).properties(width='container', height=300)
        st.altair_chart(chart_conf, use_container_width=True)

# Manual refresh
if st.button("🔄 Refresh Now"):
    st.experimental_rerun()
