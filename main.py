import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, db, get_app
from streamlit_autorefresh import st_autorefresh
import altair as alt

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
        ts_dt = None
        if isinstance(ts, str):
            try:
                ts_dt = datetime.fromisoformat(ts)
            except ValueError:
                pass
        else:
            try:
                ts_dt = datetime.fromtimestamp(float(ts))
            except Exception:
                pass
        # Correct for the one-hour offset
        if ts_dt:
            ts_dt -= timedelta(hours=1)
        rows.append({
            'Record ID': key,
            'Timestamp': ts_dt,
            'People Count': entry.get('People count') or entry.get('count'),
            'Fake Count': entry.get('Fake count') or entry.get('fake_count')
        })
    df = pd.DataFrame(rows)
    if 'Timestamp' in df.columns:
        df = df.sort_values('Timestamp', ascending=False)
    return df


def load_people_counts():
    ref = db.reference('/people_counts')
    raw = ref.get() or {}
    rows = []
    for key, entry in raw.items():
        ts = entry.get("timestamp")
        ts_dt = None
        if isinstance(ts, str):
            try:
                ts_dt = datetime.fromisoformat(ts)
            except ValueError:
                pass
        else:
            try:
                ts_dt = datetime.fromtimestamp(float(ts))
            except Exception:
                pass
        dets = entry.get('detections', {})
        rows.append({
            'Record ID': key,
            'Timestamp': ts_dt,
            'Count': entry.get('count'),
            'Average Confidence': entry.get('average_confidence'),
            'Detections': len(dets)
        })
    df = pd.DataFrame(rows)
    if 'Timestamp' in df.columns:
        df = df.sort_values('Timestamp', ascending=False)
    return df


def load_pico_count():
    ref = db.reference('/people_inside')
    raw = ref.get() or {}
    rows = []
    for key, entry in raw.items():
        ts = entry.get("timestamp")
        ts_dt = None
        if isinstance(ts, str):
            try:
                ts_dt = datetime.fromisoformat(ts)
            except ValueError:
                pass
        else:
            try:
                ts_dt = datetime.fromtimestamp(float(ts))
            except Exception:
                pass
        rows.append({
            'Record ID': key,
            'Timestamp': ts_dt,
            'Count': entry.get('count')
        })
    df = pd.DataFrame(rows)
    if 'Timestamp' in df.columns:
        df = df.sort_values('Timestamp', ascending=False)
    return df

# --- MAIN UI TABS ---
yolo_tab, people_tab, pico_tab = st.tabs(["YOLO Depth", "YOLO RGB", "Pico Hypersonic sensors"])

with yolo_tab:
    st.subheader("YOLO Depth Stream")
    df_yolo = load_yolo_depth()
    if df_yolo.empty:
        st.write("No data under `/yolo-depth`.")
    else:
        st.dataframe(df_yolo, use_container_width=True, height=400)
        # Combined People and Fake counts
        st.subheader("People vs Fake Count Over Time")
        melt1 = df_yolo.reset_index(drop=True).melt(
            id_vars=['Timestamp'],
            value_vars=['People Count', 'Fake Count'],
            var_name='Series', value_name='Value'
        )
        # Determine dynamic Y scale
        max_val = int(melt1['Value'].max() or 0) + 1
        chart1 = alt.Chart(melt1).mark_point(size=60).encode(
            x='Timestamp:T',
            y=alt.Y('Value:Q', scale=alt.Scale(domain=[0, max_val]), axis=alt.Axis(format='d')),
            color=alt.Color('Series:N', scale=alt.Scale(
                domain=['People Count', 'Fake Count'],
                range=['#00FF00', 'red']
            ))
        ).properties(width='container', height=400)
        st.altair_chart(chart1, use_container_width=True)

        # Separate Fake Count chart
        st.subheader("Fake Count Over Time")
        max_fake = int(df_yolo['Fake Count'].max() or 0) + 1
        chart_fake = alt.Chart(df_yolo).mark_point(size=60, color='red').encode(
            x='Timestamp:T',
            y=alt.Y('Fake Count:Q', scale=alt.Scale(domain=[0, max_fake]), axis=alt.Axis(format='d'))
        ).properties(width='container', height=300)
        st.altair_chart(chart_fake, use_container_width=True)

with people_tab:
    st.subheader("People Counts Stream")
    df_pc = load_people_counts()
    if df_pc.empty:
        st.write("No data under `/people_counts`.")
    else:
        st.dataframe(df_pc, use_container_width=True, height=400)
        st.subheader("Count Over Time")
        max_cnt = int(df_pc['Count'].max() or 0) + 1
        chart_count = alt.Chart(df_pc).mark_point(size=60, color='orange').encode(
            x='Timestamp:T',
            y=alt.Y('Count:Q', scale=alt.Scale(domain=[0, max_cnt]), axis=alt.Axis(format='d'))
        ).properties(width='container', height=300)
        st.altair_chart(chart_count, use_container_width=True)

        st.subheader("Average Confidence Over Time")
        chart_conf = alt.Chart(df_pc).mark_point(size=60, color='green').encode(
            x='Timestamp:T',
            y='Average Confidence:Q'
        ).properties(width='container', height=300)
        st.altair_chart(chart_conf, use_container_width=True)

with pico_tab:
    st.subheader("Pico Count Stream")
    df_pico = load_pico_count()
    if df_pico.empty:
        st.write("No data under `/people_inside`.")
    else:
        st.dataframe(df_pico, use_container_width=True, height=400)
        st.subheader("Count Over Time")
        max_pico = int(df_pico['Count'].max() or 0) + 1
        chart_pico = alt.Chart(df_pico).mark_point(size=60, color="green").encode(
            x='Timestamp:T',
            y=alt.Y('Count:Q', scale=alt.Scale(domain=[0, max_pico]), axis=alt.Axis(format='d'))
        ).properties(width='container', height=300)
        st.altair_chart(chart_pico, use_container_width=True)

# Manual refresh
if st.button("🔄 Refresh Now"):
    st.experimental_rerun()
