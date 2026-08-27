import streamlit as st
import pandas as pd
import joblib
import time
from datetime import datetime

st.set_page_config(page_title="APT Detection", layout="wide")

st.markdown("""
<style>
body { background-color: black; }
h1 { color: #00ffcc; text-shadow: 0 0 10px #00ffcc; }
</style>
""", unsafe_allow_html=True)

st.title("ADVANCED PERSISTENT THREATS (APT) DETECTION SYSTEM")

# 🔹 Load models
iso = joblib.load("models/iso.pkl")
xgb = joblib.load("models/xgb.pkl")

# 🔹 Load dataset
base_df = pd.read_csv("data/processed.csv")
base_df.columns = base_df.columns.str.strip()

drop_cols = [
    'Flow Bytes/s',
    'Flow Packets/s',
    'Packet Length Variance',
    'Avg Packet Size',
    'Fwd Packet Length Std',
    'Bwd Packet Length Std',
    'Init_Win_bytes_forward'
]

base_df = base_df.drop(columns=[col for col in drop_cols if col in base_df.columns], errors='ignore')

# 🔥 SELECT TYPE
sample_type = st.selectbox(
    "📊 Select Traffic Type",
    ["Mixed Traffic", "Normal Only", "Attack Heavy"]
)

def generate_sample(df, mode):
    if mode == "Normal Only":
        return df[df["Label"] == 0].sample(100)
    elif mode == "Attack Heavy":
        attack = df[df["Label"] == 1].sample(80)
        normal = df[df["Label"] == 0].sample(20)
        return pd.concat([attack, normal])
    else:
        return df.sample(100)

start = st.button("▶ Generate & Start Monitoring")

# UI placeholders
table = st.empty()
chart = st.empty()

col1, col2, col3 = st.columns(3)
lat_col = st.columns(1)[0]

if start:

    df = generate_sample(base_df, sample_type)
    X = df.drop("Label", axis=1)

    high = medium = normal = 0
    results = []

    chart_data = pd.DataFrame({
        "High": [],
        "Medium": [],
        "Normal": []
    })

    for i in range(len(X)):

        sample = X.iloc[i:i+1]

        # =========================
        # ⚡ LATENCY TRACKING
        # =========================
        start_time = time.time()

        xgb_prob = xgb.predict_proba(sample)[0][1]
        iso_score = -iso.decision_function(sample)[0]
        iso_score = (iso_score + 1) / 2

        final_score = 0.6 * xgb_prob + 0.4 * iso_score

        end_time = time.time()
        latency = (end_time - start_time) * 1000  # ms

        # =========================
        # 🚨 ALERT LOGIC
        # =========================
        if final_score > 0.8:
            status = "🔴 HIGH"
            high += 1
        elif final_score > 0.6:
            status = "🟠 MEDIUM"
            medium += 1
        else:
            status = "🟢 NORMAL"
            normal += 1

        timestamp = datetime.now().strftime("%H:%M:%S")

        results.append({
            "Time": timestamp,
            "Score": round(final_score, 3),
            "Latency(ms)": round(latency, 2),
            "Status": status
        })

        # =========================
        # 🔁 FEEDBACK LOGGING (UTF-8 FIX)
        # =========================
        log_entry = {
            "time": timestamp,
            "score": float(final_score),
            "status": status,
            "latency": float(latency)
        }

        with open("logs.txt", "a", encoding="utf-8") as f:
            f.write(str(log_entry) + "\n")

        # =========================
        # 🖥️ UI UPDATES
        # =========================
        table.dataframe(pd.DataFrame(results), use_container_width=True)

        col1.metric("🔴 High", high)
        col2.metric("🟠 Medium", medium)
        col3.metric("🟢 Normal", normal)

        lat_col.metric("⚡ Latency (ms)", f"{latency:.2f}")

        new_row = pd.DataFrame({
            "High": [high],
            "Medium": [medium],
            "Normal": [normal]
        })

        chart_data = pd.concat([chart_data, new_row], ignore_index=True)
        chart.line_chart(chart_data)

        time.sleep(0.03)

    st.success("Monitoring Complete!")

    st.subheader("Final Summary")

    total = high + medium + normal

    st.write(f"Total Records: {total}")
    st.write(f"High Risk: {high}")
    st.write(f"Medium Risk: {medium}")
    st.write(f"Normal: {normal}")