import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime

# 🔹 Load models
iso = joblib.load("../models/iso.pkl")
xgb = joblib.load("../models/xgb.pkl")

# 🔹 Load dataset
data_path = "../data/processed.csv"
df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()

# 🔹 Drop same features used in training
drop_cols = [
    'Flow Bytes/s',
    'Flow Packets/s',
    'Packet Length Variance',
    'Avg Packet Size',
    'Fwd Packet Length Std',
    'Bwd Packet Length Std',
    'Init_Win_bytes_forward'
]

df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

# 🔹 Separate features
X_full = df.drop("Label", axis=1)

# 🔹 Baseline stats for drift detection
train_mean = X_full.mean()

# 🔥 DEMO SETTINGS
max_rows = 300
sleep_time = 0.05

alerts = {"high": 0, "medium": 0, "normal": 0}

print("🚀 Starting Advanced Real-Time Detection...\n")

for i in range(max_rows):

    sample = X_full.iloc[i:i+1]

    # =========================
    # ⚡ LATENCY TRACKING
    # =========================
    start = time.time()

    xgb_prob = xgb.predict_proba(sample)[0][1]
    iso_score = -iso.decision_function(sample)[0]
    iso_score = (iso_score + 1) / 2

    final_score = 0.6 * xgb_prob + 0.4 * iso_score

    end = time.time()
    latency = (end - start) * 1000

    # =========================
    # 📊 DRIFT DETECTION
    # =========================
    current_mean = sample.mean()
    drift = np.abs(train_mean - current_mean).mean()

    drift_flag = "⚠️ DRIFT" if drift > 0.5 else "OK"

    # =========================
    # 🚨 ALERT LOGIC
    # =========================
    timestamp = datetime.now().strftime("%H:%M:%S")

    if final_score > 0.8:
        status = "HIGH"
        alerts["high"] += 1
    elif final_score > 0.6:
        status = "MEDIUM"
        alerts["medium"] += 1
    else:
        status = "NORMAL"
        alerts["normal"] += 1

    # =========================
    # 🔁 FEEDBACK LOGGING
    # =========================
    log_entry = {
        "time": timestamp,
        "score": float(final_score),
        "status": status,
        "latency": latency,
        "drift": float(drift)
    }

    with open("logs.txt", "a") as f:
        f.write(str(log_entry) + "\n")

    # =========================
    # 🖥️ OUTPUT
    # =========================
    print(f"[{i}] [{timestamp}] Score: {final_score:.3f} | {status} | ⚡ {latency:.2f}ms | Drift: {drift_flag}")

    time.sleep(sleep_time)

print("\n📊 FINAL ALERT SUMMARY:", alerts)