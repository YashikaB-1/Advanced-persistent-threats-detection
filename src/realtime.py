import pandas as pd
import numpy as np
import os
import joblib
import time

# 🔹 Load models
iso = joblib.load("../models/iso.pkl")
xgb = joblib.load("../models/xgb.pkl")

# 🔹 Load data
df = pd.read_csv("../data/processed.csv")
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
X = df.drop("Label", axis=1)

print("🚀 Starting Real-Time Detection...\n")

# 🔁 Simulate streaming
for i in range(len(X)):

    sample = X.iloc[i:i+1]

    # XGBoost probability
    xgb_prob = xgb.predict_proba(sample)[0][1]

    # Isolation score
    iso_score = -iso.decision_function(sample)[0]

    # Normalize (simple scaling)
    iso_score = (iso_score + 1) / 2  

    # Hybrid score
    final_score = 0.6 * xgb_prob + 0.4 * iso_score

    # Alert logic
    if final_score > 0.8:
        status = "🔴 HIGH RISK ATTACK"
    elif final_score > 0.6:
        status = "🟠 MEDIUM RISK"
    else:
        status = "🟢 NORMAL"

    print(f"[{i}] Score: {final_score:.3f} → {status}")

    # Simulate real-time delay
    time.sleep(0.3)