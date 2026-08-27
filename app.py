from flask import Flask, render_template, jsonify
import pandas as pd
import joblib
import time
from datetime import datetime

app = Flask(__name__)

# Load models
iso = joblib.load("models/iso.pkl")
xgb = joblib.load("models/xgb.pkl")

# Load data
df = pd.read_csv("data/processed.csv")
df.columns = df.columns.str.strip()

# Drop same features
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
X = df.drop("Label", axis=1)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/start")
def start_detection():
    results = []

    # limit for demo
    max_rows = 100

    for i in range(max_rows):
        sample = X.iloc[i:i+1]

        xgb_prob = xgb.predict_proba(sample)[0][1]
        iso_score = -iso.decision_function(sample)[0]
        iso_score = (iso_score + 1) / 2

        final_score = 0.6 * xgb_prob + 0.4 * iso_score

        if final_score > 0.8:
            status = "HIGH"
        elif final_score > 0.6:
            status = "MEDIUM"
        else:
            status = "NORMAL"

        results.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "score": round(final_score, 3),
            "status": status
        })

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)