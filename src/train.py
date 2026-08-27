import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from xgboost import XGBClassifier

# 🔹 Step 1: Load processed data
input_path = os.path.join("..", "data", "processed.csv")

print("📥 Loading processed dataset...")
df = pd.read_csv(input_path)

# 🔹 Step 2: Clean column names
df.columns = df.columns.str.strip()

# 🔥 Step 3: Remove leakage / overpowered features
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

# 🔹 Step 4: Split features and target
X = df.drop("Label", axis=1)
y = df["Label"]

print("📊 Dataset shape:", X.shape)

# 🔥 Step 5: Time-based split (NO leakage)
split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

print(f"📊 Train size: {X_train.shape}")
print(f"📊 Test size: {X_test.shape}")

# 🔥 Step 6: Add noise to reduce overfitting
noise = np.random.normal(0, 0.01, X_train.shape)
X_train = X_train + noise

# =========================================================
# 🧠 PART 1: HYBRID MODEL (MAIN MODEL)
# =========================================================

print("\n🧠 Training Hybrid Model (Isolation Forest + XGBoost)...")

# Isolation Forest
iso = IsolationForest(
    contamination=0.05,
    random_state=42
)

iso.fit(X_train[y_train == 0])  # train only on normal data

# XGBoost
xgb = XGBClassifier(
    max_depth=3,
    n_estimators=50,
    learning_rate=0.1,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=2,
    reg_alpha=1,
    eval_metric='logloss'
)

xgb.fit(X_train, y_train)

# 🔹 Hybrid Prediction
xgb_probs = xgb.predict_proba(X_test)[:, 1]
iso_scores = -iso.decision_function(X_test)

# Normalize iso scores
iso_scores = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())

# Final hybrid score
final_score = 0.6 * xgb_probs + 0.4 * iso_scores

# Threshold
y_pred_hybrid = (final_score > 0.6).astype(int)

print("\n📊 HYBRID MODEL PERFORMANCE:\n")
print(classification_report(y_test, y_pred_hybrid))

# =========================================================
# 📊 BUSINESS METRICS (VERY IMPORTANT)
# =========================================================

cm = confusion_matrix(y_test, y_pred_hybrid)
tn, fp, fn, tp = cm.ravel()

detection_rate = tp / (tp + fn)
false_positive_rate = fp / (fp + tn)

print("\n📊 BUSINESS METRICS:")
print(f"Detection Rate (Attack Recall): {detection_rate:.3f}")
print(f"False Positive Rate: {false_positive_rate:.3f}")

# =========================================================
# 🔍 PART 2: MODEL COMPARISON
# =========================================================

print("\n🔍 ===== MODEL COMPARISON =====")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=50),
    "XGBoost (Standalone)": XGBClassifier(eval_metric='logloss')
}

for name, model in models.items():
    print(f"\n🔹 Training {name}...")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n📊 {name} Performance:")
    print(classification_report(y_test, y_pred))

# =========================================================
# 🏆 FINAL ANALYSIS (FOR VIVA)
# =========================================================

print("\n🏆 FINAL MODEL ANALYSIS:")
print("Hybrid model combines anomaly detection (Isolation Forest) + supervised learning (XGBoost).")
print("This improves detection of unknown attacks while maintaining classification accuracy.")

# =========================================================
# 💾 SAVE MODELS
# =========================================================

os.makedirs("../models", exist_ok=True)

joblib.dump(iso, "../models/iso.pkl")
joblib.dump(xgb, "../models/xgb.pkl")

print("\n✅ Models saved in /models/")