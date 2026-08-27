import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# 🔹 Step 1: Load dataset
input_path = os.path.join("..", "data", "dataset_balanced.csv")

print("📥 Loading dataset...")
df = pd.read_csv(input_path)

# 🔹 Step 2: Clean column names
df.columns = df.columns.str.strip()

print("📊 Shape:", df.shape)

# 🔹 Step 3: Remove unwanted columns (if exist)
drop_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

# 🔹 Step 4: Handle non-numeric values
print("🔧 Converting to numeric...")

for col in df.columns:
    if col != 'Label':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 🔹 Step 5: Handle NaN / Infinite
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print("📊 Shape after cleaning:", df.shape)

# 🔹 Step 6: Split features and target
X = df.drop('Label', axis=1)
y = df['Label']

# 🔹 Step 7: Feature scaling
print("⚙️ Scaling features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 Step 8: Save processed data
processed_path = os.path.join("..", "data", "processed.csv")

processed_df = pd.DataFrame(X_scaled, columns=X.columns)
processed_df['Label'] = y.values

processed_df.to_csv(processed_path, index=False)

print(f"✅ Processed dataset saved at: {processed_path}")
print("📊 Final shape:", processed_df.shape)