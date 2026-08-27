import pandas as pd
import glob
import os

# 🔹 Step 1: Define path (relative to src/)
data_path = os.path.join("..", "data", "MachineLearningCVE", "*.csv")

# 🔹 Step 2: Get all CSV files
files = glob.glob(data_path)

print("📂 Files found:", len(files))

# ❌ If no files found → stop
if len(files) == 0:
    print("❌ ERROR: No CSV files found. Check your path!")
    exit()

# 🔹 Step 3: Read and sample data
df_list = []

for file in files:
    print(f"📥 Loading: {file}")
    
    try:
        # Read only 50k rows to avoid memory issues
        df = pd.read_csv(file, encoding='latin1', nrows=50000)
        df_list.append(df)
    except Exception as e:
        print(f"⚠️ Error reading {file}: {e}")

# ❌ If still empty → stop
if len(df_list) == 0:
    print("❌ ERROR: No data loaded!")
    exit()

# 🔹 Step 4: Combine all files
df = pd.concat(df_list, ignore_index=True)

print("📊 Combined shape:", df.shape)

# 🔹 Step 5: Save dataset
output_path = os.path.join("..", "data", "dataset_small.csv")
df.to_csv(output_path, index=False)

print(f"✅ Dataset saved at: {output_path}")