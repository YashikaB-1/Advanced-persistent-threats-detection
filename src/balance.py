import pandas as pd
import os

# Load dataset
input_path = os.path.join("..", "data", "dataset_small.csv")

print("📥 Loading dataset...")
df = pd.read_csv(input_path)

# Clean column names
df.columns = df.columns.str.strip()

print("\n🔍 Original label distribution:")
print(df['Label'].value_counts())

# 🔥 STEP 1: Convert to binary labels
df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

print("\n🔍 After conversion (0=Normal, 1=Attack):")
print(df['Label'].value_counts())

# 🔥 STEP 2: Split classes
normal = df[df['Label'] == 0]
attack = df[df['Label'] == 1]

# 🔥 STEP 3: Balance
n = min(len(normal), len(attack))

print(f"\n⚖️ Balancing to {n} samples each")

normal_sample = normal.sample(n=n, random_state=42)
attack_sample = attack.sample(n=n, random_state=42)

df_balanced = pd.concat([normal_sample, attack_sample])

# Shuffle
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
output_path = os.path.join("..", "data", "dataset_balanced.csv")
df_balanced.to_csv(output_path, index=False)

print("\n✅ AFTER BALANCING:")
print(df_balanced['Label'].value_counts())

print("\n📊 Final shape:", df_balanced.shape)
print(f"\n💾 Saved at: {output_path}")