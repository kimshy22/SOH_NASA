import pandas as pd

# ==============================
# 1. Load your dataset
# ==============================

input_path = r"C:\Users\SUZZIE\SOH_NASA\data\raw\B0007.csv"
output_path = r"C:\Users\SUZZIE\SOH_NASA\data\raw\B0007_20cycles.csv"

df = pd.read_csv(input_path)

print("Original shape:", df.shape)

# ==============================
# 2. Check columns
# ==============================

print("Columns:", df.columns)

# ==============================
# 3. Filter first 20 cycles
# ==============================

df_20 = df[df["Cycle"] <= 20].copy()

print("Filtered shape:", df_20.shape)

# ==============================
# 4. Verify cycles
# ==============================

print("Unique cycles:", sorted(df_20["Cycle"].unique()))

# ==============================
# 5. Save new dataset
# ==============================

df_20.to_csv(output_path, index=False)

print(f"Saved to: {output_path}")
