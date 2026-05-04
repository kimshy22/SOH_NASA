import pandas as pd
import os

# =========================
# PATH SETUP
# =========================

# Get current script directory (src)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Move to project root (SOH_NASA)
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Input file path
input_file = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "B0007_cycle_level.csv"
)

# Output file path
output_file = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "processed_B0007_cycle_level.csv"
)

# =========================
# SETTINGS
# =========================

# NASA reference capacity
reference_capacity_ah = 2.0


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(input_file)

print("Loaded file:")
print(input_file)
print("\nColumns found:")
print(df.columns.tolist())
print("-" * 50)


# =========================
# ADD REFERENCE CAPACITY
# =========================

df["reference_capacity_ah"] = reference_capacity_ah


# =========================
# COMPUTE NEW FEATURES
# =========================

# Capacity ratio
df["capacity_ratio"] = df["corrected_capacity_ah"] / df["reference_capacity_ah"]

# Current C-rate (absolute value)
df["current_c_rate"] = df["avg_discharge_current_a"].abs() / \
    df["reference_capacity_ah"]


# =========================
# SANITY CHECK
# =========================

print("\nSample results:")
print(df[[
    "corrected_capacity_ah",
    "reference_capacity_ah",
    "capacity_ratio",
    "avg_discharge_current_a",
    "current_c_rate"
]].head())


# =========================
# SAVE OUTPUT
# =========================

df.to_csv(output_file, index=False)

print("\n✅ Saved new file:")
print(output_file)
