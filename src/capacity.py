# src/capacity.py

import pandas as pd


# src/capacity.py

def compute_capacity_profile(df):
    df = df.copy()

    df["incremental_discharge_capacity_ah"] = (
        df["discharge_current_a"] * df["dt_s"] / 3600.0
    ).fillna(0).clip(lower=0)

    df["incremental_charge_capacity_ah"] = (
        df["charge_current_a"] * df["dt_s"] / 3600.0
    ).fillna(0).clip(lower=0)

    df["cumulative_discharge_capacity_ah"] = df["incremental_discharge_capacity_ah"].cumsum()
    df["cumulative_charge_capacity_ah"] = df["incremental_charge_capacity_ah"].cumsum()

    return df
