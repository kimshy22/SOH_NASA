import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["time_s", "voltage_v", "current_a"]


def check_required_columns(df):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def drop_missing_required_rows(df):
    return df.dropna(subset=REQUIRED_COLUMNS).copy()


def infer_discharge_sign(df):
    current = df["current_a"].dropna()
    negative_fraction = (current < 0).mean()
    positive_fraction = (current > 0).mean()

    if negative_fraction >= positive_fraction:
        return "negative"
    return "positive"


def add_current_magnitude_columns(df, discharge_sign="negative"):
    df = df.copy()

    if discharge_sign == "negative":
        df["discharge_current_a"] = (-df["current_a"]).clip(lower=0)
        df["charge_current_a"] = df["current_a"].clip(lower=0)
    elif discharge_sign == "positive":
        df["discharge_current_a"] = df["current_a"].clip(lower=0)
        df["charge_current_a"] = (-df["current_a"]).clip(lower=0)
    else:
        raise ValueError("discharge_sign must be 'negative' or 'positive'.")

    return df


def classify_row_mode(df, rest_current_threshold_a=0.02):
    df = df.copy()

    discharge_mask = df["discharge_current_a"] > rest_current_threshold_a
    charge_mask = df["charge_current_a"] > rest_current_threshold_a

    df["row_mode"] = np.where(
        discharge_mask,
        "discharge",
        np.where(charge_mask, "charge", "rest")
    )

    return df


def compute_dt_preserve_order(df, group_cols):
    """
    Compute dt_s within each group while preserving the original row order.
    """
    df = df.copy()
    df["_original_order"] = np.arange(len(df))

    # Work on a sorted copy only for dt calculation
    sorted_df = df.sort_values(group_cols + ["time_s"]).copy()

    sorted_df["dt_s"] = sorted_df.groupby(
        group_cols)["time_s"].diff().fillna(0)

    # Prevent negative dt inside groups
    sorted_df.loc[sorted_df["dt_s"] < 0, "dt_s"] = np.nan

    # Bring dt_s back to original row order
    df = df.merge(
        sorted_df[["_original_order", "dt_s"]],
        on="_original_order",
        how="left"
    )

    df = df.sort_values("_original_order").drop(columns=["_original_order"])
    return df


def assign_event_ids_preserve_order(df, group_cols):
    """
    Assign event IDs within each group, while preserving original row order.
    """
    df = df.copy()
    df["_original_order"] = np.arange(len(df))

    event_ids = np.zeros(len(df), dtype=int)
    next_event_id = 1

    for _, group_index in df.groupby(group_cols, sort=False).groups.items():
        group = df.loc[group_index].copy()

        mode_change = group["row_mode"] != group["row_mode"].shift(1)
        local_event_ids = mode_change.cumsum()

        unique_local_ids = local_event_ids.unique()
        local_to_global = {
            local_id: next_event_id + i
            for i, local_id in enumerate(unique_local_ids)
        }

        mapped_ids = local_event_ids.map(local_to_global)
        event_ids[group.index] = mapped_ids

        next_event_id += len(unique_local_ids)

    df["event_id"] = event_ids
    df = df.sort_values("_original_order").drop(columns=["_original_order"])

    return df


def preprocess_dataframe(df, config):
    check_required_columns(df)
    df = drop_missing_required_rows(df)

    inferred_sign = infer_discharge_sign(df)

    if config.get("default_discharge_sign") is not None:
        discharge_sign = config["default_discharge_sign"]
    else:
        discharge_sign = inferred_sign

    df = add_current_magnitude_columns(df, discharge_sign=discharge_sign)

    df = classify_row_mode(
        df,
        rest_current_threshold_a=config["rest_current_threshold_a"]
    )

    group_cols = []
    if "battery_id" in df.columns:
        group_cols.append("battery_id")
    if "cycle_id" in df.columns:
        group_cols.append("cycle_id")

    if not group_cols:
        raise ValueError(
            "No grouping columns found. Expected battery_id and/or cycle_id."
        )

    # Compute dt without changing row order
    df = compute_dt_preserve_order(df, group_cols)

    # Assign event ids without changing row order
    df = assign_event_ids_preserve_order(df, group_cols)

    return df, inferred_sign
