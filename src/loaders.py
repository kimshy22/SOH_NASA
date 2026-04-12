# src/loaders.py

import os
import pandas as pd


COLUMN_CANDIDATES = {
    "time_s": [
        "time_s", "Time_s", "time", "Time", "Test_Time(s)", "Test Time (s)",
        "Time(s)", "time (s)", "Test_Time", "total_time_s"
    ],
    "voltage_v": [
        "voltage_v", "Voltage_V", "Voltage(V)", "Voltage", "V", "voltage"
    ],
    "current_a": [
        "current_a", "Current_A", "Current(A)", "Current", "I", "current"
    ],
    "temperature_c": [
        "temperature_c", "Temperature_C", "Surface_Temp(degC)", "Temperature (C)_1", "Temp_C",
        "Temperature", "temperature", "Cell_Temp_C", "Surface Temperature"
    ],
    "cycle_id": [
        "cycle_id", "Cycle", "Cycle_Index", "Cycle_Index_", "cycle"
    ],
    "step_id": [
        "step_id", "Step", "Step_Index", "Step_Index_", "step"
    ],
    "battery_id": [
        "battery_id", "BatteryID", "battery", "Battery", "cell_id", "CellID"
    ]  # type: ignore
}


def find_matching_column(df_columns, candidates):
    """
    Return the first matching column from candidates that exists in df_columns.
    """
    for candidate in candidates:
        if candidate in df_columns:
            return candidate
    return None


def load_raw_file(file_path):
    """
    Load CSV or Excel file depending on extension.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(file_path)

    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)

    else:
        raise ValueError(
            f"Unsupported file type: {ext}. Please use .csv, .xlsx, or .xls"
        )


def load_and_standardize_file(file_path):
    """
    Load a file and rename recognized columns into a common schema.

    Returns:
        df: standardized DataFrame
        mapping: dict showing standardized_name -> original_name
    """
    df = load_raw_file(file_path)

    rename_map = {}
    reverse_mapping = {}

    for standard_name, candidates in COLUMN_CANDIDATES.items():
        match = find_matching_column(df.columns, candidates)
        if match is not None:
            rename_map[match] = standard_name
            reverse_mapping[standard_name] = match

    df = df.rename(columns=rename_map)

    return df, reverse_mapping
