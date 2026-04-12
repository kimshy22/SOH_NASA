# src/validators.py

import numpy as np


def summarize_event(event_df):
    summary = {
        "event_id": int(event_df["event_id"].iloc[0]),
        "row_mode": str(event_df["row_mode"].iloc[0]),
        "start_time_s": float(event_df["time_s"].iloc[0]),
        "end_time_s": float(event_df["time_s"].iloc[-1]),
        "duration_s": float(event_df["time_s"].iloc[-1] - event_df["time_s"].iloc[0]),
        "start_voltage_v": float(event_df["voltage_v"].iloc[0]),
        "end_voltage_v": float(event_df["voltage_v"].iloc[-1]),
        "min_voltage_v": float(event_df["voltage_v"].min()),
        "max_voltage_v": float(event_df["voltage_v"].max()),
        "avg_discharge_current_a": float(event_df["discharge_current_a"].mean()),
        "avg_charge_current_a": float(event_df["charge_current_a"].mean()),
        "num_rows": int(len(event_df)),
    }

    if "temperature_c" in event_df.columns:
        summary["avg_temperature_c"] = float(event_df["temperature_c"].mean())

    return summary


def classify_event(event_df, config):
    summary = summarize_event(event_df)

    result = {
        **summary,
        "event_type": "unknown",
        "valid_for_direct_soh": False,
        "reasons": []
    }

    mode = summary["row_mode"]

    if mode == "rest":
        result["event_type"] = "rest"
        result["reasons"].append("Rest event.")
        return result

    if mode == "charge":
        result["event_type"] = "charge"
        result["reasons"].append("Charge event.")
        return result

    if mode == "discharge":
        voltage_drop = summary["start_voltage_v"] - summary["end_voltage_v"]
        result["voltage_drop_v"] = float(voltage_drop)

        near_full_start = summary["start_voltage_v"] >= config["full_start_voltage_threshold_v"]
        near_cutoff_end = summary["end_voltage_v"] <= config["partial_or_full_cutoff_threshold_v"]

        if summary["duration_s"] < config["min_discharge_duration_s"]:
            result["event_type"] = "partial_discharge"
            result["reasons"].append("Discharge event too short.")
            return result

        if voltage_drop <= config["min_voltage_drop_v"]:
            result["event_type"] = "partial_discharge"
            result["reasons"].append("Voltage drop too small.")
            return result

        if near_full_start and near_cutoff_end:
            result["event_type"] = "full_discharge"
            result["valid_for_direct_soh"] = True
            result["reasons"].append("Looks like a full discharge event.")
        else:
            result["event_type"] = "partial_discharge"
            if not near_full_start:
                result["reasons"].append(
                    "Did not start near expected full-charge voltage.")
            if not near_cutoff_end:
                result["reasons"].append(
                    "Did not end near expected cutoff voltage.")

        return result

    result["event_type"] = "unknown"
    result["reasons"].append("Unrecognized event.")
    return result
