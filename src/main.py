import os
import json

from config import DATASET_CONFIGS, NASA_COMMON_CONFIG, NASA_BATTERY_CONFIG_MAP
from loaders import load_and_standardize_file
from preprocess import preprocess_dataframe
from capacity import compute_capacity_profile
from validators import classify_event
from soh import compute_event_soh
from visualize import (
    plot_soh_trend,
    plot_soh_trend_all_batteries,
    plot_per_battery,
    plot_all_batteries_on_one_graph,
)


def ensure_directories():
    os.makedirs("../data/processed", exist_ok=True)
    os.makedirs("../results/summaries", exist_ok=True)
    os.makedirs("../results/plots", exist_ok=True)
    os.makedirs("../results/plots/voltage_per_battery", exist_ok=True)
    os.makedirs("../results/plots/current_per_battery", exist_ok=True)
    os.makedirs("../results/plots/temperature_per_battery", exist_ok=True)


def summarize_discharge_capacity_for_event(event_df):
    return float(event_df["incremental_discharge_capacity_ah"].sum())


def get_event_battery_config(event_df):
    """
    Select the correct config for an event based on battery_id.
    """
    if "battery_id" not in event_df.columns:
        raise ValueError("battery_id column not found in event data.")

    battery_id = str(event_df["battery_id"].iloc[0]).strip()

    if battery_id not in NASA_BATTERY_CONFIG_MAP:
        raise ValueError(
            f"battery_id '{battery_id}' not found in NASA_BATTERY_CONFIG_MAP."
        )

    config_name = NASA_BATTERY_CONFIG_MAP[battery_id]
    return battery_id, config_name, DATASET_CONFIGS[config_name]


def apply_monotonic_smoothing(discharge_results):
    """
    Enforce a non-increasing SOH trend per battery using NASA cycle_id order.
    """
    by_battery = {}

    for item in discharge_results:
        by_battery.setdefault(item["battery_id"], []).append(item)

    for battery_id, items in by_battery.items():
        items.sort(key=lambda x: x["cycle_id"])

        previous_smoothed = None

        for item in items:
            soh = item["soh_result"]

            raw = soh.get("soh_percent_raw")
            if raw is None:
                raw = soh.get("soh_percent")

            confidence = soh.get("confidence", 0.5)

            if raw is None:
                soh["soh_percent_smoothed"] = previous_smoothed
                soh["soh_percent"] = previous_smoothed
                continue

            if previous_smoothed is None:
                smoothed = raw
            else:
                weighted = confidence * raw + \
                    (1.0 - confidence) * previous_smoothed
                smoothed = min(previous_smoothed, weighted)

            soh["soh_percent_smoothed"] = float(smoothed)
            soh["soh_percent"] = float(smoothed)

            previous_smoothed = smoothed


def run_pipeline(file_path, preprocess_config):
    ensure_directories()

    print(f"\nLoading file: {file_path}")
    df, original_mapping = load_and_standardize_file(file_path)

    print("Detected column mapping:")
    for standard_name, original_name in original_mapping.items():
        print(f"  {original_name} -> {standard_name}")

    df, inferred_sign = preprocess_dataframe(df, preprocess_config)
    print(f"Inferred discharge sign convention: {inferred_sign}")

    df = compute_capacity_profile(df)

    # Plot voltage/current/temperature for each battery separately
    plot_per_battery(
        df,
        column="voltage_v",
        ylabel="Voltage (V)",
        title="Voltage vs Time",
        save_dir="../results/plots/voltage_per_battery"
    )

    plot_per_battery(
        df,
        column="current_a",
        ylabel="Current (A)",
        title="Current vs Time",
        save_dir="../results/plots/current_per_battery"
    )

    if "temperature_c" in df.columns:
        plot_per_battery(
            df,
            column="temperature_c",
            ylabel="Temperature (°C)",
            title="Temperature vs Time",
            save_dir="../results/plots/temperature_per_battery"
        )

    # Plot all batteries together on one graph
    plot_all_batteries_on_one_graph(
        df,
        column="voltage_v",
        ylabel="Voltage (V)",
        title="Voltage vs Time Across All Batteries",
        save_path="../results/plots/voltage_all_batteries.png"
    )

    plot_all_batteries_on_one_graph(
        df,
        column="current_a",
        ylabel="Current (A)",
        title="Current vs Time Across All Batteries",
        save_path="../results/plots/current_all_batteries.png"
    )

    if "temperature_c" in df.columns:
        plot_all_batteries_on_one_graph(
            df,
            column="temperature_c",
            ylabel="Temperature (°C)",
            title="Temperature vs Time Across All Batteries",
            save_path="../results/plots/temperature_all_batteries.png"
        )

    print("\nFirst 20 rows after preprocessing:")
    preview_cols = ["time_s", "current_a", "row_mode", "event_id"]

    extra_cols = []
    if "battery_id" in df.columns:
        extra_cols.append("battery_id")
    if "cycle_id" in df.columns:
        extra_cols.append("cycle_id")

    print(df[extra_cols + preview_cols].head(20))

    print("\nCycle sizes (first 20 battery_id + cycle_id groups):")
    print(df.groupby(["battery_id", "cycle_id"]).size().head(20))

    event_results = []

    # Use NASA battery_id + cycle_id as the main unit of analysis
    for (battery_id_key, cycle_id_key), cycle_df in df.groupby(["battery_id", "cycle_id"], sort=True):
        # Keep only the discharge portion of the cycle
        discharge_df = cycle_df[cycle_df["row_mode"] == "discharge"].copy()

        # Skip cycles with no discharge rows
        if discharge_df.empty:
            continue

        # Skip tiny noisy fragments
        if len(discharge_df) < 100:
            continue

        battery_id, config_name, event_config = get_event_battery_config(
            discharge_df)

        event_validation = classify_event(discharge_df, event_config)
        event_capacity_ah = summarize_discharge_capacity_for_event(
            discharge_df)

        soh_result = compute_event_soh(
            event_capacity_ah, event_validation, event_config
        )

        event_results.append({
            "event_id": int(discharge_df["event_id"].iloc[0]),
            "cycle_id": int(cycle_id_key),
            "battery_id": battery_id,
            "config_name": config_name,
            "event_validation": event_validation,
            "soh_result": soh_result
        })

    # Keep only discharge events
    discharge_results = [
        item for item in event_results
        if item["event_validation"]["row_mode"] == "discharge"
    ]

    # Sort by battery, then by original NASA cycle_id
    discharge_results = sorted(
        discharge_results,
        key=lambda x: (x["battery_id"], x["cycle_id"])
    )

    # Apply monotonic smoothing using cycle_id order
    apply_monotonic_smoothing(discharge_results)

    # Plot SOH for one battery
    plot_soh_trend(
        discharge_results,
        save_path="../results/plots/soh_trend_B0005.png",
        battery_id="B0005"
    )

    # Plot SOH for all batteries
    plot_soh_trend_all_batteries(
        discharge_results,
        save_path="../results/plots/soh_trend_all_batteries.png"
    )

    base_name = os.path.splitext(os.path.basename(file_path))[0]

    processed_csv_path = f"../data/processed/{base_name}_processed.csv"
    summary_json_path = f"../results/summaries/{base_name}_summary.json"

    df.to_csv(processed_csv_path, index=False)

    summary = {
        "file_name": file_path,
        "column_mapping": original_mapping,
        "inferred_discharge_sign": inferred_sign,
        "event_results": event_results,
        "discharge_cycle_results": discharge_results
    }

    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("\n=== MEANINGFUL DISCHARGE CYCLES ===")

    if not discharge_results:
        print("No meaningful discharge cycles found.")
    else:
        for item in discharge_results[:20]:
            ev = item["event_validation"]
            soh = item["soh_result"]

            print(
                f"\nBattery {item['battery_id']} | "
                f"NASA Cycle {item['cycle_id']} | "
                f"raw event_id: {item['event_id']} | "
                f"config: {item['config_name']}"
            )

            print(f"  event_type: {ev['event_type']}")
            print(f"  rows: {ev['num_rows']}")
            print(f"  duration_s: {ev['duration_s']:.2f}")
            print(f"  start_voltage_v: {ev['start_voltage_v']:.4f}")
            print(f"  end_voltage_v: {ev['end_voltage_v']:.4f}")
            print(f"  min_voltage_v: {ev['min_voltage_v']:.4f}")
            print(f"  max_voltage_v: {ev['max_voltage_v']:.4f}")
            print(f"  valid_for_direct_soh: {ev['valid_for_direct_soh']}")
            print(f"  event_capacity_ah: {soh['event_capacity_ah']:.6f}")

            if soh.get("corrected_capacity_ah") is not None:
                print(
                    f"  corrected_capacity_ah: {soh['corrected_capacity_ah']:.6f}")

            if soh.get("soc_start") is not None:
                print(f"  soc_start: {soh['soc_start']:.4f}")

            if soh.get("soc_end") is not None:
                print(f"  soc_end: {soh['soc_end']:.4f}")

            if soh.get("soc_window") is not None:
                print(f"  soc_window: {soh['soc_window']:.4f}")

            if soh.get("confidence") is not None:
                print(f"  confidence: {soh['confidence']:.3f}")

            if soh.get("soh_percent_raw") is not None:
                print(f"  soh_percent_raw: {soh['soh_percent_raw']}")

            if soh.get("soh_percent_smoothed") is not None:
                print(f"  soh_percent_smoothed: {soh['soh_percent_smoothed']}")

            print(f"  soh_percent: {soh['soh_percent']}")
            print(f"  soh_status: {soh['soh_status']}")
            print(
                f"  needs_soc_window_correction: {soh['needs_soc_window_correction']}")
            print(f"  notes: {soh['notes']}")

    print("\nSaved:")
    print(f"  Processed CSV: {processed_csv_path}")
    print(f"  Summary JSON:  {summary_json_path}")
    print("  Plot:         ../results/plots/soh_trend_B0005.png")
    print("  Plot:         ../results/plots/soh_trend_all_batteries.png")
    print("  Plot:         ../results/plots/voltage_all_batteries.png")
    print("  Plot:         ../results/plots/current_all_batteries.png")
    if "temperature_c" in df.columns:
        print("  Plot:         ../results/plots/temperature_all_batteries.png")


if __name__ == "__main__":
    input_file = "..\\data\\raw\\NASA_Combined.csv"

    # Common preprocessing config
    preprocess_config = NASA_COMMON_CONFIG

    run_pipeline(input_file, preprocess_config)
