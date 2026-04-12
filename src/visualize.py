# src/visualize.py

import os
import matplotlib.pyplot as plt


def plot_voltage_vs_time(df, save_path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(df["time_s"], df["voltage_v"])
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("Voltage vs Time")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()


def plot_current_vs_time(df, save_path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(df["time_s"], df["current_a"])
    plt.xlabel("Time (s)")
    plt.ylabel("Current (A)")
    plt.title("Current vs Time")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()


def plot_capacity_vs_time(df, save_path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(df["time_s"], df["cumulative_capacity_ah"])
    plt.xlabel("Time (s)")
    plt.ylabel("Cumulative Discharged Capacity (Ah)")
    plt.title("Cumulative Capacity vs Time")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()


def plot_temperature_vs_time(df, save_path=None):
    if "temperature_c" not in df.columns:
        return

    plt.figure(figsize=(8, 4))
    plt.plot(df["time_s"], df["temperature_c"])
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (°C)")
    plt.title("Temperature vs Time")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()


def plot_soh_trend(discharge_results, save_path=None, battery_id=None):
    """
    Plot SOH trend using detected cycle number.

    Parameters:
        discharge_results: list of event result dictionaries from main.py
        save_path: optional output image path
        battery_id: optional filter for one battery only, e.g. "B0005"
    """
    cycles = []
    soh_raw = []
    soh_smoothed = []

    for item in discharge_results:
        item_battery_id = item.get("battery_id")

        if battery_id is not None and item_battery_id != battery_id:
            continue

        cycle_number = item.get("detected_cycle_number")
        soh_result = item.get("soh_result", {})

        raw_value = soh_result.get("soh_percent_raw")
        smoothed_value = soh_result.get("soh_percent_smoothed")

        if cycle_number is None:
            continue

        cycles.append(cycle_number)
        soh_raw.append(raw_value)
        soh_smoothed.append(smoothed_value)

    if not cycles:
        print("No SOH data available for plotting.")
        return

    plt.figure(figsize=(9, 4))
    plt.plot(cycles, soh_raw, marker="o", label="Raw SOH")
    plt.plot(cycles, soh_smoothed, marker="o", label="Smoothed SOH")

    plt.xlabel("Detected Cycle Number")
    plt.ylabel("SOH (%)")

    if battery_id is None:
        plt.title("SOH Trend")
    else:
        plt.title(f"SOH Trend - {battery_id}")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.close()
