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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
    plt.close()


def plot_per_battery(df, column, ylabel, title, save_dir=None):
    """
    Plot one variable vs time for each battery separately.
    Example columns: voltage_v, current_a, temperature_c
    """
    if "battery_id" not in df.columns:
        print("battery_id column not found in dataframe.")
        return

    if column not in df.columns:
        print(f"{column} column not found in dataframe.")
        return

    battery_ids = df["battery_id"].dropna().unique()

    for battery_id in battery_ids:
        sub_df = df[df["battery_id"] == battery_id].copy()

        if sub_df.empty:
            continue

        plt.figure(figsize=(9, 4))
        plt.plot(sub_df["time_s"], sub_df[column])

        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)
        plt.title(f"{title} - {battery_id}")
        plt.grid(True)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{battery_id}_{column}.png")
            plt.savefig(save_path, dpi=200)

        plt.close()


def plot_all_batteries_on_one_graph(df, column, ylabel, title, save_path=None):
    """
    Plot one variable vs time for all battery IDs on the same figure.
    """
    if "battery_id" not in df.columns:
        print("battery_id column not found in dataframe.")
        return

    if column not in df.columns:
        print(f"{column} column not found in dataframe.")
        return

    battery_ids = df["battery_id"].dropna().unique()

    plt.figure(figsize=(10, 5))

    for battery_id in battery_ids:
        sub_df = df[df["battery_id"] == battery_id].copy()

        if sub_df.empty:
            continue

        plt.plot(sub_df["time_s"], sub_df[column], label=battery_id)

    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.close()


def plot_soh_trend(discharge_results, save_path=None, battery_id=None):
    """
    Plot SOH trend for one battery or all discharge results using NASA cycle_id.
    """
    cycles = []
    soh_raw = []
    soh_smoothed = []

    for item in discharge_results:
        item_battery_id = item.get("battery_id")

        if battery_id is not None and item_battery_id != battery_id:
            continue

        cycle_number = item.get("cycle_id")
        soh_result = item.get("soh_result", {})

        raw_value = soh_result.get("soh_percent_raw")
        if raw_value is None:
            raw_value = soh_result.get("soh_percent")

        smoothed_value = soh_result.get("soh_percent_smoothed")
        if smoothed_value is None:
            smoothed_value = soh_result.get("soh_percent")

        if cycle_number is None:
            continue

        if raw_value is None and smoothed_value is None:
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

    plt.xlabel("NASA Cycle ID")
    plt.ylabel("SOH (%)")

    if battery_id is None:
        plt.title("SOH Trend")
    else:
        plt.title(f"SOH Trend - {battery_id}")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.close()


def plot_soh_trend_all_batteries(discharge_results, save_path=None):
    """
    Plot final SOH trend for all battery IDs on one graph using NASA cycle_id.
    """
    by_battery = {}

    for item in discharge_results:
        battery_id = item["battery_id"]
        by_battery.setdefault(battery_id, []).append(item)

    plt.figure(figsize=(10, 5))

    plotted_any = False

    for battery_id, items in by_battery.items():
        items.sort(key=lambda x: x["cycle_id"])

        cycles = []
        soh_values = []

        for item in items:
            cycle = item.get("cycle_id")
            soh_result = item.get("soh_result", {})
            soh = soh_result.get("soh_percent_smoothed")

            if soh is None:
                soh = soh_result.get("soh_percent")

            if cycle is None or soh is None:
                continue

            cycles.append(cycle)
            soh_values.append(soh)

        if cycles:
            plt.plot(cycles, soh_values, marker="o", label=battery_id)
            plotted_any = True

    if not plotted_any:
        print("No SOH data available for plotting.")
        plt.close()
        return

    plt.xlabel("NASA Cycle ID")
    plt.ylabel("SOH (%)")
    plt.title("SOH Trend Across All Batteries")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.close()
