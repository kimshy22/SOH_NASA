from pathlib import Path
import json
import pandas as pd


# Change this to your actual summaries folder
SUMMARY_DIR = Path("estimator/results/summaries")

rows = []

for json_file in SUMMARY_DIR.glob("*.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Case 1: summary JSON is a list of event/cycle results
    if isinstance(data, list):
        records = data

    # Case 2: summary JSON has results inside a key
    elif isinstance(data, dict):
        if "cycle_results" in data:
            records = data["cycle_results"]
        elif "event_results" in data:
            records = data["event_results"]
        elif "results" in data:
            records = data["results"]
        else:
            records = [data]
    else:
        continue

    for r in records:
        event_type = r.get("event_type")

        if event_type == "partial_discharge":
            rows.append({
                "source_file": json_file.name,
                "battery_id": r.get("battery_id"),
                "cycle_id": r.get("cycle_id"),
                "event_id": r.get("event_id"),
                "event_type": event_type,
                "event_capacity_ah": r.get("event_capacity_ah"),
                "soc_start": r.get("soc_start"),
                "soc_end": r.get("soc_end"),
                "soc_window": r.get("soc_window"),
                "estimated_full_capacity_ah": r.get("estimated_full_capacity_ah"),
                "corrected_capacity_ah": r.get("corrected_capacity_ah"),
                "soh_percent": r.get("soh_percent"),
                "soh_status": r.get("soh_status"),
                "needs_soc_window_correction": r.get("needs_soc_window_correction"),
                "notes": " | ".join(r.get("notes", [])) if isinstance(r.get("notes"), list) else r.get("notes")
            })


if not rows:
    print("No partial discharge events found in the JSON summaries.")
else:
    df = pd.DataFrame(rows)

    print("\nPARTIAL DISCHARGE EVENTS FOUND")
    print("=" * 80)
    print(df.to_string(index=False))

    output_path = SUMMARY_DIR / "partial_discharge_inspection.csv"
    df.to_csv(output_path, index=False)

    print("\nSaved inspection report to:")
    print(output_path)

    print("\nStatus counts:")
    print(df["soh_status"].value_counts(dropna=False))
