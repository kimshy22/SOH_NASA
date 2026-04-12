NASA_COMMON_CONFIG = {
    "reference_capacity_ah": 2.0,
    "nominal_voltage_v": 3.7,
    "expected_full_charge_v": 4.2,
    "full_start_voltage_threshold_v": 4.0,
    "min_discharge_duration_s": 1500,
    "min_voltage_drop_v": 0.50,
    "max_time_gap_s": 60,
    "default_discharge_sign": "negative",
    "rest_current_threshold_a": 0.02,
    "min_event_rows": 5,
    "min_event_duration_s": 10.0,
}

DATASET_CONFIGS = {
    "nasa_b0005": {
        **NASA_COMMON_CONFIG,
        "expected_cutoff_v": 2.7,
        "partial_or_full_cutoff_threshold_v": 2.75,
    },
    "nasa_b0006": {
        **NASA_COMMON_CONFIG,
        "expected_cutoff_v": 2.5,
        "partial_or_full_cutoff_threshold_v": 2.55,
    },
    "nasa_b0007": {
        **NASA_COMMON_CONFIG,
        "expected_cutoff_v": 2.2,
        "partial_or_full_cutoff_threshold_v": 2.25,
    },
    "nasa_b0018": {
        **NASA_COMMON_CONFIG,
        "expected_cutoff_v": 2.5,
        "partial_or_full_cutoff_threshold_v": 2.55,
    },
}

NASA_BATTERY_CONFIG_MAP = {
    "B0005": "nasa_b0005",
    "B0006": "nasa_b0006",
    "B0007": "nasa_b0007",
    "B0018": "nasa_b0018",
}
