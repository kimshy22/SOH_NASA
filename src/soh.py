# src/soh.py

def clamp(value, min_value=0.0, max_value=1.0):
    """
    Restrict a numeric value to a given range.
    """
    return max(min_value, min(value, max_value))


def estimate_soc_from_voltage(voltage_v, config):
    """
    Simple linear SOC estimate from voltage.

    SOC = (V - V_cutoff) / (V_full - V_cutoff)

    Returns a value between 0 and 1.
    """
    v_full = config["expected_full_charge_v"]
    v_cutoff = config["expected_cutoff_v"]

    if v_full <= v_cutoff:
        raise ValueError(
            f"Invalid config: expected_full_charge_v ({v_full}) "
            f"must be greater than expected_cutoff_v ({v_cutoff})."
        )

    soc = (voltage_v - v_cutoff) / (v_full - v_cutoff)
    return clamp(soc, 0.0, 1.0)


def compute_soh_from_full_discharge(event_capacity_ah, reference_capacity_ah):
    """
    Compute direct SOH from a full-discharge capacity measurement.
    """
    if reference_capacity_ah is None or reference_capacity_ah <= 0:
        raise ValueError("Reference capacity must be positive.")

    return (event_capacity_ah / reference_capacity_ah) * 100.0


def build_full_discharge_soh_result(event_capacity_ah, event_validation, config):
    """
    Build SOH result for a full discharge event.
    """
    soh_percent = compute_soh_from_full_discharge(
        event_capacity_ah,
        config["reference_capacity_ah"]
    )

    return {
        "event_id": event_validation["event_id"],
        "event_type": event_validation["event_type"],
        "valid_for_direct_soh": True,
        "event_capacity_ah": float(event_capacity_ah),
        "reference_capacity_ah": float(config["reference_capacity_ah"]),
        "estimated_full_capacity_ah": float(event_capacity_ah),
        "corrected_capacity_ah": float(event_capacity_ah),
        "soc_start": None,
        "soc_end": None,
        "soc_window": None,
        "soh_percent": float(soh_percent),
        "soh_status": "direct_full_discharge",
        "needs_soc_window_correction": False,
        "notes": list(event_validation["reasons"])
    }


def build_partial_discharge_result(event_capacity_ah, event_validation, config):
    """
    Build SOH result for a partial discharge event using SOC-window correction.
    """
    start_voltage_v = event_validation["start_voltage_v"]
    end_voltage_v = event_validation["end_voltage_v"]
    reference_capacity_ah = config["reference_capacity_ah"]

    soc_start = estimate_soc_from_voltage(start_voltage_v, config)
    soc_end = estimate_soc_from_voltage(end_voltage_v, config)
    soc_window = soc_start - soc_end

    if soc_window <= 0:
        return {
            "event_id": event_validation["event_id"],
            "event_type": event_validation["event_type"],
            "valid_for_direct_soh": False,
            "event_capacity_ah": float(event_capacity_ah),
            "reference_capacity_ah": float(reference_capacity_ah),
            "estimated_full_capacity_ah": None,
            "corrected_capacity_ah": None,
            "soc_start": float(soc_start),
            "soc_end": float(soc_end),
            "soc_window": float(soc_window),
            "soh_percent": None,
            "soh_status": "invalid_soc_window",
            "needs_soc_window_correction": True,
            "notes": list(event_validation["reasons"]) + [
                "SOC window is zero or negative; cannot correct partial discharge."
            ]
        }

    if soc_window < 0.10:
        return {
            "event_id": event_validation["event_id"],
            "event_type": event_validation["event_type"],
            "valid_for_direct_soh": False,
            "event_capacity_ah": float(event_capacity_ah),
            "reference_capacity_ah": float(reference_capacity_ah),
            "estimated_full_capacity_ah": None,
            "corrected_capacity_ah": None,
            "soc_start": float(soc_start),
            "soc_end": float(soc_end),
            "soc_window": float(soc_window),
            "soh_percent": None,
            "soh_status": "soc_window_too_small",
            "needs_soc_window_correction": True,
            "notes": list(event_validation["reasons"]) + [
                "SOC window too small for reliable correction."
            ]
        }

    estimated_full_capacity_ah = event_capacity_ah / soc_window
    soh_percent = (estimated_full_capacity_ah / reference_capacity_ah) * 100.0

    return {
        "event_id": event_validation["event_id"],
        "event_type": event_validation["event_type"],
        "valid_for_direct_soh": False,
        "event_capacity_ah": float(event_capacity_ah),
        "reference_capacity_ah": float(reference_capacity_ah),
        "estimated_full_capacity_ah": float(estimated_full_capacity_ah),
        "corrected_capacity_ah": float(estimated_full_capacity_ah),
        "soc_start": float(soc_start),
        "soc_end": float(soc_end),
        "soc_window": float(soc_window),
        "soh_percent": float(soh_percent),
        "soh_status": "soc_corrected_partial_discharge",
        "needs_soc_window_correction": False,
        "notes": list(event_validation["reasons"]) + [
            "SOH estimated using SOC-window correction for partial discharge."
        ]
    }


def build_non_discharge_result(event_capacity_ah, event_validation, config):
    """
    Build result for charge/rest/unknown events.
    """
    return {
        "event_id": event_validation["event_id"],
        "event_type": event_validation["event_type"],
        "valid_for_direct_soh": False,
        "event_capacity_ah": float(event_capacity_ah),
        "reference_capacity_ah": float(config["reference_capacity_ah"]),
        "estimated_full_capacity_ah": None,
        "corrected_capacity_ah": None,
        "soc_start": None,
        "soc_end": None,
        "soc_window": None,
        "soh_percent": None,
        "soh_status": "not_applicable",
        "needs_soc_window_correction": False,
        "notes": list(event_validation["reasons"])
    }


def compute_event_soh(event_capacity_ah, event_validation, config):
    """
    Main SOH decision function.

    Rules:
    - full_discharge -> compute direct SOH
    - partial_discharge -> compute SOC-corrected SOH
    - charge/rest/other -> no SOH
    """
    row_mode = event_validation["row_mode"]
    event_type = event_validation["event_type"]

    if row_mode != "discharge":
        return build_non_discharge_result(event_capacity_ah, event_validation, config)

    if event_type == "full_discharge":
        return build_full_discharge_soh_result(event_capacity_ah, event_validation, config)

    if event_type == "partial_discharge":
        return build_partial_discharge_result(event_capacity_ah, event_validation, config)

    return build_non_discharge_result(event_capacity_ah, event_validation, config)
