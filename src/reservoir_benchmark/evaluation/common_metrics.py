"""Common external physical metrics for all single-reservoir trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize_single_reservoir_trajectory(
    trajectory: pd.DataFrame,
    *,
    firm_power_kw: float,
    method: str,
) -> dict:
    required = {
        "Power_kW",
        "Electricity_GWh",
        "Spill_Discharge_m3s",
        "Duration_Days",
    }
    missing = required - set(trajectory.columns)
    if missing:
        raise ValueError(f"Trajectory is missing external metric columns: {sorted(missing)}")
    power = pd.to_numeric(trajectory["Power_kW"], errors="raise").to_numpy(float)
    generation = pd.to_numeric(
        trajectory["Electricity_GWh"], errors="raise"
    ).to_numpy(float)
    spill = pd.to_numeric(
        trajectory["Spill_Discharge_m3s"], errors="raise"
    ).to_numpy(float)
    days = pd.to_numeric(trajectory["Duration_Days"], errors="raise").to_numpy(float)
    deficit_kw = np.maximum(0.0, float(firm_power_kw) - power)
    violation = power < float(firm_power_kw)
    if "Deficit_Energy_GWh" in trajectory.columns:
        deficit_energy = pd.to_numeric(
            trajectory["Deficit_Energy_GWh"], errors="raise"
        ).to_numpy(float)
    else:
        deficit_energy = deficit_kw * days * 24.0 / 1e6
    months = len(trajectory)
    if months == 0:
        raise ValueError("Trajectory cannot be empty")
    total_generation = float(generation.sum())
    summary = {
        "method": str(method),
        "months": months,
        "years": months / 12.0,
        "total_generation_gwh": total_generation,
        "mean_annual_generation_gwh": total_generation * 12.0 / months,
        "guarantee_rate": float(np.mean(~violation)),
        "violation_month_count": int(np.sum(violation)),
        "deficit_months": int(np.sum(violation)),
        "mean_deficit_kw": float(np.mean(deficit_kw)),
        "maximum_deficit_kw": float(np.max(deficit_kw)),
        "total_deficit_energy_gwh": float(deficit_energy.sum()),
        "mean_spill_discharge_m3s": float(np.mean(spill)),
        "total_spill_volume_m3": float(np.sum(spill * days * 86400.0)),
    }
    if "Water_Level_m" in trajectory.columns:
        levels = pd.to_numeric(
            trajectory["Water_Level_m"], errors="raise"
        ).to_numpy(float)
        summary["final_water_level_m"] = float(levels[-1])
    return summary


def write_summary_json(summary: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def ddp_relative_metrics(method_summary: dict, ddp_summary: dict) -> dict:
    ddp_generation = float(ddp_summary["total_generation_gwh"])
    if ddp_generation <= 0.0:
        raise ValueError("DDP generation must be positive")
    return {
        "generation_gap_to_same_problem_ddp": (
            ddp_generation - float(method_summary["total_generation_gwh"])
        )
        / ddp_generation,
        "guarantee_rate_difference_to_same_problem_ddp": (
            float(method_summary["guarantee_rate"])
            - float(ddp_summary["guarantee_rate"])
        ),
    }
