"""Continuous closed-loop evaluation of formal one-dimensional SDP policies."""
from __future__ import annotations

import numpy as np
import pandas as pd

from reservoir_benchmark.evaluation.common_metrics import summarize_single_reservoir_trajectory, write_summary_json
from reservoir_benchmark.policy_generation.sdp.sdp_probability import MonthlyInflowModel
from reservoir_benchmark.policy_generation.sdp.sdp_solver import SDPPolicy
from reservoir_benchmark.shared.experiment_contract import (
    SDP_SOLVER_VARIANT, base_metadata, data_hash, physics_hash, require_contract,
)
from reservoir_benchmark.shared.io_utils import calendar_days_matrix, load_formal_penalty_candidates, load_monthly_inflows_with_calendar
from reservoir_benchmark.shared.project_paths import get_data_dir, get_formal_algorithm_dir, format_lambda_for_filename
from reservoir_benchmark.shared.reservoir_model import SharedReservoirModel


def evaluate(policy, inflow_model, reservoir, inflows, years, penalty):
    days = calendar_days_matrix(years)
    current_level = reservoir.initial_level
    rows = []
    prior_end_storage = None
    max_jump = 0.0
    for year_index in range(len(years)):
        for month_index in range(12):
            month = month_index + 1
            inflow = float(inflows[year_index, month_index])
            inflow_class = inflow_model.classify(month, inflow)
            requested, lookup_state = policy.action(month, current_level, inflow_class)
            transition = reservoir.calculate_transition(
                current_level, requested, inflow, days[year_index, month_index], month=month)
            if prior_end_storage is not None:
                start_storage = reservoir.calculate_storage(current_level)
                max_jump = max(max_jump, abs(start_storage - prior_end_storage) * 1e9)
            violation = bool(transition["power_kw"] < reservoir.firm_power)
            rows.append({
                "Month": len(rows) + 1,
                "Calendar_Year": int(years[year_index]),
                "Calendar_Month": month,
                "Start_Water_Level_m": current_level,
                "Requested_Target_Water_Level_m": requested,
                "Water_Level_m": transition["end_level"],
                "Lookup_State_Level_m": float(policy.levels[lookup_state]),
                "Inflow_Class": inflow_class,
                "Inflow_m3s": inflow,
                "Power_kW": transition["power_kw"],
                "Electricity_GWh": transition["energy_gwh"],
                "Firm_Power_Violation": int(violation),
                "Firm_Power_Deficit_kW": max(0.0, reservoir.firm_power - transition["power_kw"]),
                "Deficit_Energy_GWh": transition["deficit_energy_gwh"],
                "Failure_Penalty_GWh": float(penalty) * int(violation),
                "Penalized_Objective_GWh": transition["energy_gwh"] - float(penalty) * int(violation),
                "Total_Discharge_m3s": transition["total_discharge"],
                "Turbine_Discharge_m3s": transition["turbine_discharge"],
                "Spill_Discharge_m3s": transition["spill_discharge"],
                "Duration_Days": float(days[year_index, month_index]),
                "Monthly_Max_Target_Level_m": transition["monthly_max_target_level"],
                "Was_Corrected": transition["was_corrected"],
                "Correction_Amount_m": transition["correction_amount_m"],
                "Water_Balance_Error_m3": transition["water_balance_error_m3"],
                "Max_Total_Release_Triggered": transition["max_total_release_triggered"],
                "Unresolved_Upper_Storage_Excess_m3": transition["unresolved_upper_storage_excess_m3"],
            })
            current_level = transition["end_level"]
            prior_end_storage = transition["end_storage"]
    return pd.DataFrame(rows), max_jump


def main_sdp_eval():
    data_dir = get_data_dir()
    root = get_formal_algorithm_dir("sdp")
    penalties = load_formal_penalty_candidates(data_dir, "sdp")
    inflow_model = MonthlyInflowModel.load(root / "training" / "monthly_inflow_model.npz")
    frame, years = load_monthly_inflows_with_calendar(data_dir / "SynMonthlyInflow_evaluation.xlsx")
    inflows = frame.to_numpy(dtype=float)
    reservoir = SharedReservoirModel(data_dir=data_dir)
    output = root / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    for penalty in penalties:
        tag = format_lambda_for_filename(penalty)
        policy = SDPPolicy.load(root / "policy" / f"sdp_policy_lambda_{tag}.npz",
                                penalty_lambda=penalty)
        require_contract(policy.metadata, penalty_lambda=penalty,
                         solver_variant=SDP_SOLVER_VARIANT)
        if policy.metadata.get("data_hash") != data_hash(data_dir, "train"):
            raise ValueError("SDP policy training data hash is incompatible")
        if policy.metadata.get("physics_hash") != physics_hash(data_dir):
            raise ValueError("SDP policy physics hash is incompatible")
        if policy.metadata["training_inflow_model"].get("training_content_sha256") != inflow_model.metadata.get("training_content_sha256"):
            raise ValueError("SDP policy and inflow model came from different training data")
        schedule, max_jump = evaluate(policy, inflow_model, reservoir, inflows, years, penalty)
        csv_path = output / f"sdp_eval_lambda_{tag}.csv"
        schedule.to_csv(csv_path, index=False, encoding="utf-8-sig")
        summary = summarize_single_reservoir_trajectory(
            schedule, firm_power_kw=reservoir.firm_power, method="SDP")
        summary.update(base_metadata(data_dir=data_dir, split="evaluation",
                                     penalty_lambda=penalty, method="SDP",
                                     solver_variant=SDP_SOLVER_VARIANT, grid_step=1.0))
        summary.update({
            "violation_month_count": int(schedule["Firm_Power_Violation"].sum()),
            "total_failure_penalty_gwh": float(schedule["Failure_Penalty_GWh"].sum()),
            "total_penalized_objective_gwh": float(schedule["Penalized_Objective_GWh"].sum()),
            "target_repair_count": int(schedule["Was_Corrected"].sum()),
            "maximum_intermonth_storage_jump_m3": max_jump,
            "max_abs_water_balance_error_m3": float(schedule["Water_Balance_Error_m3"].abs().max()),
            "max_total_release_trigger_count": int(schedule["Max_Total_Release_Triggered"].sum()),
            "unresolved_upper_storage_excess_count": int((schedule["Unresolved_Upper_Storage_Excess_m3"] > 0).sum()),
            "policy_metadata": policy.metadata,
        })
        write_summary_json(summary, csv_path.with_suffix(".json"))
        print(f"Saved formal SDP evaluation: {csv_path}")


if __name__ == "__main__":
    main_sdp_eval()
