"""
DP execution and result-export module for the unified reservoir operation
framework.

This module connects the deterministic dynamic programming (DP) solver to the
shared project infrastructure, including:
1) loading training inflow data,
2) loading penalty coefficient candidates for DP runs,
3) constructing the shared reservoir object,
4) executing the DP solver for each specified penalty coefficient, and
5) exporting monthly optimization results to the standardized output directory.

The purpose of this module is to separate experiment orchestration from the DP
solver itself, thereby keeping the optimization logic independent from path
management, data loading, parameter looping, and result saving.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from reservoir_benchmark.evaluation.common_metrics import (
    summarize_single_reservoir_trajectory,
    write_summary_json,
)
from reservoir_benchmark.policy_generation.ddp.ddp_solver import ReservoirOperation
from reservoir_benchmark.shared.io_utils import (
    calendar_days_matrix,
    load_monthly_inflows_with_calendar,
    load_formal_penalty_candidates,
)
from reservoir_benchmark.shared.project_paths import (
    format_lambda_for_filename,
    get_data_dir,
    get_formal_algorithm_dir,
)
from reservoir_benchmark.shared.experiment_contract import (
    DDP_SOLVER_VARIANT,
    base_metadata,
    failure_indicator,
)
from reservoir_benchmark.shared.reservoir_model import SharedReservoirModel


def save_dp_results(
    result_dir: Path,
    lam: float,
    optimal_levels,
    optimal_storages,
    optimal_powers,
    optimal_electricity,
    optimal_discharges,
    optimal_turbine_flows,
    optimal_spill_flows,
    inflows,
    years,
    reservoir,
    deficit_energy,
    duration_days,
    monthly_max_levels,
) -> None:
    """
    Save monthly DP results as a single CSV file.

    File name pattern:
        ddp_eval_lambda_{lam}.csv
    """
    result_dir.mkdir(parents=True, exist_ok=True)

    lam_str = format_lambda_for_filename(lam)
    csv_filename = f"ddp_eval_lambda_{lam_str}.csv"
    csv_path = result_dir / csv_filename

    calendar_years = np.repeat(np.asarray(years, dtype=int), 12)
    calendar_months = np.tile(np.arange(1, 13, dtype=int), len(years))
    start_levels = np.concatenate(
        ([reservoir.initial_level], np.asarray(optimal_levels[:-1], dtype=float))
    )
    start_storages = np.asarray([reservoir.calculate_storage(z) for z in start_levels])
    end_storages = np.asarray(optimal_storages, dtype=float)
    duration_array = np.asarray(duration_days, dtype=float)
    release_array = np.asarray(optimal_discharges, dtype=float)
    inflow_array = np.asarray(inflows[: len(optimal_levels)], dtype=float)
    balance_error = ((end_storages - start_storages) * 1e9
                     - (inflow_array - release_array) * duration_array * 86400.0)
    monthly_results = pd.DataFrame(
        {
            "Month": range(1, len(optimal_levels) + 1),
            "Calendar_Year": calendar_years[: len(optimal_levels)],
            "Calendar_Month": calendar_months[: len(optimal_levels)],
            "Start_Water_Level_m": start_levels,
            "Water_Level_m": optimal_levels,
            "Target_Water_Level_m": optimal_levels,
            "Storage_1e9_m3": optimal_storages,
            "Power_kW": optimal_powers,
            "Electricity_GWh": optimal_electricity,
            "Total_Discharge_m3s": optimal_discharges,
            "Turbine_Discharge_m3s": optimal_turbine_flows,
            "Spill_Discharge_m3s": optimal_spill_flows,
            "Inflow_m3s": inflows[: len(optimal_levels)],
            "Firm_Power_Deficit_kW": np.maximum(
                0.0,
                reservoir.firm_power - np.asarray(optimal_powers, dtype=float),
            ),
            "Deficit_Energy_GWh": deficit_energy,
            "Firm_Power_Violation": failure_indicator(
                optimal_powers, reservoir.firm_power
            ).astype(np.int8),
            "Failure_Penalty_GWh": float(lam) * failure_indicator(
                optimal_powers, reservoir.firm_power
            ).astype(float),
            "Penalized_Objective_GWh": np.asarray(optimal_electricity, dtype=float)
            - float(lam) * failure_indicator(optimal_powers, reservoir.firm_power),
            "Duration_Days": duration_days,
            "Monthly_Max_Target_Level_m": monthly_max_levels,
            "Water_Balance_Error_m3": balance_error,
            "Max_Total_Release_Triggered": np.isclose(
                release_array, reservoir.max_total_release, rtol=0.0, atol=1e-8),
        }
    )

    monthly_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary = summarize_single_reservoir_trajectory(
        monthly_results,
        firm_power_kw=reservoir.firm_power,
        method="DDP",
    )
    summary.update(base_metadata(
            data_dir=reservoir.data_dir,
            split="evaluation",
            penalty_lambda=lam,
            method="DDP",
            solver_variant=DDP_SOLVER_VARIANT,
            grid_step=1.0,
        ))
    summary.update({
            "information_set": "complete_realized_future_inflow_upper_bound",
            "terminal_level_m": reservoir.terminal_level,
            "violation_month_count": int(monthly_results["Firm_Power_Violation"].sum()),
            "total_failure_penalty_gwh": float(monthly_results["Failure_Penalty_GWh"].sum()),
            "total_penalized_objective_gwh": float(monthly_results["Penalized_Objective_GWh"].sum()),
            "max_abs_water_balance_error_m3": float(np.max(np.abs(balance_error))),
            "max_total_release_trigger_count": int(monthly_results["Max_Total_Release_Triggered"].sum()),
        })
    write_summary_json(summary, csv_path.with_suffix(".json"))
    print(f"Saved DDP results: {csv_path.resolve()}")


def main_ddp_eval() -> None:
    """
    Run formal violation-count DDP candidates.
    """
    data_dir = get_data_dir()
    result_dir = get_formal_algorithm_dir("ddp") / "evaluation"

    inflow_path = data_dir / "SynMonthlyInflow_evaluation.xlsx"
    penalty_settings_path = data_dir / "violation_count_penalty_settings.yaml"

    inflows_df, years = load_monthly_inflows_with_calendar(inflow_path)
    inflows = inflows_df.values
    month_days = calendar_days_matrix(years)

    penalty_lambdas = load_formal_penalty_candidates(data_dir, "ddp")

    reservoir = SharedReservoirModel(data_dir=data_dir)
    dp_solver = ReservoirOperation(
        reservoir=reservoir,
        inflows=inflows,
        month_days=month_days,
    )

    print("\n====================================")
    print("DP evaluation started")
    print(f"Inflow file: {inflow_path}")
    print(f"Penalty settings file: {penalty_settings_path}")
    print(f"Output directory: {result_dir}")
    print(f"Penalty lambdas: {penalty_lambdas}")
    print("====================================")

    for lam in penalty_lambdas:
        print("\n------------------------------------")
        print(f"Running failure_count_penalty_lambda = {lam} GWh/failed month")
        print("------------------------------------")

        (
            value_function,
            decisions,
            optimal_levels,
            optimal_storages,
            optimal_powers,
            optimal_electricity,
            optimal_discharges,
            optimal_turbine_flows,
            optimal_spill_flows,
            inflows_flat,
            deficit_energy,
            duration_days,
            monthly_max_levels,
        ) = dp_solver.optimize(firm_power_penalty_lambda=lam)

        save_dp_results(
            result_dir=result_dir,
            lam=lam,
            optimal_levels=optimal_levels,
            optimal_storages=optimal_storages,
            optimal_powers=optimal_powers,
            optimal_electricity=optimal_electricity,
            optimal_discharges=optimal_discharges,
            optimal_turbine_flows=optimal_turbine_flows,
            optimal_spill_flows=optimal_spill_flows,
            inflows=inflows_flat,
            years=years,
            reservoir=reservoir,
            deficit_energy=deficit_energy,
            duration_days=duration_days,
            monthly_max_levels=monthly_max_levels,
        )
    dp_solver.clear_transition_cache()
