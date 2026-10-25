"""
ISO-SURF preprocessing module for the unified reservoir operation framework.

This module prepares the upstream deterministic dynamic programming (DP)
training trajectories required for ISO-SURF policy fitting, including:
1) loading the synthetic training inflow series,
2) loading ISO-SURF penalty coefficient candidates,
3) reusing the shared DP solver under the unified reservoir definition, and
4) exporting standardized monthly DP trajectory tables for subsequent
   ISO-SURF fitting.

The purpose of this module is to provide an ISO-specific wrapper around the
existing DP solver without duplicating the DP core algorithm.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from reservoir_benchmark.policy_generation.ddp.ddp_solver import ReservoirOperation as DPSolver
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
from reservoir_benchmark.shared.reservoir_model import SharedReservoirModel
from reservoir_benchmark.shared.experiment_contract import (
    DDP_SOLVER_VARIANT, base_metadata, failure_indicator, sha256_file, write_json,
)


def save_iso_upstream_dp_results(
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
) -> Path:
    """
    Save upstream DP training results for ISO-SURF fitting.

    File name pattern:
        iso_upstream_dp_train_lambda_{lam}.csv
    """
    result_dir.mkdir(parents=True, exist_ok=True)

    lam_str = format_lambda_for_filename(lam)
    csv_path = result_dir / f"iso_upstream_dp_train_lambda_{lam_str}.csv"

    calendar_years = np.repeat(np.asarray(years, dtype=int), 12)
    calendar_months = np.tile(np.arange(1, 13, dtype=int), len(years))
    start_levels = np.concatenate(
        ([reservoir.initial_level], np.asarray(optimal_levels[:-1], dtype=float))
    )
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
        }
    )

    monthly_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    metadata = base_metadata(
        data_dir=reservoir.data_dir, split="train", penalty_lambda=lam,
        method="DDP teacher for ISO-SURF", solver_variant=DDP_SOLVER_VARIANT,
        grid_step=1.0)
    metadata.update({
        "teacher_csv": str(csv_path),
        "teacher_hash": sha256_file(csv_path),
        "policy_source": "full_400_year_training_DDP_trajectory",
        "terminal_constraint": "hard_final_end_state_1240m",
    })
    write_json(csv_path.with_suffix(".json"), metadata)
    print(f"Saved ISO upstream DP results: {csv_path.resolve()}")
    return csv_path


class ISOPreprocessor:
    """
    Prepare and export upstream DP training trajectories for ISO-SURF fitting.
    """

    def __init__(self):
        self.data_dir = get_data_dir()
        self.output_dir = get_formal_algorithm_dir("iso_surf") / "teacher"

        self.train_inflow_file = self.data_dir / "SynMonthlyInflow_train.xlsx"
        self.penalty_settings_file = self.data_dir / "violation_count_penalty_settings.yaml"

        inflow_frame, self.years = load_monthly_inflows_with_calendar(
            self.train_inflow_file
        )
        self.inflows = inflow_frame.values
        self.month_days = calendar_days_matrix(self.years)
        self.penalty_lambdas = load_formal_penalty_candidates(self.data_dir, "iso_surf")

        self.reservoir = SharedReservoirModel(data_dir=self.data_dir)
        self.dp_solver = DPSolver(
            reservoir=self.reservoir,
            inflows=self.inflows,
            month_days=self.month_days,
        )

    def run_one(self, lam: float) -> Path:
        """
        Run the shared DP solver for one ISO-SURF lambda and save the upstream
        DP monthly trajectory table.
        """
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
        ) = self.dp_solver.optimize(firm_power_penalty_lambda=float(lam))

        return save_iso_upstream_dp_results(
            result_dir=self.output_dir,
            lam=float(lam),
            optimal_levels=optimal_levels,
            optimal_storages=optimal_storages,
            optimal_powers=optimal_powers,
            optimal_electricity=optimal_electricity,
            optimal_discharges=optimal_discharges,
            optimal_turbine_flows=optimal_turbine_flows,
            optimal_spill_flows=optimal_spill_flows,
            inflows=inflows_flat,
            years=self.years,
            reservoir=self.reservoir,
            deficit_energy=deficit_energy,
            duration_days=duration_days,
            monthly_max_levels=monthly_max_levels,
        )

    def run_all(self) -> None:
        """
        Generate and save upstream DP training trajectories for all ISO-SURF lambdas.
        """
        print("\n============================================================")
        print("ISO-SURF preprocessing started")
        print("============================================================")
        print(f"Training inflow file: {self.train_inflow_file}")
        print(f"Penalty settings file: {self.penalty_settings_file}")
        print(f"Output directory: {self.output_dir}")
        print(f"Penalty lambdas: {self.penalty_lambdas}")
        print("============================================================")

        try:
            for lam in self.penalty_lambdas:
                print("\n------------------------------------------------------------")
                print(f"Generating ISO upstream DP data for lambda = {lam}")
                print("------------------------------------------------------------")
                self.run_one(float(lam))
        finally:
            self.dp_solver.clear_transition_cache()


def main_iso_preprocessor() -> None:
    preprocessor = ISOPreprocessor()
    preprocessor.run_all()


if __name__ == "__main__":
    main_iso_preprocessor()
