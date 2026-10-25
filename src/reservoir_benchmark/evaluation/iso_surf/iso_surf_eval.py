"""
ISO-SURF evaluation module for the unified reservoir operation framework.

This module evaluates previously saved ISO-SURF policies under the unified
reservoir definition, including:
1) loading fitted ISO-SURF policy PKL artifacts,
2) running closed-loop simulation on the synthetic evaluation inflow series,
3) computing reliability and average annual electricity, and
4) exporting standardized ISO-SURF evaluation results.

The purpose of this module is to keep ISO-SURF evaluation independent from
upstream DP preprocessing and policy fitting, so that evaluation only consumes
saved policy artifacts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from reservoir_benchmark.evaluation.common_metrics import (
    summarize_single_reservoir_trajectory,
    write_summary_json,
)
from reservoir_benchmark.policy_generation.iso_surf.iso_surf_runner import ISOSurfPolicyFitter
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
    DDP_SOLVER_VARIANT, base_metadata, data_hash, failure_indicator, physics_hash,
)


class ISOReservoirSimulator:
    """
    Closed-loop simulator using fitted ISO-SURF policy surfaces.
    """

    def __init__(self):
        self.data_dir = get_data_dir()
        self.policy_dir = get_formal_algorithm_dir("iso_surf") / "policy"
        self.output_dir = get_formal_algorithm_dir("iso_surf") / "evaluation"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.evaluation_inflow_file = self.data_dir / "SynMonthlyInflow_evaluation.xlsx"
        self.penalty_settings_file = self.data_dir / "violation_count_penalty_settings.yaml"
        self.penalty_lambdas = load_formal_penalty_candidates(self.data_dir, "iso_surf")

        self.reservoir = SharedReservoirModel(data_dir=self.data_dir)

        self.dead_level = self.reservoir.dead_level
        self.normal_level = self.reservoir.normal_level
        self.initial_level = self.reservoir.initial_level
        self.max_turbine_discharge = self.reservoir.max_turbine_discharge
        self.max_power = self.reservoir.max_power
        self.firm_power = self.reservoir.firm_power
        self.power_coefficient = self.reservoir.power_coefficient
        self.delta_t_days = self.reservoir.delta_t_days

    def simulate_one_year(
        self,
        year_inflows,
        duration_days,
        initial_level: float,
        policy: ISOSurfPolicyFitter,
    ):
        """
        Simulate one year of operation using the fitted ISO-SURF policy.
        """
        monthly_levels = [initial_level]
        monthly_energy = []
        monthly_power = []
        monthly_spill = []
        monthly_turbine_discharge = []
        monthly_total_discharge = []
        monthly_deficit_energy = []
        monthly_max_levels = []
        monthly_raw_targets = []
        monthly_corrected = []
        monthly_balance_error = []
        monthly_release_cap = []
        monthly_unresolved_excess = []

        current_level = initial_level

        for month in range(12):
            inflow = float(year_inflows[month])
            month_of_year = month + 1

            raw_target = policy.predict_next_level(month_of_year, current_level, inflow)
            transition = self.reservoir.calculate_transition(
                current_level,
                raw_target,
                inflow,
                float(duration_days[month]),
                month=month_of_year,
            )
            target_level = transition["end_level"]
            total_discharge = transition["total_discharge"]
            turbine_discharge = transition["turbine_discharge"]
            spill_discharge = transition["spill_discharge"]
            power_kw = transition["power_kw"]
            energy_gwh = transition["energy_gwh"]

            monthly_power.append(power_kw)
            monthly_energy.append(energy_gwh)
            monthly_turbine_discharge.append(turbine_discharge)
            monthly_spill.append(spill_discharge)
            monthly_total_discharge.append(total_discharge)
            monthly_deficit_energy.append(transition["deficit_energy_gwh"])
            monthly_max_levels.append(transition["monthly_max_target_level"])
            monthly_raw_targets.append(raw_target)
            monthly_corrected.append(transition["was_corrected"])
            monthly_balance_error.append(transition["water_balance_error_m3"])
            monthly_release_cap.append(transition["max_total_release_triggered"])
            monthly_unresolved_excess.append(transition["unresolved_upper_storage_excess_m3"])

            monthly_levels.append(target_level)
            current_level = target_level

        return (
            monthly_levels,
            monthly_energy,
            monthly_power,
            monthly_spill,
            monthly_turbine_discharge,
            monthly_total_discharge,
            monthly_deficit_energy,
            monthly_max_levels,
            monthly_raw_targets,
            monthly_corrected,
            monthly_balance_error,
            monthly_release_cap,
            monthly_unresolved_excess,
        )

    def run_simulation(self, policy: ISOSurfPolicyFitter):
        """
        Run ISO-SURF evaluation using the synthetic evaluation inflow series.
        """
        inflows_data, calendar_years = load_monthly_inflows_with_calendar(
            self.evaluation_inflow_file
        )
        inflows = inflows_data.values
        day_matrix = calendar_days_matrix(calendar_years)
        years = len(inflows)

        all_monthly_levels = [self.initial_level]
        all_monthly_energy = []
        all_monthly_power = []
        all_monthly_spill = []
        all_monthly_turbine_discharge = []
        all_monthly_total_discharge = []
        all_monthly_deficit_energy = []
        all_monthly_max_levels = []
        all_raw_targets = []
        all_corrected = []
        all_balance_error = []
        all_release_cap = []
        all_unresolved_excess = []

        annual_energy_list = []

        for year in range(years):
            year_inflows = inflows[year]

            if year == 0:
                initial_level = self.initial_level
            else:
                initial_level = all_monthly_levels[-1]

            (
                monthly_levels,
                monthly_energy,
                monthly_power,
                monthly_spill,
                monthly_turbine_discharge,
                monthly_total_discharge,
                monthly_deficit_energy,
                monthly_max_levels,
                monthly_raw_targets,
                monthly_corrected,
                monthly_balance_error,
                monthly_release_cap,
                monthly_unresolved_excess,
            ) = self.simulate_one_year(
                year_inflows,
                day_matrix[year],
                initial_level,
                policy,
            )

            all_monthly_levels.extend(monthly_levels[1:])
            all_monthly_energy.extend(monthly_energy)
            all_monthly_power.extend(monthly_power)
            all_monthly_spill.extend(monthly_spill)
            all_monthly_turbine_discharge.extend(monthly_turbine_discharge)
            all_monthly_total_discharge.extend(monthly_total_discharge)
            all_monthly_deficit_energy.extend(monthly_deficit_energy)
            all_monthly_max_levels.extend(monthly_max_levels)
            all_raw_targets.extend(monthly_raw_targets)
            all_corrected.extend(monthly_corrected)
            all_balance_error.extend(monthly_balance_error)
            all_release_cap.extend(monthly_release_cap)
            all_unresolved_excess.extend(monthly_unresolved_excess)

            annual_energy_list.append(float(np.sum(monthly_energy)))

        inflow_flat = inflows.reshape(-1)
        n = years * 12

        df_iso = pd.DataFrame(
            {
                "Month": range(1, n + 1),
                "Calendar_Year": np.repeat(calendar_years, 12)[:n],
                "Calendar_Month": np.tile(np.arange(1, 13), years)[:n],
                "Start_Water_Level_m": all_monthly_levels[:n],
                "Water_Level_m": all_monthly_levels[1 : n + 1],
                "Raw_Target_Water_Level_m": all_raw_targets,
                "Power_kW": all_monthly_power,
                "Electricity_GWh": all_monthly_energy,
                "Total_Discharge_m3s": all_monthly_total_discharge,
                "Turbine_Discharge_m3s": all_monthly_turbine_discharge,
                "Spill_Discharge_m3s": all_monthly_spill,
                "Inflow_m3s": inflow_flat[:n],
                "Firm_Power_Deficit_kW": np.maximum(
                    0.0,
                    self.firm_power - np.asarray(all_monthly_power, dtype=float),
                ),
                "Deficit_Energy_GWh": all_monthly_deficit_energy,
                "Firm_Power_Violation": failure_indicator(
                    all_monthly_power, self.firm_power
                ).astype(np.int8),
                "Duration_Days": day_matrix.reshape(-1)[:n],
                "Monthly_Max_Target_Level_m": all_monthly_max_levels,
                "Was_Corrected": all_corrected,
                "Water_Balance_Error_m3": all_balance_error,
                "Max_Total_Release_Triggered": all_release_cap,
                "Unresolved_Upper_Storage_Excess_m3": all_unresolved_excess,
            }
        )

        power_series = np.asarray(all_monthly_power, dtype=float)
        reliability = float(np.mean(power_series >= self.firm_power)) if len(power_series) > 0 else None
        avg_annual_energy = float(np.mean(annual_energy_list)) if len(annual_energy_list) > 0 else 0.0

        return df_iso, avg_annual_energy, reliability

    def save_results(self, df_iso: pd.DataFrame, lam: float) -> Path:
        """
        Save ISO-SURF evaluation results to CSV.
        """
        lam_str = format_lambda_for_filename(lam)
        csv_path = self.output_dir / f"iso_surf_eval_lambda_{lam_str}.csv"
        df_iso["Failure_Penalty_GWh"] = float(lam) * df_iso["Firm_Power_Violation"]
        df_iso["Penalized_Objective_GWh"] = (
            df_iso["Electricity_GWh"] - df_iso["Failure_Penalty_GWh"]
        )
        df_iso.to_csv(csv_path, index=False, encoding="utf-8-sig")
        summary = summarize_single_reservoir_trajectory(
            df_iso,
            firm_power_kw=self.firm_power,
            method="ISO-SURF",
        )
        summary.update(base_metadata(
                data_dir=self.data_dir, split="evaluation", penalty_lambda=lam,
                method="ISO-SURF", solver_variant=DDP_SOLVER_VARIANT,
                grid_step=1.0))
        summary.update({
                "information_set": "current_month_inflow_known_before_action",
                "policy_model": "monthly_surface_fitting",
                "violation_month_count": int(df_iso["Firm_Power_Violation"].sum()),
                "total_failure_penalty_gwh": float(df_iso["Failure_Penalty_GWh"].sum()),
                "total_penalized_objective_gwh": float(df_iso["Penalized_Objective_GWh"].sum()),
                "max_abs_water_balance_error_m3": float(df_iso["Water_Balance_Error_m3"].abs().max()),
                "max_total_release_trigger_count": int(df_iso["Max_Total_Release_Triggered"].sum()),
                "unresolved_upper_storage_excess_count": int((df_iso["Unresolved_Upper_Storage_Excess_m3"] > 0).sum()),
            })
        write_summary_json(summary, csv_path.with_suffix(".json"))
        print(f"ISO-SURF evaluation results saved to: {csv_path}")
        return csv_path

    def run_all(self) -> None:
        """
        Load existing ISO-SURF policies and evaluate them on the evaluation inflow series.
        """
        print("\n============================================================")
        print("ISO-SURF evaluation started")
        print("============================================================")
        print(f"Evaluation inflow file: {self.evaluation_inflow_file}")
        print(f"Penalty settings file: {self.penalty_settings_file}")
        print(f"Policy directory: {self.policy_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Penalty lambdas: {self.penalty_lambdas}")
        print("============================================================")

        for lam in self.penalty_lambdas:
            lam_str = format_lambda_for_filename(lam)
            policy_path = self.policy_dir / f"iso_surf_train_lambda_{lam_str}.pkl"

            if not policy_path.exists():
                raise FileNotFoundError(
                    f"Required ISO-SURF policy file not found: {policy_path}"
                )

            print("\n------------------------------------------------------------")
            print(f"Evaluating ISO-SURF policy for lambda = {lam}")
            print("------------------------------------------------------------")

            policy = ISOSurfPolicyFitter.load_policy(
                policy_path, penalty_lambda=float(lam))
            if policy.artifact_metadata.get("data_hash") != data_hash(self.data_dir, "train"):
                raise ValueError("ISO policy training data hash is incompatible")
            if policy.artifact_metadata.get("physics_hash") != physics_hash(self.data_dir):
                raise ValueError("ISO policy physics hash is incompatible")
            df_iso, avg_annual_energy, reliability = self.run_simulation(policy)
            self.save_results(df_iso, lam)

            if reliability is None:
                print(
                    f"lam={lam_str}  reliability=NA  "
                    f"avg annual electricity={avg_annual_energy:.2f} GWh/year"
                )
            else:
                print(
                    f"lam={lam_str}  reliability={reliability * 100:.2f}%  "
                    f"avg annual electricity={avg_annual_energy:.2f} GWh/year"
                )


def main_iso_surf_eval() -> None:
    simulator = ISOReservoirSimulator()
    simulator.run_all()


if __name__ == "__main__":
    main_iso_surf_eval()
