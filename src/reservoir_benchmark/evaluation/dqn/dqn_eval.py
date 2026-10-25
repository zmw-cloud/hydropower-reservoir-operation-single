"""Independent per-seed evaluation for the single-reservoir Double DQN."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from reservoir_benchmark.policy_generation.dqn.dqn_agent import (
    BasicInfo,
    Config,
    DQNAgent,
    set_global_seed,
)
from reservoir_benchmark.policy_generation.dqn.dqn_reservoir import ReservoirEnv
from reservoir_benchmark.shared.project_paths import (
    format_lambda_for_filename,
    get_formal_algorithm_dir,
)
from reservoir_benchmark.shared.experiment_contract import base_metadata, data_hash, physics_hash
from reservoir_benchmark.shared.project_paths import get_data_dir


class ReservoirEvaluator:
    """Evaluate one frozen seed over the independent 400-year workbook."""

    def __init__(self, env: ReservoirEnv, agent: DQNAgent, output_dir: Path):
        self.env = env
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_evaluation(
        self,
        model_path: str | Path,
        *,
        seed: int,
        lam: float,
    ) -> tuple[Path, Path, dict]:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not self.agent.load_model(model_path, expected_lambda=lam):
            raise RuntimeError(f"Cannot load DQN model: {model_path}")
        if self.agent.artifact_metadata.get("data_hash") != data_hash(get_data_dir(), "train"):
            raise ValueError("DQN checkpoint training data hash is incompatible")
        if self.agent.artifact_metadata.get("physics_hash") != physics_hash(get_data_dir()):
            raise ValueError("DQN checkpoint physics hash is incompatible")
        if self.agent.artifact_metadata.get("checkpoint_role") != "final":
            raise ValueError("Independent evaluation requires a final checkpoint")

        self.env.is_training = False
        self.env.current_year_index = 0
        self.env.max_steps = len(self.env.test_data) * 12
        observation, _ = self.env.reset()
        records: list[dict] = []
        out_of_range_count = 0
        correction_count = 0

        eval_was_training = self.agent.eval_net.training
        target_was_training = self.agent.target_net.training
        self.agent.eval_net.eval()
        self.agent.target_net.eval()
        try:
            with torch.no_grad():
                while True:
                    inflow = float(observation[2])
                    outside = bool(
                        inflow < self.agent.state_low[2]
                        or inflow > self.agent.state_high[2]
                    )
                    out_of_range_count += int(outside)
                    mask = self.env.get_feasible_action_mask()
                    action = self.agent.choose_action(
                        observation,
                        env=self.env,
                        evaluate=True,
                        action_mask=mask,
                    )
                    next_observation, reward, terminated, truncated, info = (
                        self.env.step(action)
                    )
                    correction_count += int(info["was_corrected"])
                    records.append(
                        {
                            "Month": len(records) + 1,
                            "Calendar_Year": int(
                                self.env.test_years[int(info["year_index"])]
                            ),
                            "Calendar_Month": int(info["month"]),
                            "Start_Water_Level_m": float(
                                info["start_water_level"]
                            ),
                            "Water_Level_m": float(info["water_level"]),
                            "Inflow_m3s": float(info["inflow"]),
                            "Raw_Action": int(info["raw_action"]),
                            "Nearest_Grid_Index_To_Executed_Level": int(
                                info["nearest_grid_index_to_executed_level"]
                            ),
                            "Was_Corrected": bool(info["was_corrected"]),
                            "Correction_Amount_m": float(
                                info["correction_amount_m"]
                            ),
                            "Feasible_Action_Count": int(
                                info["feasible_action_count"]
                            ),
                            "Target_Storage_1e9_m3": float(
                                info["target_storage"]
                            ),
                            "Requested_Target_Water_Level_m": float(
                                info["requested_target_level"]
                            ),
                            "Executed_End_Water_Level_m": float(
                                info["executed_end_level"]
                            ),
                            "End_Storage_1e9_m3": float(info["end_storage"]),
                            "Total_Discharge_m3s": float(
                                info["total_discharge"]
                            ),
                            "Turbine_Discharge_m3s": float(
                                info["turbine_discharge"]
                            ),
                            "Spill_Discharge_m3s": float(
                                info["spill_discharge"]
                            ),
                            "Power_kW": float(info["power_output"]),
                            "Electricity_GWh": float(info["electricity_kwh"])
                            / 1e6,
                            "Firm_Power_Deficit_kW": max(
                                0.0,
                                self.env.firm_power
                                - float(info["power_output"]),
                            ),
                            "Deficit_Energy_GWh": float(
                                info["deficit_energy_gwh"]
                            ),
                            "Penalized_Objective_GWh": float(
                                info["penalized_objective_gwh"]
                            ),
                            "Firm_Power_Violation": int(info["firm_power_violation"]),
                            "Failure_Penalty_GWh": float(info["failure_penalty_gwh"]),
                            "Duration_Days": float(info["duration_days"]),
                            "Inflow_Out_Of_Training_Range": outside,
                            "Normalized_Reward": float(reward),
                            "Max_Total_Release_Triggered": bool(
                                info["max_total_release_triggered"]
                            ),
                            "Unresolved_Upper_Storage_Excess_m3": float(
                                info["unresolved_upper_storage_excess_m3"]
                            ),
                            "Water_Balance_Error_m3": float(
                                info["water_balance_error_m3"]
                            ),
                        }
                    )
                    if terminated or truncated:
                        break
                    observation = next_observation
        finally:
            self.agent.eval_net.train(eval_was_training)
            self.agent.target_net.train(target_was_training)

        trajectory = pd.DataFrame(records)
        months = len(trajectory)
        power = trajectory["Power_kW"].to_numpy(dtype=float)
        generation = float(trajectory["Electricity_GWh"].sum())
        deficit_energy = float(trajectory["Deficit_Energy_GWh"].sum())
        deficits = trajectory["Firm_Power_Deficit_kW"].to_numpy(dtype=float)
        summary = {
            "method": "Double DQN",
            "seed": int(seed),
            "lambda": float(lam),
            "months": months,
            "years": months / 12.0,
            "total_generation_gwh": generation,
            "mean_annual_generation_gwh": generation * 12.0 / months,
            "guarantee_rate": float(np.mean(power >= self.env.firm_power)),
            "deficit_months": int(np.sum(deficits > 0.0)),
            "mean_deficit_kw": float(np.mean(deficits)),
            "maximum_deficit_kw": float(np.max(deficits)),
            "total_deficit_energy_gwh": deficit_energy,
            "final_water_level_m": float(trajectory["Water_Level_m"].iloc[-1]),
            "total_spill_m3s_sum": float(
                trajectory["Spill_Discharge_m3s"].sum()
            ),
            "mean_spill_discharge_m3s": float(
                trajectory["Spill_Discharge_m3s"].mean()
            ),
            "total_spill_volume_m3": float(
                np.sum(
                    trajectory["Spill_Discharge_m3s"].to_numpy(dtype=float)
                    * trajectory["Duration_Days"].to_numpy(dtype=float)
                    * 86400.0
                )
            ),
            "total_penalized_objective_gwh": float(
                trajectory["Penalized_Objective_GWh"].sum()
            ),
            "correction_count": int(correction_count),
            "out_of_range_count": int(out_of_range_count),
            "out_of_range_rate": float(out_of_range_count / months),
            "violation_month_count": int(trajectory["Firm_Power_Violation"].sum()),
            "total_failure_penalty_gwh": float(trajectory["Failure_Penalty_GWh"].sum()),
            "normalization_inflow_min_m3s": float(self.agent.state_low[2]),
            "normalization_inflow_max_m3s": float(self.agent.state_high[2]),
            "evaluation_split": "independent_400_year_evaluation",
            "checkpoint": str(model_path),
        }
        summary.update(base_metadata(
            data_dir=get_data_dir(), split="evaluation", penalty_lambda=lam,
            method="Double DQN", solver_variant="fixed_75_output_double_dqn_v2",
            grid_step=1.0))
        summary.update({
            "state_order": ["month", "xiaowan_level_m", "xiaowan_current_inflow_m3s"],
            "reward_scale_gwh": float(self.env.reward_scale_gwh),
            "reward_clip": None,
            "gamma": float(self.agent.gamma),
            "epsilon_schedule": "linear_environment_steps",
            "evaluation_greedy": True,
            "evaluation_learning": False,
            "checkpoint_metadata": self.agent.artifact_metadata,
            "max_total_release_trigger_count": int(
                trajectory["Max_Total_Release_Triggered"].sum()
            ),
            "unresolved_upper_storage_excess_count": int(
                (trajectory["Unresolved_Upper_Storage_Excess_m3"] > 0).sum()
            ),
            "max_abs_water_balance_error_m3": float(
                trajectory["Water_Balance_Error_m3"].abs().max()
            ),
        })

        lam_str = format_lambda_for_filename(float(lam))
        stem = f"dqn_eval_lambda_{lam_str}_seed_{seed}"
        trajectory_path = self.output_dir / f"{stem}.csv"
        summary_path = self.output_dir / f"{stem}.json"
        trajectory.to_csv(trajectory_path, index=False, encoding="utf-8-sig")
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return trajectory_path, summary_path, summary


def _aggregate_seed_metrics(rows: list[dict], output_dir: Path, lam: float) -> None:
    """Aggregate seed-level metrics, never threshold an averaged trajectory."""
    frame = pd.DataFrame(rows)
    lam_str = format_lambda_for_filename(float(lam))
    frame.to_csv(
        output_dir / f"dqn_seed_metrics_lambda_{lam_str}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metric_columns = [
        "total_generation_gwh",
        "mean_annual_generation_gwh",
        "guarantee_rate",
        "violation_month_count",
        "deficit_months",
        "mean_deficit_kw",
        "maximum_deficit_kw",
        "total_deficit_energy_gwh",
        "total_failure_penalty_gwh",
        "total_spill_m3s_sum",
        "mean_spill_discharge_m3s",
        "total_spill_volume_m3",
        "correction_count",
        "out_of_range_count",
        "out_of_range_rate",
    ]
    aggregate = {
        metric: {
            "mean": float(frame[metric].mean()),
            "std": float(frame[metric].std(ddof=1)) if len(frame) > 1 else 0.0,
            "median": float(frame[metric].median()),
        }
        for metric in metric_columns
    }
    payload = {
        "method": "Double DQN",
        "lambda": float(lam),
        "seed_count": len(frame),
        "seeds": frame["seed"].astype(int).tolist(),
        "aggregation_unit": "per_seed_physical_metrics",
        "monthly_trajectory_averaging_used_for_guarantee_rate": False,
        "metrics": aggregate,
    }
    (output_dir / f"dqn_seed_summary_lambda_{lam_str}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_evaluation(*, penalty_lambda: float) -> None:
    config = Config()
    lam = float(penalty_lambda)
    config.agent_config["firm_penalty_lambda"] = lam
    lam_str = format_lambda_for_filename(lam)
    model_dir = get_formal_algorithm_dir("dqn") / "policy"
    output_dir = get_formal_algorithm_dir("dqn") / "evaluation" / f"lambda_{lam_str}" / "by_seed"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []

    for seed in config.get_seed_list():
        seed = int(seed)
        set_global_seed(seed)
        seed_config = Config()
        seed_config.set_seed(seed)
        seed_config.agent_config["firm_penalty_lambda"] = lam
        basic_info = BasicInfo(seed_config)
        env = ReservoirEnv(basic_info)
        env.is_training = False
        agent = DQNAgent(
            n_actions=env.action_space.n,
            n_features=env.observation_space.shape[0],
            config=seed_config,
            basic_info=basic_info,
        )
        model_path = model_dir / f"dqn_train_lambda_{lam_str}_seed_{seed}.pth"
        _, _, summary = ReservoirEvaluator(env, agent, output_dir).run_evaluation(
            model_path,
            seed=seed,
            lam=lam,
        )
        summaries.append(summary)

    _aggregate_seed_metrics(summaries, output_dir.parent, lam)


def main_dqn_eval(*, penalty_lambda: float) -> None:
    run_evaluation(penalty_lambda=penalty_lambda)


if __name__ == "__main__":
    raise SystemExit("Use scripts/run_single_dqn.py evaluation --lambda VALUE")
