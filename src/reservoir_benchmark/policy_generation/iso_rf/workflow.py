"""Held-out development, lambda selection, final refit and evaluation for ISO-RF.

Development DDP teachers are optimized on the first 320 years only. The last
80 years never enter teacher labels or forest fitting. Independent evaluation
is opened only by the explicit evaluation stage after final fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np
import pandas as pd
import sklearn
import yaml

from reservoir_benchmark.evaluation.common_metrics import summarize_single_reservoir_trajectory
from reservoir_benchmark.policy_generation.ddp.ddp_solver import ReservoirOperation
from reservoir_benchmark.shared.experiment_contract import (
    DDP_SOLVER_VARIANT, OBJECTIVE_VERSION, PENALTY_DEFINITION, PENALTY_UNIT,
    physics_hash, sha256_file,
)
from reservoir_benchmark.shared.io_utils import calendar_days_matrix, load_monthly_inflows_with_calendar
from reservoir_benchmark.shared.project_paths import get_data_dir, get_formal_algorithm_dir, format_lambda_for_filename
from reservoir_benchmark.shared.reservoir_model import SharedReservoirModel
from .policy import FEATURE_NAMES, TARGET_NAME, MonthlySingleRFPolicy, validate_hyperparameters


WORKFLOW_VERSION = "xiaowan_iso_rf_validation_v1"


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def content_hash(frame: pd.DataFrame, years: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(years, dtype="<i8").tobytes())
    digest.update(np.asarray(frame, dtype="<f8").tobytes())
    return digest.hexdigest()


def implementation_hash() -> str:
    project = Path(__file__).resolve().parents[4]
    digest = hashlib.sha256()
    for relative in (
        "src/reservoir_benchmark/policy_generation/iso_rf/policy.py",
        "src/reservoir_benchmark/policy_generation/iso_rf/workflow.py",
        "src/reservoir_benchmark/policy_generation/ddp/ddp_solver.py",
        "src/reservoir_benchmark/policy_generation/ddp/ddp_acceleration.py",
        "src/reservoir_benchmark/shared/reservoir_model.py",
        "src/reservoir_benchmark/evaluation/common_metrics.py",
        "src/reservoir_benchmark/shared/io_utils.py",
    ):
        digest.update(relative.encode())
        digest.update((project / relative).read_bytes())
    return digest.hexdigest()


def split_training(frame: pd.DataFrame, years: np.ndarray):
    values = frame.to_numpy(dtype=float)
    years = np.asarray(years, dtype=int)
    if values.shape != (400, 12) or len(years) != 400:
        raise ValueError("ISO-RF requires exactly 400 complete years with 12 monthly inflows")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Monthly inflows must be finite and nonnegative")
    if not np.all(np.diff(years) == 1):
        raise ValueError("Training years must be consecutive")
    return (frame.iloc[:320].reset_index(drop=True), years[:320].copy(),
            frame.iloc[320:].reset_index(drop=True), years[320:].copy())


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("version") != 1 or config.get("experiment_id") != WORKFLOW_VERSION:
        raise ValueError("Unsupported ISO-RF configuration")
    protocol = config["protocol"]
    if (protocol["fit_years"], protocol["validation_years"]) != (320, 80):
        raise ValueError("The formal split must be 320/80 years")
    if not np.isfinite(protocol["anchor_lambda"]) or protocol["anchor_lambda"] < 0:
        raise ValueError("anchor_lambda must be finite and nonnegative")
    if not 0 <= protocol["lambda_target_guarantee_rate"] <= 1:
        raise ValueError("lambda_target_guarantee_rate must be in [0,1]")
    penalties = [float(value) for value in protocol["lambda_candidates"]]
    if not penalties or len(set(penalties)) != len(penalties):
        raise ValueError("Lambda candidates must be nonempty and unique")
    if not np.isfinite(penalties).all() or min(penalties) < 0:
        raise ValueError("Lambda candidates must be finite and nonnegative")
    rf = config["random_forest"]
    for key in ("fit_n_jobs", "predict_n_jobs"):
        if not isinstance(rf[key], int) or rf[key] == 0 or rf[key] < -1:
            raise ValueError(f"{key} must be -1 or a positive integer")
    identifiers, signatures = set(), set()
    for candidate in rf["candidates"]:
        identifier = candidate["id"]
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
            raise ValueError("Candidate IDs must use only letters, numbers, '_' and '-'")
        parameters = validate_hyperparameters({key: value for key, value in candidate.items() if key != "id"})
        signature = json.dumps(parameters, sort_keys=True)
        if identifier in identifiers or signature in signatures:
            raise ValueError("Duplicate RF candidate ID or parameter combination")
        identifiers.add(identifier)
        signatures.add(signature)
    if not identifiers:
        raise ValueError("At least one RF candidate is required")
    return config


@dataclass
class Context:
    config: dict
    output: Path
    data: Path
    reservoir: SharedReservoirModel
    train: pd.DataFrame
    train_years: np.ndarray
    fit: pd.DataFrame
    fit_years: np.ndarray
    validation: pd.DataFrame
    validation_years: np.ndarray
    contract: dict

    @classmethod
    def build(cls, config_path: Path, output: Path | None = None):
        config = load_config(config_path)
        data = get_data_dir()
        train_path = data / "SynMonthlyInflow_train.xlsx"
        train, years = load_monthly_inflows_with_calendar(train_path)
        fit, fit_years, validation, validation_years = split_training(train, years)
        reservoir = SharedReservoirModel(data_dir=data)
        protocol = config["protocol"]
        if reservoir.initial_level != protocol["initial_level_m"] or reservoir.initial_level != 1240:
            raise ValueError("The formal initial level must be 1240 m")
        if reservoir.terminal_level != protocol["teacher_terminal_level_m"] or reservoir.terminal_level != 1240:
            raise ValueError("The formal DDP teacher terminal level must be 1240 m")
        contract = {
            "workflow_version": WORKFLOW_VERSION,
            "objective_version": OBJECTIVE_VERSION,
            "penalty_definition": PENALTY_DEFINITION,
            "penalty_unit": PENALTY_UNIT,
            "strict_failure_comparison": "power_kw < firm_power_kw",
            "train_workbook_hash": sha256_file(train_path),
            "development_train_content_hash": content_hash(fit, fit_years),
            "validation_content_hash": content_hash(validation, validation_years),
            "physics_hash": physics_hash(data),
            "implementation_hash": implementation_hash(),
            "development_years": 320,
            "validation_years": 80,
            "initial_level_m": reservoir.initial_level,
            "teacher_terminal_level_m": reservoir.terminal_level,
        }
        return cls(config, (output or get_formal_algorithm_dir("iso_rf")).resolve(),
                   data, reservoir, train, years, fit, fit_years, validation, validation_years,
                   contract)

    def require_record(self, record: dict):
        if record.get("experiment_contract") != self.contract:
            raise ValueError("Artifact belongs to different data, physics or ISO-RF implementation")


def forecast_metadata(ctx: Context) -> dict:
    return {
        **ctx.contract,
        "information_set": "month_start_level_and_current_month_inflow",
        "state_timing": "month_start",
        "action_timing": "same_month_end",
        "action_semantics": "continuous_requested_target_with_physical_repair",
        "teacher_solver_variant": DDP_SOLVER_VARIANT,
        "policy_family": "monthly_random_forest",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
    }


class TeacherBuilder:
    """Cache compatible teacher artifacts; reuse the DDP transition cache across lambdas."""
    def __init__(self, ctx: Context, *, final: bool = False):
        self.ctx = ctx
        self.final = final
        self.frame = ctx.train if final else ctx.fit
        self.years = ctx.train_years if final else ctx.fit_years
        self.scope = "full_train_400y" if final else "development_train_320y"
        self.solver = None

    def close(self):
        if self.solver is not None:
            self.solver.clear_transition_cache()

    def get(self, penalty: float) -> tuple[pd.DataFrame, dict]:
        root = self.ctx.output / "teachers" / self.scope
        path = root / f"ddp_teacher_lambda_{format_lambda_for_filename(penalty)}.csv"
        meta_path = path.with_suffix(".json")
        expected = {
            "experiment_contract": self.ctx.contract,
            "teacher_split": self.scope,
            "teacher_input_content_hash": content_hash(self.frame, self.years),
            "teacher_months": len(self.frame) * 12,
            "teacher_solver_variant": DDP_SOLVER_VARIANT,
            "teacher_terminal_level_m": self.ctx.reservoir.terminal_level,
            "lambda": float(penalty),
        }
        if path.exists() or meta_path.exists():
            if not path.is_file() or not meta_path.is_file():
                raise ValueError(f"Incomplete teacher cache: {path}; use a fresh output directory")
            metadata = read_json(meta_path)
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Incompatible cached teacher: {path}")
            if metadata["teacher_hash"] != sha256_file(path):
                raise ValueError(f"Teacher CSV hash mismatch: {path}")
            print(f"Reusing teacher: {path}", flush=True)
            return pd.read_csv(path), metadata
        if self.solver is None:
            self.solver = ReservoirOperation(
                self.ctx.reservoir, self.frame.to_numpy(float),
                month_days=calendar_days_matrix(self.years),
            )
        print(f"Generating DDP teacher: {self.scope}, lambda={penalty:g}", flush=True)
        result = self.solver.optimize(firm_power_penalty_lambda=penalty)
        levels, powers, energy = np.asarray(result[2]), np.asarray(result[4]), np.asarray(result[5])
        starts = np.r_[self.ctx.reservoir.initial_level, levels[:-1]]
        teacher = pd.DataFrame({
            "Month": np.arange(1, len(levels) + 1),
            "Calendar_Year": np.repeat(self.years, 12),
            "Calendar_Month": np.tile(np.arange(1, 13), len(self.years)),
            "Start_Water_Level_m": starts,
            "Target_Water_Level_m": levels,
            "Water_Level_m": levels,
            "Inflow_m3s": self.frame.to_numpy(float).ravel(),
            "Power_kW": powers,
            "Electricity_GWh": energy,
            "Duration_Days": result[11],
        })
        write_csv(path, teacher)
        metadata = {**expected, "teacher_hash": sha256_file(path),
                    "teacher_first_year": int(self.years[0]),
                    "teacher_last_year": int(self.years[-1]),
                    "terminal_constraint": "hard_only_at_end_of_teacher_horizon",
                    "validation_future_used_by_ddp": False}
        write_json(meta_path, metadata)
        return teacher, metadata


def simulate_policy(policy, frame: pd.DataFrame, years: np.ndarray,
                    reservoir: SharedReservoirModel, penalty: float) -> tuple[pd.DataFrame, dict]:
    """Continuous closed loop, with no annual reset or forced final refill."""
    values = frame.to_numpy(float)
    years = np.asarray(years, dtype=int)
    if values.shape != (len(years), 12) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Simulation input must contain complete finite nonnegative years")
    days = calendar_days_matrix(years)
    current_level = reservoir.initial_level
    previous_end_storage = None
    rows = []
    for year_index, year in enumerate(years):
        for month_index in range(12):
            month = month_index + 1
            inflow = float(values[year_index, month_index])
            raw, outside = policy.predict(month, current_level, inflow)
            transition = reservoir.calculate_transition(
                current_level, raw, inflow, days[year_index, month_index], month=month
            )
            failure = int(transition["power_kw"] < reservoir.firm_power)
            jump = (0.0 if previous_end_storage is None else
                    (transition["current_storage"] - previous_end_storage) * 1e9)
            rows.append({
                "Month": len(rows) + 1,
                "Calendar_Year": int(year), "Calendar_Month": month,
                "Start_Water_Level_m": current_level,
                "Raw_Target_Water_Level_m": raw,
                "Requested_Target_Water_Level_m": transition["bounded_target_level"],
                "Water_Level_m": transition["end_level"],
                "Start_Storage_1e9_m3": transition["current_storage"],
                "End_Storage_1e9_m3": transition["end_storage"],
                "Inflow_m3s": inflow,
                "Power_kW": transition["power_kw"],
                "Electricity_GWh": transition["energy_gwh"],
                "Head_m": transition["head"],
                "Tailwater_Level_m": transition["tailwater_level"],
                "Total_Discharge_m3s": transition["total_discharge"],
                "Turbine_Discharge_m3s": transition["turbine_discharge"],
                "Spill_Discharge_m3s": transition["spill_discharge"],
                "Firm_Power_Violation": failure,
                "Firm_Power_Deficit_kW": max(0.0, reservoir.firm_power - transition["power_kw"]),
                "Deficit_Energy_GWh": transition["deficit_energy_gwh"],
                "Failure_Penalty_GWh": penalty * failure,
                "Penalized_Objective_GWh": transition["energy_gwh"] - penalty * failure,
                "Duration_Days": float(days[year_index, month_index]),
                "Monthly_Max_Target_Level_m": transition["monthly_max_target_level"],
                "Was_Corrected": transition["was_corrected"],
                "Correction_Amount_m": transition["correction_amount_m"],
                "Raw_Target_Clipped": not np.isclose(raw, transition["bounded_target_level"], rtol=0, atol=1e-8),
                "Out_Of_Training_Level_Domain": bool(outside[0]),
                "Out_Of_Training_Inflow_Domain": bool(outside[1]),
                "Water_Balance_Error_m3": transition["water_balance_error_m3"],
                "Storage_Jump_m3": jump,
                "Max_Total_Release_Triggered": transition["max_total_release_triggered"],
                "Unresolved_Upper_Storage_Excess_m3": transition["unresolved_upper_storage_excess_m3"],
            })
            current_level = transition["end_level"]
            previous_end_storage = transition["end_storage"]
    schedule = pd.DataFrame(rows)
    summary = summarize_single_reservoir_trajectory(
        schedule, firm_power_kw=reservoir.firm_power, method="ISO-RF"
    )
    objective = float(schedule["Penalized_Objective_GWh"].sum())
    summary.update({
        "lambda": float(penalty),
        "total_penalized_objective_gwh": objective,
        "mean_annual_penalized_objective_gwh": objective / len(years),
        "total_failure_penalty_gwh": float(schedule["Failure_Penalty_GWh"].sum()),
        "target_repair_count": int(schedule["Was_Corrected"].sum()),
        "target_clip_count": int(schedule["Raw_Target_Clipped"].sum()),
        "out_of_domain_count": int((schedule["Out_Of_Training_Level_Domain"] |
                                    schedule["Out_Of_Training_Inflow_Domain"]).sum()),
        "max_abs_water_balance_error_m3": float(schedule["Water_Balance_Error_m3"].abs().max()),
        "max_abs_storage_jump_m3": float(schedule["Storage_Jump_m3"].abs().max()),
        "max_total_release_trigger_count": int(schedule["Max_Total_Release_Triggered"].sum()),
        "unresolved_upper_storage_excess_count": int((schedule["Unresolved_Upper_Storage_Excess_m3"] > 0).sum()),
        "flood_limit_exceedance_count": int((schedule["Water_Level_m"] > schedule["Monthly_Max_Target_Level_m"] + 1e-8).sum()),
        "simulation_initial_level_m": reservoir.initial_level,
        "simulation_terminal_constraint": "none; final level reported as diagnostic",
        "continuous_across_calendar_years": True,
    })
    return schedule, summary


def segment_metrics(schedule: pd.DataFrame) -> list[dict]:
    """Four consecutive 20-year pieces of the SAME unbroken validation rollout."""
    rows = []
    for start in range(0, len(schedule), 240):
        frame = schedule.iloc[start:start + 240]
        failure = frame["Firm_Power_Violation"].to_numpy(bool)
        rows.append({
            "segment": len(rows) + 1,
            "first_year": int(frame["Calendar_Year"].iloc[0]),
            "last_year": int(frame["Calendar_Year"].iloc[-1]),
            "guarantee_rate": float(np.mean(~failure)),
            "mean_annual_generation_gwh": float(frame["Electricity_GWh"].sum()) * 12 / len(frame),
        })
    return rows


def simulation_metadata(ctx: Context, frame: pd.DataFrame, years: np.ndarray, split: str) -> dict:
    return {**forecast_metadata(ctx), "experiment_contract": ctx.contract,
            "data_split": split, "data_content_hash": content_hash(frame, years),
            "data_first_year": int(years[0]), "data_last_year": int(years[-1])}


def run_candidate(ctx: Context, teacher: pd.DataFrame, teacher_meta: dict,
                  parameters: dict, candidate_id: str, penalty: float, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    rf = ctx.config["random_forest"]
    expected = {
        "experiment_contract": ctx.contract,
        "artifact_role": "development_policy",
        "candidate_id": candidate_id,
        "lambda": float(penalty),
        "teacher_hash": teacher_meta["teacher_hash"],
        "teacher_split": "development_train_320y",
        "hyperparameters": validate_hyperparameters(parameters),
        "random_seed": int(rf["random_seed"]),
    }
    csv_path, meta_path, model_path = (
        directory / "validation_trajectory.csv", directory / "validation_summary.json",
        directory / "policy.pkl",
    )
    if meta_path.is_file() and csv_path.is_file() and model_path.is_file():
        saved = read_json(meta_path)
        if saved.get("candidate_contract") != expected:
            raise ValueError(f"Existing candidate has a different contract: {directory}")
        if saved["trajectory_hash"] != sha256_file(csv_path) or saved["policy_hash"] != sha256_file(model_path):
            raise ValueError(f"Candidate artifact hash mismatch: {directory}")
        print(f"Reusing complete validation: {candidate_id}, lambda={penalty:g}", flush=True)
        return saved
    started = time.perf_counter()
    policy = MonthlySingleRFPolicy.fit(
        teacher, hyperparameters=parameters, random_seed=rf["random_seed"],
        fit_n_jobs=rf["fit_n_jobs"], predict_n_jobs=rf["predict_n_jobs"],
        metadata={**forecast_metadata(ctx), **expected},
    )
    policy.save(model_path)
    schedule, summary = simulate_policy(policy, ctx.validation, ctx.validation_years,
                                        ctx.reservoir, penalty)
    write_csv(csv_path, schedule)
    segments = segment_metrics(schedule)
    write_csv(directory / "validation_segments.csv", pd.DataFrame(segments))
    summary.update({
        **simulation_metadata(ctx, ctx.validation, ctx.validation_years, "validation"),
        "candidate_id": candidate_id, "hyperparameters": parameters,
        "candidate_contract": expected,
        "policy_hash": sha256_file(model_path), "trajectory_hash": sha256_file(csv_path),
        "validation_segments": segments,
        "worst_segment_guarantee_rate": min(row["guarantee_rate"] for row in segments),
        "elapsed_seconds": time.perf_counter() - started,
    })
    write_json(meta_path, summary)
    print(f"VALIDATION_OK {candidate_id}: lambda={penalty:g}, "
          f"generation={summary['mean_annual_generation_gwh']:.3f}, "
          f"guarantee={100 * summary['guarantee_rate']:.3f}%, "
          f"score={summary['mean_annual_penalized_objective_gwh']:.3f}", flush=True)
    return summary


def validation_table_row(summary: dict) -> dict:
    return {key: summary[key] for key in (
        "candidate_id", "lambda", "mean_annual_generation_gwh", "guarantee_rate",
        "violation_month_count", "mean_annual_penalized_objective_gwh",
        "total_deficit_energy_gwh", "total_spill_volume_m3", "target_repair_count",
        "worst_segment_guarantee_rate", "elapsed_seconds",
    )} | {"hyperparameters": json.dumps(summary["hyperparameters"], sort_keys=True)}


def rank_hyperparameters(table: pd.DataFrame) -> pd.DataFrame:
    return table.sort_values(["mean_annual_penalized_objective_gwh", "candidate_id"],
                             ascending=[False, True], kind="stable").reset_index(drop=True)


def rank_lambdas(table: pd.DataFrame, target: float) -> pd.DataFrame:
    result = table.copy()
    result["distance_to_target_pct"] = (result["guarantee_rate"] - target).abs() * 100
    return result.sort_values(
        ["distance_to_target_pct", "mean_annual_generation_gwh", "lambda"],
        ascending=[True, False, True], kind="stable",
    ).reset_index(drop=True)


def run_development(ctx: Context):
    penalty = float(ctx.config["protocol"]["anchor_lambda"])
    root = ctx.output / "development" / f"lambda_{format_lambda_for_filename(penalty)}"
    rf = ctx.config["random_forest"]
    contract = {"experiment_contract": ctx.contract, "anchor_lambda": penalty,
                "random_forest": rf, "selection_metric": "mean_annual_penalized_objective_gwh"}
    contract_path = root / "search_contract.json"
    if contract_path.exists() and read_json(contract_path) != contract:
        raise ValueError("Development search contract changed; use a fresh --output-root")
    write_json(contract_path, contract)
    write_json(root / "config_snapshot.json", ctx.config)
    builder = TeacherBuilder(ctx)
    rows = []
    try:
        teacher, teacher_meta = builder.get(penalty)
        for candidate in rf["candidates"]:
            candidate_id = candidate["id"]
            parameters = {key: value for key, value in candidate.items() if key != "id"}
            print(f"DEVELOPMENT {len(rows)+1}/{len(rf['candidates'])}: {candidate_id}", flush=True)
            summary = run_candidate(ctx, teacher, teacher_meta, parameters, candidate_id,
                                    penalty, root / "candidates" / candidate_id)
            rows.append(validation_table_row(summary))
            write_csv(root / "development_results.csv", rank_hyperparameters(pd.DataFrame(rows)))
    finally:
        builder.close()
    ranked = rank_hyperparameters(pd.DataFrame(rows))
    best = ranked.iloc[0]
    selected = {"experiment_contract": ctx.contract,
                "selection_stage": "anchor_hyperparameters",
                "selection_data": "held_out_last_80_training_years",
                "selection_metric": "mean_annual_penalized_objective_gwh",
                "anchor_lambda": penalty, "candidate_id": str(best["candidate_id"]),
                "hyperparameters": json.loads(best["hyperparameters"]),
                "random_seed": int(rf["random_seed"]),
                "score": float(best["mean_annual_penalized_objective_gwh"]),
                "guarantee_rate": float(best["guarantee_rate"]),
                "mean_annual_generation_gwh": float(best["mean_annual_generation_gwh"])}
    write_json(ctx.output / "selection" / "selected_hyperparameters.json", selected)
    print(ranked.to_string(index=False), flush=True)
    print("ISO_RF_DEVELOPMENT_OK", flush=True)


def selected_hyperparameters(ctx: Context) -> dict:
    path = ctx.output / "selection" / "selected_hyperparameters.json"
    selected = read_json(path)
    ctx.require_record(selected)
    if selected["anchor_lambda"] != ctx.config["protocol"]["anchor_lambda"]:
        raise ValueError("Anchor lambda differs from the saved selection")
    if selected["random_seed"] != ctx.config["random_forest"]["random_seed"]:
        raise ValueError("RF seed differs from the frozen selection")
    validate_hyperparameters(selected["hyperparameters"])
    return selected


def run_lambda_scan(ctx: Context):
    selected = selected_hyperparameters(ctx)
    root = ctx.output / "lambda_scan"
    root.mkdir(parents=True, exist_ok=True)
    contract = {"experiment_contract": ctx.contract,
                "hyperparameters": selected["hyperparameters"],
                "random_seed": selected["random_seed"]}
    contract_path = root / "scan_contract.json"
    if contract_path.exists() and read_json(contract_path) != contract:
        raise ValueError("Frozen hyperparameters changed; use a new lambda scan output root")
    write_json(contract_path, contract)
    builder = TeacherBuilder(ctx)
    rows = []
    try:
        for penalty in ctx.config["protocol"]["lambda_candidates"]:
            penalty = float(penalty)
            tag = format_lambda_for_filename(penalty)
            if penalty == float(selected["anchor_lambda"]):
                # Reuse the winning anchor candidate, rather than fitting it twice.
                summary = read_json(
                    ctx.output / "development" / f"lambda_{tag}" / "candidates"
                    / selected["candidate_id"] / "validation_summary.json"
                )
                ctx.require_record(summary)
                teacher, teacher_meta = builder.get(penalty)
                summary = run_candidate(ctx, teacher, teacher_meta, selected["hyperparameters"],
                                        selected["candidate_id"], penalty,
                                        ctx.output / "development" / f"lambda_{tag}" / "candidates"
                                        / selected["candidate_id"])
            else:
                teacher, teacher_meta = builder.get(penalty)
                summary = run_candidate(ctx, teacher, teacher_meta, selected["hyperparameters"],
                                        "frozen_rf", penalty, root / f"lambda_{tag}")
            rows.append(validation_table_row(summary))
            write_csv(root / "lambda_scan_results.csv", pd.DataFrame(rows).sort_values("lambda"))
    finally:
        builder.close()
    print(pd.DataFrame(rows).sort_values("lambda").to_string(index=False), flush=True)
    print("ISO_RF_LAMBDA_SCAN_OK", flush=True)


def select_lambda(ctx: Context, *, target: float | None = None):
    selected = selected_hyperparameters(ctx)
    target = float(ctx.config["protocol"]["lambda_target_guarantee_rate"] if target is None else target)
    if not np.isfinite(target) or not 0 <= target <= 1:
        raise ValueError("Selection target must be in [0,1]")
    scan_root = ctx.output / "lambda_scan"
    contract = read_json(scan_root / "scan_contract.json")
    ctx.require_record(contract)
    if contract["hyperparameters"] != selected["hyperparameters"]:
        raise ValueError("Lambda scan did not use the frozen RF hyperparameters")
    table_path = scan_root / "lambda_scan_results.csv"
    table = pd.read_csv(table_path)
    expected = sorted(float(value) for value in ctx.config["protocol"]["lambda_candidates"])
    if sorted(table["lambda"].tolist()) != expected:
        raise ValueError("Lambda scan is incomplete for the configured candidate list")
    ranked = rank_lambdas(table, target)
    best = ranked.iloc[0]
    result = {"experiment_contract": ctx.contract,
              "selection_stage": "lambda_on_validation",
              "selection_data": "held_out_last_80_training_years",
              "selection_rule": "nearest_guarantee_target_then_higher_generation_then_lower_lambda",
              "target_guarantee_rate": target,
              "lambda": float(best["lambda"]),
              "hyperparameters": selected["hyperparameters"],
              "random_seed": selected["random_seed"],
              "anchor_lambda": selected["anchor_lambda"],
              "validation_guarantee_rate": float(best["guarantee_rate"]),
              "validation_mean_annual_generation_gwh": float(best["mean_annual_generation_gwh"]),
              "validation_scan_hash": sha256_file(table_path),
              "hyperparameter_selection_hash": sha256_file(ctx.output / "selection" / "selected_hyperparameters.json")}
    write_csv(scan_root / "lambda_selection_ranking.csv", ranked)
    write_json(ctx.output / "selection" / "selected_lambda.json", result)
    print(ranked.to_string(index=False), flush=True)
    print(f"SELECTED_LAMBDA={result['lambda']:g}", flush=True)
    print("ISO_RF_LAMBDA_SELECTION_OK", flush=True)


def frozen_selection(ctx: Context) -> dict:
    path = ctx.output / "selection" / "selected_lambda.json"
    selection = read_json(path)
    ctx.require_record(selection)
    hp = selected_hyperparameters(ctx)
    if selection["hyperparameters"] != hp["hyperparameters"] or selection["random_seed"] != hp["random_seed"]:
        raise ValueError("Frozen hyperparameters differ from selected lambda metadata")
    if selection["hyperparameter_selection_hash"] != sha256_file(ctx.output / "selection" / "selected_hyperparameters.json"):
        raise ValueError("Hyperparameter selection was changed after lambda selection")
    if selection["validation_scan_hash"] != sha256_file(ctx.output / "lambda_scan" / "lambda_scan_results.csv"):
        raise ValueError("Lambda scan was changed after selection; repeat select-lambda")
    return selection


def run_final_training(ctx: Context):
    selected = frozen_selection(ctx)
    penalty = selected["lambda"]
    tag = format_lambda_for_filename(penalty)
    root = ctx.output / "final" / f"lambda_{tag}"
    model_path = root / "iso_rf_final_policy.pkl"
    meta_path = root / "iso_rf_final_policy.json"
    selection_hash = sha256_file(ctx.output / "selection" / "selected_lambda.json")
    if model_path.exists() or meta_path.exists():
        if not model_path.is_file() or not meta_path.is_file():
            raise ValueError("Incomplete final artifact; use a fresh output directory")
        metadata = read_json(meta_path)
        ctx.require_record(metadata)
        if metadata["selection_hash"] != selection_hash or metadata["policy_hash"] != sha256_file(model_path):
            raise ValueError("Final artifact does not match frozen selection")
        print("Reusing already fitted final policy", flush=True)
        return
    builder = TeacherBuilder(ctx, final=True)
    try:
        teacher, teacher_meta = builder.get(penalty)
        rf = ctx.config["random_forest"]
        metadata = {**forecast_metadata(ctx), "experiment_contract": ctx.contract,
                    "artifact_role": "final_policy", "fit_years": 400,
                    "lambda": penalty, "teacher_hash": teacher_meta["teacher_hash"],
                    "teacher_split": "full_train_400y", "selection_hash": selection_hash}
        policy = MonthlySingleRFPolicy.fit(
            teacher, hyperparameters=selected["hyperparameters"],
            random_seed=selected["random_seed"], fit_n_jobs=rf["fit_n_jobs"],
            predict_n_jobs=rf["predict_n_jobs"], metadata=metadata,
        )
        policy.save(model_path)
        write_json(meta_path, {**policy.metadata, "policy_hash": sha256_file(model_path)})
    finally:
        builder.close()
    print(f"FINAL_POLICY={model_path}", flush=True)
    print("ISO_RF_FINAL_TRAINING_OK", flush=True)


def run_evaluation(ctx: Context):
    selected = frozen_selection(ctx)
    tag = format_lambda_for_filename(selected["lambda"])
    model_root = ctx.output / "final" / f"lambda_{tag}"
    model_path = model_root / "iso_rf_final_policy.pkl"
    model_meta = read_json(model_root / "iso_rf_final_policy.json")
    ctx.require_record(model_meta)
    selection_hash = sha256_file(ctx.output / "selection" / "selected_lambda.json")
    if model_meta["selection_hash"] != selection_hash or model_meta["policy_hash"] != sha256_file(model_path):
        raise ValueError("Final policy hash/selection does not match evaluation request")
    policy = MonthlySingleRFPolicy.load(model_path, required_role="final_policy")
    ctx.require_record(policy.metadata)
    if policy.metadata["lambda"] != selected["lambda"] or policy.hyperparameters != selected["hyperparameters"]:
        raise ValueError("Final model parameters differ from frozen selection")
    # The independent workbook is first opened here, after checking final provenance.
    path = ctx.data / "SynMonthlyInflow_evaluation.xlsx"
    frame, years = load_monthly_inflows_with_calendar(path)
    split_training(frame, years)  # validate shape/calendar; no fitting on these data.
    schedule, summary = simulate_policy(policy, frame, years, ctx.reservoir, selected["lambda"])
    output = ctx.output / "evaluation"
    csv_path = output / f"iso_rf_eval_lambda_{tag}.csv"
    write_csv(csv_path, schedule)
    summary.update({
        **simulation_metadata(ctx, frame, years, "evaluation"),
        "data_hash": sha256_file(path), "evaluation_split": "independent_400_year_evaluation",
        "artifact_role": "final_evaluation", "policy_hash": sha256_file(model_path),
        "hyperparameters": selected["hyperparameters"], "random_seed": selected["random_seed"],
        "selection_hash": selection_hash,
        "evaluation_used_for_selection": False,
        "lambda_selection_data": "validation_80y",
        "trajectory_hash": sha256_file(csv_path),
    })
    write_json(csv_path.with_suffix(".json"), summary)
    write_csv(output / "iso_rf_evaluation_summary.csv", pd.DataFrame([{
        "lambda": selected["lambda"], "annual_generation_gwh": summary["mean_annual_generation_gwh"],
        "guarantee_rate_pct": 100 * summary["guarantee_rate"],
        "violation_month_count": summary["violation_month_count"],
        "selection_basis": "validation_selected_lambda",
    }]))
    print(f"EVALUATION_CSV={csv_path}", flush=True)
    print(f"generation={summary['mean_annual_generation_gwh']:.3f} GWh/yr, "
          f"guarantee={100 * summary['guarantee_rate']:.3f}%", flush=True)
    print("ISO_RF_INDEPENDENT_EVALUATION_OK", flush=True)


def preflight(ctx: Context):
    """Inspect training data/configuration only, without generating any teacher/model."""
    config = ctx.config
    print(f"OUTPUT_ROOT={ctx.output}")
    print(f"ANCHOR_LAMBDA={config['protocol']['anchor_lambda']:g}")
    print(f"DEVELOPMENT_TRAIN={ctx.fit_years[0]}..{ctx.fit_years[-1]}, months=3840")
    print(f"VALIDATION={ctx.validation_years[0]}..{ctx.validation_years[-1]}, months=960")
    print(f"RF_CANDIDATES={len(config['random_forest']['candidates'])}")
    print(f"RF_SEED={config['random_forest']['random_seed']}")
    print(f"LAMBDA_CANDIDATES={config['protocol']['lambda_candidates']}")
    print("TEACHER_SCOPE=independently optimized first 320 years")
    print("INDEPENDENT_EVALUATION_OPENED=False")
    print("ISO_RF_PREFLIGHT_OK")
