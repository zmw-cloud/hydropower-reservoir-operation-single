"""Machine-checkable contract for the formal violation-count experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


OBJECTIVE_VERSION = "single_violation_count_v1"
PENALTY_DEFINITION = "generation_gwh_minus_lambda_per_failed_reservoir_month"
PENALTY_UNIT = "GWh_per_failed_reservoir_month"
STRICT_FAILURE_COMPARISON = "power_kw < firm_power_kw"
EXPERIMENT_ID = "xiaowan_single_violation_count_v1"
CODE_VERSION = "single_comparison_formal_objective_v1"
DDP_SOLVER_VARIANT = "exact_grid_continuation_vectorized_transition_cache_v1"
SDP_SOLVER_VARIANT = "one_dimensional_linear_continuation_relative_value_iteration_v1"


def failure_indicator(power_kw, firm_power_kw: float):
    """Strict failure: equality with firm power is successful."""
    return np.asarray(power_kw, dtype=float) < float(firm_power_kw)


def penalized_generation_gwh(generation_gwh, power_kw, firm_power_kw, penalty_lambda):
    failures = failure_indicator(power_kw, firm_power_kw)
    return np.asarray(generation_gwh, dtype=float) - float(penalty_lambda) * failures


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_hash(data_dir: str | Path, split: str) -> str:
    if split not in {"train", "evaluation"}:
        raise ValueError("split must be train or evaluation")
    return sha256_file(Path(data_dir) / f"SynMonthlyInflow_{split}.xlsx")


def physics_hash(data_dir: str | Path) -> str:
    root = Path(data_dir)
    digest = hashlib.sha256()
    for name in ("reservoir_settings.txt", "level_storage_curve.csv", "tailwater_discharge_curve.csv"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def code_hash(*paths: str | Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(item).resolve() for item in paths), key=str):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def base_metadata(*, data_dir, split, penalty_lambda, method, solver_variant, grid_step=1.0) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "code_version": CODE_VERSION,
        "objective_version": OBJECTIVE_VERSION,
        "penalty_definition": PENALTY_DEFINITION,
        "penalty_unit": PENALTY_UNIT,
        "lambda": float(penalty_lambda),
        "strict_failure_comparison": STRICT_FAILURE_COMPARISON,
        "method": str(method),
        "solver_variant": str(solver_variant),
        "grid_step_m": float(grid_step),
        "data_split": str(split),
        "data_hash": data_hash(data_dir, split),
        "physics_hash": physics_hash(data_dir),
        "deficit_energy_role": "diagnostic_only_not_in_objective",
    }


def require_contract(metadata: dict, *, penalty_lambda: float | None = None, solver_variant: str | None = None):
    expected = {
        "objective_version": OBJECTIVE_VERSION,
        "penalty_definition": PENALTY_DEFINITION,
        "penalty_unit": PENALTY_UNIT,
        "strict_failure_comparison": STRICT_FAILURE_COMPARISON,
    }
    if not isinstance(metadata, dict):
        raise ValueError("Artifact metadata must be a dictionary")
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Artifact is not a compatible violation-count artifact: {key}")
    if penalty_lambda is not None and not np.isclose(float(metadata.get("lambda", np.nan)), float(penalty_lambda), rtol=0, atol=0):
        raise ValueError("Artifact lambda does not match the requested experiment")
    if solver_variant is not None and metadata.get("solver_variant") != solver_variant:
        raise ValueError("Artifact solver variant is incompatible")


def write_json(path: str | Path, payload: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return target
