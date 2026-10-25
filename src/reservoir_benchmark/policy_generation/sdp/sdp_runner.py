"""Train formal one-dimensional V11-style SDP policies."""
from __future__ import annotations

from reservoir_benchmark.shared.io_utils import (
    load_algorithm_settings, load_formal_penalty_candidates,
    load_monthly_inflows_with_calendar,
)
from reservoir_benchmark.shared.project_paths import get_data_dir, get_formal_algorithm_dir, format_lambda_for_filename
from reservoir_benchmark.shared.reservoir_model import SharedReservoirModel
from .sdp_probability import MonthlyInflowModel, N_INFLOW_CLASSES
from .sdp_solver import ReservoirOperation


def main_sdp():
    data_dir = get_data_dir()
    settings = load_algorithm_settings(data_dir / "experiment_settings.yaml", "sdp")
    if int(settings["n_inflow_classes"]) != N_INFLOW_CLASSES:
        raise ValueError("Formal SDP requires 10 V11-style equal-width inflow classes")
    frame, years = load_monthly_inflows_with_calendar(data_dir / "SynMonthlyInflow_train.xlsx")
    model = MonthlyInflowModel.fit(frame.to_numpy(dtype=float), years)
    root = get_formal_algorithm_dir("sdp")
    probability_path = root / "training" / "monthly_inflow_model.npz"
    model.save(probability_path)
    penalties = load_formal_penalty_candidates(data_dir, "sdp")
    reservoir = SharedReservoirModel(data_dir=data_dir)
    solver_cfg = settings["solver"]
    for penalty in penalties:
        solver = ReservoirOperation(
            reservoir, model, penalty_lambda=penalty, data_dir=data_dir,
            water_level_step=float(settings["water_level_step"]),
            monthly_max_iterations=int(solver_cfg["max_iterations"]),
            tolerance_gwh=float(solver_cfg["tolerance_gwh"]),
            stable_policy_iterations=int(solver_cfg["stable_policy_iterations"]),
        )
        policy = solver.optimize(progress=True)
        tag = format_lambda_for_filename(penalty)
        path = root / "policy" / f"sdp_policy_lambda_{tag}.npz"
        policy.save(path)
        print(f"Saved formal SDP policy: {path}")


if __name__ == "__main__":
    main_sdp()
