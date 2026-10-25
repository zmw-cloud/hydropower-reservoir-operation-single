"""One-reservoir reduction of the V11 cyclic one-step-forecast SDP."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from reservoir_benchmark.shared.experiment_contract import (
    SDP_SOLVER_VARIANT, base_metadata, failure_indicator, require_contract,
)
from .sdp_probability import MonthlyInflowModel, N_INFLOW_CLASSES


def make_level_grid(dead: float, normal: float, step: float = 1.0):
    levels = np.arange(dead, normal + 1e-9, step, dtype=float)
    if not np.isclose(levels[-1], normal):
        levels = np.append(levels, normal)
    return levels


def nearest_index(grid, value):
    right = int(np.searchsorted(grid, value, side="left"))
    if right <= 0:
        return 0
    if right >= len(grid):
        return len(grid) - 1
    return right - 1 if value - grid[right - 1] <= grid[right] - value else right


@dataclass(frozen=True)
class SDPPolicy:
    levels: np.ndarray
    action_indices: np.ndarray  # [12, state, class]
    metadata: dict

    def action(self, month, current_level, inflow_class):
        if not 1 <= int(month) <= 12 or not 0 <= int(inflow_class) < 10:
            raise ValueError("Invalid SDP policy lookup key")
        state = nearest_index(self.levels, float(current_level))
        action = int(self.action_indices[int(month) - 1, state, int(inflow_class)])
        return float(self.levels[action]), state

    def save(self, path):
        require_contract(self.metadata, solver_variant=SDP_SOLVER_VARIANT)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, levels=self.levels,
                            action_indices=self.action_indices.astype(np.int16),
                            metadata_json=np.asarray(json.dumps(self.metadata, sort_keys=True)))
        return target

    @classmethod
    def load(cls, path, *, penalty_lambda=None):
        with np.load(path, allow_pickle=False) as saved:
            policy = cls(saved["levels"], saved["action_indices"],
                         json.loads(str(saved["metadata_json"].item())))
        require_contract(policy.metadata, penalty_lambda=penalty_lambda,
                         solver_variant=SDP_SOLVER_VARIANT)
        return policy


class ReservoirOperation:
    """Periodic relative-value iteration with current inflow class observed."""
    def __init__(self, reservoir, inflow_model: MonthlyInflowModel, *,
                 penalty_lambda: float, data_dir, cache_dir=None,
                 water_level_step=1.0, monthly_max_iterations=400,
                 tolerance_gwh=1e-5, stable_policy_iterations=2):
        inflow_model.validate()
        self.reservoir = reservoir
        self.inflow_model = inflow_model
        self.penalty_lambda = float(penalty_lambda)
        self.data_dir = Path(data_dir)
        self.levels = make_level_grid(reservoir.dead_level, reservoir.normal_level,
                                      float(water_level_step))
        self.storages = np.asarray([reservoir.calculate_storage(z) for z in self.levels])
        self.max_iterations = int(monthly_max_iterations)
        self.tolerance_gwh = float(tolerance_gwh)
        self.stable_policy_iterations = int(stable_policy_iterations)
        self.cache_dir = None if cache_dir is None else Path(cache_dir)

    def _transition_arrays(self, month_index, representative_inflow):
        model = self.reservoir
        days = float(self.inflow_model.mean_days[month_index])
        seconds = days * 86400.0
        max_level = model.monthly_max_target_level(month_index + 1)
        action_count = int(np.searchsorted(self.levels, max_level + 1e-8, side="right"))
        targets = self.levels[:action_count]
        requested_storage = self.storages[:action_count]
        start = self.storages[:, None]
        available = start + float(representative_inflow) * seconds / 1e9
        end_storage = np.minimum(np.minimum(requested_storage[None, :], available),
                                 model.calculate_storage(max_level))
        end_storage = np.maximum(end_storage, model.calculate_storage(model.dead_level))
        release = np.maximum(0.0, (start + float(representative_inflow) * seconds / 1e9
                                   - end_storage) * 1e9 / seconds)
        capped = release > model.max_total_release
        release = np.minimum(release, model.max_total_release)
        capped_storage = np.clip(start + (float(representative_inflow) - release) * seconds / 1e9,
                                 model.calculate_storage(model.dead_level),
                                 model.calculate_storage(max_level))
        end_storage = np.where(capped, capped_storage, end_storage)
        end_level = np.interp(end_storage, model._storages_s, model._levels_s)
        tailwater = np.polyval(model.tailwater_curve_coefficients, release)
        head = np.maximum(0.0, (self.levels[:, None] + end_level) / 2.0 - tailwater)
        turbine = np.minimum(release, model.max_turbine_discharge)
        capacity_flow = np.divide(model.max_power, model.power_coefficient * head,
                                  out=np.full_like(head, np.inf), where=head > 0.0)
        turbine = np.where(head <= 0.0, 0.0, np.minimum(turbine, capacity_flow))
        power = np.minimum(model.max_power, model.power_coefficient * turbine * head)
        energy = power * days * 24.0 / 1e6
        violation = failure_indicator(power, model.firm_power)
        stage = energy - self.penalty_lambda * violation
        corrected = ~np.isclose(end_level, targets[None, :], rtol=0, atol=1e-8)
        balance_error = ((end_storage - start) * 1e9
                         - (float(representative_inflow) - release) * seconds)
        return stage, end_storage, end_level, corrected, balance_error, action_count

    def _bellman_month(self, month_index, future_expected):
        n = len(self.levels)
        values = np.empty((n, N_INFLOW_CLASSES), dtype=float)
        actions = np.empty((n, N_INFLOW_CLASSES), dtype=np.int16)
        max_balance = 0.0
        correction_count = 0
        for inflow_class in range(N_INFLOW_CLASSES):
            stage, end_storage, _, corrected, balance, action_count = self._transition_arrays(
                month_index, self.inflow_model.representatives[month_index, inflow_class])
            continuation = np.interp(end_storage, self.storages, future_expected)
            q = stage + continuation
            best = np.argmax(q, axis=1)
            values[:, inflow_class] = q[np.arange(n), best]
            actions[:, inflow_class] = best.astype(np.int16)
            max_balance = max(max_balance, float(np.max(np.abs(balance))))
            correction_count += int(corrected[np.arange(n), best].sum())
        return values, actions, max_balance, correction_count

    def optimize(self, progress=True):
        n = len(self.levels)
        january_relative = np.zeros(n, dtype=float)
        previous_actions = None
        stable_count = 0
        actions = np.zeros((12, n, N_INFLOW_CLASSES), dtype=np.int16)
        residual = np.inf
        max_balance = 0.0
        corrections = 0
        anchor = n - 1
        for iteration in range(1, self.max_iterations + 1):
            future = january_relative
            iteration_corrections = 0
            for month_index in range(11, -1, -1):
                class_values, actions[month_index], balance, count = self._bellman_month(
                    month_index, future)
                max_balance = max(max_balance, balance)
                iteration_corrections += count
                future = class_values @ self.inflow_model.probabilities[month_index]
            annual_gain = float(future[anchor])
            updated = future - annual_gain
            residual = float(np.max(np.abs(updated - january_relative)))
            same_policy = previous_actions is not None and np.array_equal(actions, previous_actions)
            stable_count = stable_count + 1 if same_policy else 0
            january_relative = updated
            previous_actions = actions.copy()
            corrections = iteration_corrections
            if progress:
                print(f"SDP iteration {iteration}: residual={residual:.6g} GWh, "
                      f"stable_policy_passes={stable_count}", flush=True)
            if stable_count >= self.stable_policy_iterations:
                break
        else:
            raise RuntimeError("SDP did not reach the configured policy-stability criterion")
        metadata = base_metadata(data_dir=self.data_dir, split="train",
                                 penalty_lambda=self.penalty_lambda, method="SDP",
                                 solver_variant=SDP_SOLVER_VARIANT, grid_step=1.0)
        metadata.update({
            "information_set": "month_level_current_inflow_class_one_step_perfect_forecast",
            "n_inflow_classes": N_INFLOW_CLASSES,
            "classification": self.inflow_model.metadata["classification"],
            "probability_definition": self.inflow_model.metadata["probability_definition"],
            "representative_inflow": self.inflow_model.metadata["representative_inflow"],
            "continuation_value_lookup": "linear_interpolation_at_executed_continuous_storage",
            "policy_state_lookup": "nearest_level_grid_point",
            "action_semantics": "requested_target_with_physical_repair",
            "solver": "cyclic_relative_value_iteration",
            "iterations": iteration,
            "relative_value_residual_gwh": residual,
            "relative_value_residual_within_tolerance": residual <= self.tolerance_gwh,
            "stable_policy_passes": stable_count,
            "selected_action_repair_count_last_iteration": corrections,
            "max_abs_training_water_balance_error_m3": max_balance,
            "training_inflow_model": self.inflow_model.metadata,
        })
        return SDPPolicy(self.levels.copy(), actions.copy(), metadata)
