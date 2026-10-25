"""
Deterministic dynamic programming solver module for the unified reservoir
operation framework.

This module implements the core deterministic dynamic programming (DDP)
optimization logic for single-reservoir operation, including:
1) one-step reservoir transition calculation,
2) hydraulic head and hydropower computation under shared reservoir physics,
3) soft penalty treatment for firm-power deficit,
4) backward Bellman recursion over discretized water levels, and
5) forward simulation of the optimal trajectory.

The purpose of this module is to isolate the algorithmic core of DDP from
project-specific path handling, file input/output, and script-level execution,
so that the solver remains focused on optimization logic and can be reused
cleanly within the unified framework.
"""

from __future__ import annotations

import numpy as np

from .ddp_acceleration import TransitionCache, cache_signature
from reservoir_benchmark.shared.experiment_contract import (
    failure_indicator,
)


class ReservoirOperation:
    """
    Deterministic dynamic programming (DDP) for single-reservoir operation.

    This solver only contains the DDP logic itself.
    It does not handle project paths, file reading, or output writing.
    """

    def __init__(
        self,
        reservoir,
        inflows: np.ndarray,
        month_days: np.ndarray | None = None,
    ):
        self._transition_cache = None
        self.reservoir = reservoir
        self.inflows = inflows  # shape: [years, 12]
        self.total_years = len(self.inflows)
        if month_days is None:
            month_days = np.full_like(
                self.inflows,
                self.reservoir.delta_t_days,
                dtype=float,
            )
        self.month_days = np.asarray(month_days, dtype=float)
        if self.month_days.shape != self.inflows.shape:
            raise ValueError("month_days must have the same [years, 12] shape as inflows")
        if np.any(~np.isfinite(self.month_days)) or np.any(self.month_days <= 0.0):
            raise ValueError("month_days must contain positive finite values")

        # Shared parameters from SharedReservoirModel
        self.dead_level = self.reservoir.dead_level
        self.normal_level = self.reservoir.normal_level
        self.initial_level = self.reservoir.initial_level
        self.max_turbine_discharge = self.reservoir.max_turbine_discharge
        self.max_power = self.reservoir.max_power
        self.firm_power = self.reservoir.firm_power
        self.power_coefficient = self.reservoir.power_coefficient
        self.delta_t_days = self.reservoir.delta_t_days

        # Time settings
        self.months_per_year = 12

        # State discretization
        self.levels = np.arange(self.dead_level, self.normal_level + 1, 1)
        self.volumes = np.array([self.reservoir.calculate_storage(h) for h in self.levels])

    def _calc_penalized_energy_gwh(
        self,
        power_kw: float,
        firm_power_penalty_lambda: float,
        duration_days: float,
    ) -> float:
        """
        Compute generation minus one fixed penalty for a failed month.
        """
        hours = float(duration_days) * 24.0
        energy_gwh = power_kw * hours / 1e6

        if bool(failure_indicator(power_kw, self.firm_power)):
            energy_gwh -= float(firm_power_penalty_lambda)

        return energy_gwh

    def _calculate_average_head(
        self,
        start_level: float,
        end_level: float,
        total_discharge: float,
    ) -> float:
        """
        Compute average hydraulic head (m).
        """
        average_upstream_level = (start_level + end_level) / 2.0
        tailwater_level = self.reservoir.calculate_tailwater(total_discharge)
        return max(average_upstream_level - tailwater_level, 0.0)

    def calculate_operation(
        self,
        current_level: float,
        target_level: float,
        inflow: float,
        duration_days: float,
        month: int,
    ):
        """
        Compute one-step reservoir operation.

        Returns:
            (
                power_kw,
                total_discharge_m3s,
                turbine_discharge_m3s,
                spill_discharge_m3s,
                final_level_m,
                feasible,
                current_storage_1e9_m3,
            )
        """
        transition = self.reservoir.calculate_transition(
            current_level,
            target_level,
            inflow,
            duration_days,
            month=month,
        )

        # Return one-step operation results for DDP recursion or forward simulation.
        return (
            transition["power_kw"],
            transition["total_discharge"],
            transition["turbine_discharge"],
            transition["spill_discharge"],
            transition["end_level"],
            transition["target_reached"],
            transition["current_storage"],
        )

    def clear_transition_cache(self):
        """Release temporary disk storage; subsequent optimize rebuilds it."""
        if self._transition_cache is not None:
            self._transition_cache.close()
            self._transition_cache = None

    def optimize(self, firm_power_penalty_lambda: float):
        """
        Run deterministic dynamic programming.

        Returns:
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
                inflows,
            )
        """
        # Total number of monthly stages over the full deterministic horizon.
        months_per_year = self.total_years * self.months_per_year
        n_states = len(self.levels)

        # Flatten the inflow matrix [years, 12] into one deterministic monthly
        # inflow sequence so the DDP can be solved over a single finite horizon.
        inflows = self.inflows.flatten()
        month_days = self.month_days.flatten()

        # value_function[t, state] stores the maximum cumulative value from stage t
        # to the end of the horizon, starting from the given water-level state.
        #
        # Initialize all stages as very negative values to represent infeasible or
        # not-yet-updated states, and set the terminal value to zero.
        value_function = np.full((months_per_year + 1, n_states), -np.inf)
        terminal_state = int(
            np.argmin(np.abs(self.levels - self.reservoir.terminal_level))
        )
        if not np.isclose(
            self.levels[terminal_state],
            self.reservoir.terminal_level,
            rtol=0.0,
            atol=1e-8,
        ):
            raise ValueError("terminal_level must lie on the 1 m DDP grid")
        value_function[months_per_year, terminal_state] = 0.0

        # decisions[t, state] stores the optimal next water level selected at stage t
        # when the current state is the given water-level state.
        # Integer metre targets preserve the existing return semantics, at 1/4
        # the previous float64 decision-table size for XiaoWan.
        decision_dtype = np.min_scalar_type(int(max(abs(self.dead_level), abs(self.normal_level))))
        if self.dead_level < 0:
            decision_dtype = np.int32
        decisions = np.zeros((months_per_year, n_states), dtype=decision_dtype)

        signature = cache_signature(self.reservoir, inflows, month_days, self.levels)
        if self._transition_cache is None or self._transition_cache.signature != signature:
            self.clear_transition_cache()
            self._transition_cache = TransitionCache(
                self.reservoir, inflows, month_days, self.levels)
        else:
            print("DDP reusing transition cache across lambda values", flush=True)

        # Stages remain sequential. All 75 x 75 candidates within a stage are
        # evaluated in NumPy. argmax returns the first maximum, matching the
        # original ascending-action loop and strict '>' tie-breaking.
        for t in reversed(range(months_per_year)):
            power = self._transition_cache.power[t]
            hours = month_days[t] * 24.0
            candidates = power * hours / 1e6
            candidates -= float(firm_power_penalty_lambda) * failure_indicator(
                power, self.firm_power
            )
            future = value_function[t + 1]
            candidates += future[None, :]
            valid = np.isfinite(power) & np.isfinite(future[None, :])
            candidates[~valid] = -np.inf
            if not np.any(valid):
                raise RuntimeError(f"DDP month {t} has no feasible state-action continuation")
            best = np.argmax(candidates, axis=1)
            best_value = candidates[np.arange(n_states), best]
            reachable = np.isfinite(best_value)
            value_function[t, reachable] = best_value[reachable]
            decisions[t] = np.where(reachable, self.levels[best], self.levels)
            if (months_per_year - t) % 600 == 0 or t == 0:
                print(f"DDP Bellman solved {months_per_year - t}/{months_per_year} months", flush=True)

        # Forward simulation of the optimal trajectory:
        # starting from the initial level, follow the stored optimal decisions
        # stage by stage and record the resulting operation sequence.
        optimal_levels = []
        optimal_storages = []
        optimal_powers = []
        optimal_electricity = []
        optimal_discharges = []
        optimal_turbine_flows = []
        optimal_spill_flows = []
        optimal_deficit_energy = []
        optimal_duration_days = []
        optimal_monthly_max_levels = []

        # Find the discrete state closest to the prescribed initial water level.
        current_state = np.argmin(np.abs(self.levels - self.initial_level))
        if not np.isfinite(value_function[0, current_state]):
            raise RuntimeError("DDP has no feasible path from the configured initial level")
        current_level = self.levels[current_state]

        for t in range(months_per_year):
            month = t % self.months_per_year + 1
            next_level = decisions[t, current_state]

            (
                power_kw,
                total_discharge,
                turbine_discharge,
                spill_discharge,
                final_level,
                feasible,
                storage_current,
            ) = self.calculate_operation(
                current_level,
                next_level,
                inflows[t],
                month_days[t],
                month,
            )
            if not feasible or not np.isclose(final_level, next_level, rtol=0.0, atol=1e-8):
                raise RuntimeError("DDP selected transition did not reach its exact 1 m target grid")

            # Save end-of-period state for standardized outputs.
            storage_final = self.reservoir.calculate_storage(final_level)

            optimal_levels.append(final_level)
            optimal_storages.append(storage_final)

            hours = month_days[t] * 24.0
            optimal_powers.append(power_kw)
            optimal_electricity.append(power_kw * hours / 1e6)
            optimal_deficit_energy.append(
                max(0.0, self.firm_power - power_kw) * hours / 1e6
            )
            optimal_duration_days.append(month_days[t])
            optimal_monthly_max_levels.append(
                self.reservoir.monthly_max_target_level(month)
            )
            optimal_discharges.append(total_discharge)
            optimal_turbine_flows.append(turbine_discharge)
            optimal_spill_flows.append(spill_discharge)

            current_level = final_level
            current_state = np.argmin(np.abs(self.levels - current_level))

        if not np.isclose(optimal_levels[-1], self.reservoir.terminal_level, rtol=0.0, atol=1e-8):
            raise RuntimeError("DDP forward trajectory violated the hard terminal level")

        # Return both the DDP solution objects and the forward-simulated optimal
        # trajectory for downstream saving, evaluation, or plotting.
        return (
            value_function,
            decisions,
            optimal_levels,
            optimal_storages,
            optimal_powers,
            optimal_electricity,
            optimal_discharges,
            optimal_turbine_flows,
            optimal_spill_flows,
            inflows,
            optimal_deficit_energy,
            optimal_duration_days,
            optimal_monthly_max_levels,
        )
