"""
Shared DQN reservoir environment module for the unified reservoir operation
framework.

This module provides the common environment and state-transition logic used by
the DQN-based reservoir operation workflow, including:
1) storage-level conversion through the shared reservoir model,
2) tailwater calculation and hydraulic head evaluation,
3) month-by-month closed-loop reservoir simulation,
4) reward calculation based on electricity generation and firm-power penalty, and
5) a Gymnasium-compatible environment for DQN training and evaluation.

The purpose of this module is to keep the DQN environment consistent with the
shared reservoir physics, parameter definitions, and operating constraints used
by the other algorithms in the unified framework.
"""

from __future__ import annotations

import calendar
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box as GBox, Discrete as GDiscrete

from reservoir_benchmark.shared.reservoir_model import SharedReservoirModel


class ReservoirUtils:
    """
    Utility wrapper around the shared reservoir model.
    """

    def __init__(self, reservoir_model: SharedReservoirModel):
        self.reservoir = reservoir_model

    def level_to_storage(self, level: float) -> float:
        """
        Convert water level (m) to storage (1e9 m3).
        """
        return float(self.reservoir.calculate_storage(level))

    def storage_to_level(self, storage: float) -> float:
        """
        Convert storage (1e9 m3) to water level (m).
        """
        return float(self.reservoir.calculate_level(storage))

    def calculate_tailwater(self, discharge: float) -> float:
        """
        Calculate tailwater level (m) from discharge (m3/s).
        """
        return float(self.reservoir.calculate_tailwater(discharge))


class ReservoirEnv(gym.Env):
    """
    Gymnasium environment for single-reservoir DQN operation.

    State definition:
        [storage (1e9 m3), inflow (m3/s), month]

    Action definition:
        Discrete target end-of-month storage candidates.
    """

    metadata = {"render_modes": []}

    def __init__(self, basic_info, debug: bool = False,):
        super().__init__()

        self.debug = debug
        self.basic_info = basic_info
        self.max_steps = int(self.basic_info.config.train_config["max_steps"])

        self.reservoir_model = self.basic_info.reservoir_model
        self.utils = ReservoirUtils(self.reservoir_model)

        # Shared physical parameters
        self.dead_level = float(self.basic_info.dead_level)
        self.normal_level = float(self.basic_info.normal_level)
        self.initial_level = float(self.basic_info.initial_level)
        self.max_turbine_discharge = float(self.basic_info.max_turbine_discharge)
        self.max_power = float(self.basic_info.max_power)
        self.power_coefficient = float(self.basic_info.power_coefficient)
        self.firm_power = float(self.basic_info.firm_power)
        self.firm_penalty_lambda = float(self.basic_info.firm_penalty_lambda)
        self.inflow_upper_bound = float(self.basic_info.inflow_upper_bound)

        # Unified time-step parameters
        self.delta_t_days = float(self.basic_info.delta_t_days)
        self.delta_t_hours = float(self.basic_info.delta_t_hours)
        self.delta_t_seconds = float(self.basic_info.delta_t_seconds)
        self.step_hours = self.delta_t_hours

        # Action space: discrete target end-of-month storage candidates
        self.action_space = GDiscrete(len(self.basic_info.state_present_storage))

        # Storage bounds
        self.dead_storage = float(self.basic_info.dead_storage)
        self.normal_storage = float(self.basic_info.normal_storage)

        # Raw policy observation: [month, current level, current-month inflow].
        self.observation_space = GBox(
            low=np.array(
                [1.0, self.dead_level, self.basic_info.inflow_lower_bound],
                dtype=np.float64,
            ),
            high=np.array(
                [12.0, self.normal_level, self.inflow_upper_bound],
                dtype=np.float64,
            ),
            dtype=np.float64,
        )

        # Inflow data
        self.train_data = self.basic_info.train_inflow
        self.test_data = self.basic_info.test_inflow
        self.train_years = np.asarray(self.basic_info.train_years, dtype=int)
        self.test_years = (
            None
            if self.basic_info.test_years is None
            else np.asarray(self.basic_info.test_years, dtype=int)
        )

        # Monthly inflow columns in SynMonthlyInflow_train.xlsx and SynMonthlyInflow_evaluation.xlsx
        self.month_columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]

        # Running state
        self.current_step = 0
        self.current_year_index = 0
        self.is_training = True
        self.current_month = 1
        self.state = None
        self.current_storage = float(self.normal_storage)
        self._training_window = False

        # Annual tracking
        self.yearly_electricity = 0.0
        self.yearly_electricity_history = []
        self.monthly_electricity_in_year = []
        self.year_counter = 0

        # Reward normalization scale: one month of full-capacity generation
        self.reward_scale_gwh = (self.max_power * self.step_hours) / 1e6
        if self.reward_scale_gwh <= 0:
            self.reward_scale_gwh = 1.0

    def reset(self, seed=None, options=None, debug: bool = False):
        """
        Reset the environment.
        """
        super().reset(seed=seed)

        self.debug = debug
        data = self.train_data if self.is_training else self.test_data
        max_years = len(data)
        options = dict(options or {})
        self._training_window = "window_start_year_index" in options
        if self._training_window:
            start = options["window_start_year_index"]
            if not self.is_training or isinstance(start, bool) or not isinstance(start, (int, np.integer)):
                raise ValueError("A training window requires an integer start-year index")
            if self.max_steps != 240 or not 0 <= start or start * 12 + self.max_steps >= max_years * 12:
                raise ValueError("January start must leave 240 transitions and a real next month")
            self.current_year_index = int(start)
        else:
            self.current_year_index = max(0, min(self.current_year_index, max_years - 1))

        initial_inflow = float(data.iloc[self.current_year_index][self.month_columns[0]])
        self.current_storage = float(self.normal_storage)

        self.current_month = 1
        self.state = np.array(
            [self.current_month, self.initial_level, initial_inflow],
            dtype=np.float64,
        )
        self.current_step = 0

        self.yearly_electricity = 0.0
        self.monthly_electricity_in_year = []
        self.year_counter = 0

        return self.state, {}

    def _current_duration_days(self) -> float:
        years = self.train_years if self.is_training else self.test_years
        year = int(years[self.current_year_index])
        return float(calendar.monthrange(year, int(self.current_month))[1])

    def get_feasible_action_mask(self) -> np.ndarray:
        """Return reachability and flood-limit feasibility for all 75 actions."""
        if self.state is None:
            raise RuntimeError("Reset the environment before requesting an action mask")
        _, current_level, current_inflow = self.state
        duration_days = self._current_duration_days()
        mask = self.reservoir_model.feasible_target_level_mask(
            current_level,
            float(current_inflow),
            duration_days,
            month=int(self.current_month),
            target_levels=self.basic_info.action_levels,
        )
        if not np.any(mask):
            raise RuntimeError("Current physical state has no feasible DQN action")
        return mask

    def step(self, action):
        """
        Advance the environment by one monthly step.
        """
        if self._training_window and self.current_step >= self.max_steps:
            raise RuntimeError("Training window ended; reset before another transition")
        action = int(action)
        action = np.clip(action, 0, len(self.basic_info.state_present_storage) - 1)
        feasible_mask = self.get_feasible_action_mask()
        if not bool(feasible_mask[action]):
            raise ValueError(f"DQN selected infeasible action index {action}")

        current_storage = float(self.current_storage)
        current_inflow = float(self.state[2])

        next_inflow = self._get_next_inflow()
        target_storage = float(self.basic_info.state_present_storage[action])

        (
            next_storage,
            reward,
            total_discharge,
            turbine_discharge,
            spill_discharge,
            power_output,
            electricity_kwh,
            head,
            tailwater_level,
            start_level,
            end_level,
            water_balance_error_m3,
            was_corrected,
            correction_amount_m,
            executed_action,
            duration_days,
            deficit_energy_gwh,
            penalized_objective_gwh,
            firm_power_violation,
            max_total_release_triggered,
            unresolved_upper_storage_excess_m3,
        ) = self._calculate_operation_reward(
            current_storage=current_storage,
            current_inflow=current_inflow,
            target_storage=target_storage,
        )

        self.yearly_electricity += electricity_kwh
        is_year_end = self.current_month == 12
        if is_year_end:
            self.yearly_electricity_history.append(self.yearly_electricity)

        info = {
            "month": int(self.current_month),
            "year": self.current_year_index + 1,
            "start_water_level": start_level,
            "water_level": end_level,
            "start_storage": current_storage,
            "end_storage": next_storage,
            "inflow": current_inflow,
            "total_discharge": total_discharge,
            "turbine_discharge": turbine_discharge,
            "spill_discharge": spill_discharge,
            "power_output": power_output,
            "electricity_kwh": electricity_kwh,
            "target_storage": target_storage,
            "requested_target_level": float(self.basic_info.action_levels[action]),
            "executed_end_level": float(end_level),
            "raw_action": int(action),
            "nearest_grid_index_to_executed_level": int(executed_action),
            "was_corrected": bool(was_corrected),
            "correction_amount_m": float(correction_amount_m),
            "duration_days": float(duration_days),
            "feasible_action_count": int(feasible_mask.sum()),
            "mask_anomaly": False,
            "tailwater_level": tailwater_level,
            "head": head,
            "water_balance_error_m3": water_balance_error_m3,
            "deficit_energy_gwh": float(deficit_energy_gwh),
            "penalized_objective_gwh": float(penalized_objective_gwh),
            "firm_power_violation": bool(firm_power_violation),
            "failure_penalty_gwh": self.firm_penalty_lambda * int(firm_power_violation),
            "max_total_release_triggered": bool(max_total_release_triggered),
            "unresolved_upper_storage_excess_m3": float(unresolved_upper_storage_excess_m3),
            "yearly_electricity_so_far": self.yearly_electricity,
            "is_year_end": is_year_end,
            "year_index": self.year_counter,
        }

        # Advance the calendar by one month. When December is finished, move to
        # January and update the year index accordingly.
        next_month = 1 if self.current_month == 12 else self.current_month + 1

        if is_year_end:
            data = self.train_data if self.is_training else self.test_data
            if self.is_training and not self._training_window:
                self.current_year_index = (self.current_year_index + 1) % len(data)
            else:
                self.current_year_index = self.current_year_index + 1
            self.year_counter += 1
            self.yearly_electricity = 0.0
            self.monthly_electricity_in_year = []

        self.current_month = next_month
        self.current_storage = float(next_storage)
        self.state = np.array(
            [self.current_month, end_level, next_inflow],
            dtype=np.float64,
        )

        self.current_step += 1
        reached_limit = self.current_step >= self.max_steps
        terminated = reached_limit and not self._training_window
        truncated = reached_limit and self._training_window

        return self.state, reward, terminated, truncated, info

    def _get_next_inflow(self) -> float:
        """
        Return the next-month inflow used in the next state.
        """
        data = self.train_data if self.is_training else self.test_data
        current_year = self.current_year_index
        current_month = int(self.current_month)
        next_month = 1 if current_month == 12 else current_month + 1

        if self._training_window:
            next_year = current_year + 1 if current_month == 12 else current_year
            if next_year >= len(data):
                raise RuntimeError("Training window cannot wrap or cross its learning split")
        elif self.is_training:
            next_year = (current_year + 1) % len(data) if current_month == 12 else current_year
        else:
            if current_month == 12:
                if current_year >= len(data) - 1:
                    return float(data.iloc[current_year][self.month_columns[11]])
                next_year = current_year + 1
            else:
                next_year = current_year

        month_col = self.month_columns[next_month - 1]
        return float(data.iloc[next_year][month_col])

    def _calculate_operation_reward(
        self,
        current_storage: float,
        current_inflow: float,
        target_storage: float,
    ):
        """
        Compute one-step operation, reward, and diagnostic variables using
        storage-based state transition and water balance.

        Returns:
            (
                next_storage,
                reward,
                total_discharge,
                turbine_discharge,
                spill_discharge,
                power_output,
                electricity_kwh,
                head,
                tailwater_level,
                start_level,
                end_level,
                water_balance_error_m3,
            )
        """
        start_level = float(self.utils.storage_to_level(current_storage))
        requested_level = float(self.utils.storage_to_level(target_storage))
        duration_days = self._current_duration_days()
        transition = self.reservoir_model.calculate_transition(
            start_level,
            requested_level,
            current_inflow,
            duration_days,
            month=int(self.current_month),
        )
        next_storage = float(transition["end_storage"])
        power_output = float(transition["power_kw"])
        electricity_gwh = float(transition["energy_gwh"])
        electricity_kwh = electricity_gwh * 1e6
        deficit_energy_gwh = float(transition["deficit_energy_gwh"])
        firm_power_violation = bool(power_output < self.firm_power)
        penalized_objective_gwh = (
            electricity_gwh - self.firm_penalty_lambda * int(firm_power_violation)
        )
        reward = float(penalized_objective_gwh / self.reward_scale_gwh)
        executed_action = int(
            np.argmin(
                np.abs(self.basic_info.state_present_storage - next_storage)
            )
        )
        return (
            next_storage,
            reward,
            float(transition["total_discharge"]),
            float(transition["turbine_discharge"]),
            float(transition["spill_discharge"]),
            power_output,
            electricity_kwh,
            float(transition["head"]),
            float(transition["tailwater_level"]),
            start_level,
            float(transition["end_level"]),
            float(transition["water_balance_error_m3"]),
            bool(transition["was_corrected"]),
            float(transition["correction_amount_m"]),
            executed_action,
            duration_days,
            deficit_energy_gwh,
            penalized_objective_gwh,
            firm_power_violation,
            transition["max_total_release_triggered"],
            transition["unresolved_upper_storage_excess_m3"],
        )

