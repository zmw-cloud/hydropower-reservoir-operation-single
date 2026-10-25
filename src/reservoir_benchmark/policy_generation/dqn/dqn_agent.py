"""
DQN agent module for the unified reservoir operation framework.

This module provides the DQN-side shared data container, replay buffer,
network definition, and agent wrapper, including:
1) unified DQN configuration,
2) shared reservoir and inflow data loading,
3) action discretization and storage candidate generation,
4) replay buffer implementation,
5) DQN network definition, and
6) DQN agent training utilities.

The purpose of this module is to isolate model- and agent-related logic from
environment simulation, training orchestration, and evaluation workflows.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import amp

from reservoir_benchmark.shared.io_utils import (
    load_algorithm_settings,
    load_monthly_inflows_with_calendar,
    require_keys,
)
from reservoir_benchmark.shared.project_paths import get_data_dir
from reservoir_benchmark.shared.reservoir_model import SharedReservoirModel
from reservoir_benchmark.shared.experiment_contract import require_contract


def set_global_seed(seed: int = 42) -> None:
    """
    Set Python / NumPy / PyTorch random seeds consistently.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Config:
    """
    Configuration container for DQN training and evaluation.
    """

    def __init__(self):
        settings_path = get_data_dir() / "experiment_settings.yaml"
        settings = load_algorithm_settings(settings_path, "dqn")

        require_keys(
            settings,
            ["agent", "network", "train", "validation", "experiment", "display"],
            "dqn",
        )

        self.agent_config = settings["agent"]
        self.network_config = settings["network"]
        self.train_config = settings["train"]
        self.validation_config = settings["validation"]
        self.experiment_config = settings["experiment"]
        self.display_config = settings["display"]

        require_keys(
            self.agent_config,
            [
                "learning_rate",
                "reward_decay",
                "epsilon_start",
                "epsilon_end",
                "epsilon_schedule",
                "epsilon_decay_steps",
                "learning_starts",
                "learn_every_steps",
                "updates_per_learning_event",
                "replace_target_iter",
                "memory_size",
                "batch_size",
                "firm_penalty_lambda",
            ],
            "dqn.agent",
        )
        require_keys(self.network_config, ["hidden_size"], "dqn.network")
        require_keys(
            self.train_config,
            [
                "development_episode_budget",
                "final_budget",
                "max_steps",
                "num_envs",
                "validation_interval_episodes",
                "use_amp",
                "reward_clip",
            ],
            "dqn.train",
        )
        require_keys(
            self.validation_config,
            ["development_years", "validation_years", "checkpoint_metric"],
            "dqn.validation",
        )
        require_keys(
            self.experiment_config,
            ["seed_list", "hyperparameter_status"],
            "dqn.experiment",
        )
        require_keys(
            self.display_config,
            ["show_progress", "progress_update_freq"],
            "dqn.display",
        )

        self.seed_list = [int(seed) for seed in self.experiment_config["seed_list"]]
        if not self.seed_list:
            raise ValueError("dqn.experiment.seed_list must not be empty.")

        self.seed = self.seed_list[0]
        if int(self.agent_config["learning_starts"]) < 0:
            raise ValueError("dqn.agent.learning_starts must be nonnegative")
        if int(self.agent_config["epsilon_decay_steps"]) <= 0:
            raise ValueError("dqn.agent.epsilon_decay_steps must be positive")
        if not 0.0 <= float(self.agent_config["epsilon_end"]) <= float(self.agent_config["epsilon_start"]) <= 1.0:
            raise ValueError("DQN epsilon values must satisfy 0 <= min <= start <= 1")
        if self.train_config["reward_clip"] is not None:
            raise ValueError("Formal DQN reward clipping must remain disabled")

    def get_seed(self):
        return self.seed

    def get_seed_list(self):
        return self.seed_list

    def set_seed(self, seed: int):
        self.seed = int(seed)

    def get_agent_params(self):
        return self.agent_config

    def get_network_params(self):
        return self.network_config

    def get_train_params(self):
        return self.train_config


class BasicInfo:
    """
    Basic data and shared reservoir information for DQN.
    """

    def __init__(self, config: Config, *, load_evaluation: bool = True):
        self.config = config
        self.load_evaluation = bool(load_evaluation)
        self.data_dir = get_data_dir()
        self.reservoir_model = SharedReservoirModel(data_dir=self.data_dir)

        # Shared reservoir parameters
        self.normal_level = self.reservoir_model.normal_level
        self.dead_level = self.reservoir_model.dead_level
        self.initial_level = self.reservoir_model.initial_level
        self.max_turbine_discharge = self.reservoir_model.max_turbine_discharge
        self.max_power = self.reservoir_model.max_power
        self.power_coefficient = self.reservoir_model.power_coefficient
        self.firm_power = self.reservoir_model.firm_power
        self.inflow_upper_bound = self.reservoir_model.inflow_upper_bound

        # Shared time-step parameters
        self.delta_t_days = self.reservoir_model.delta_t_days
        self.delta_t_hours = self.reservoir_model.delta_t_hours
        self.delta_t_seconds = self.reservoir_model.delta_t_seconds

        # Experiment-controlled penalty parameter
        self.firm_penalty_lambda = float(self.config.agent_config["firm_penalty_lambda"])

        self.months = list(range(1, 13))

        self.init_arrays()

    def init_arrays(self):
        """
        Load action storage candidates and inflow series.
        """
        # Action discretization: dead_level to normal_level at 1 m resolution
        action_levels = np.arange(
            int(self.dead_level),
            int(self.normal_level) + 1,
            1,
            dtype=np.int32,
        )
        self.action_levels = action_levels.astype(np.float64)

        self.state_present_storage = np.array(
            [float(self.reservoir_model.calculate_storage(float(z))) for z in action_levels],
            dtype=np.float64,
        )

        self.dead_storage = float(
            self.reservoir_model.calculate_storage(float(self.dead_level))
        )
        self.normal_storage = float(
            self.reservoir_model.calculate_storage(float(self.normal_level))
        )

        # Inflow series
        train_path = self.data_dir / "SynMonthlyInflow_train.xlsx"
        test_path = self.data_dir / "SynMonthlyInflow_evaluation.xlsx"

        if not train_path.exists():
            raise FileNotFoundError(f"Training inflow file not found: {train_path}")
        if self.load_evaluation and not test_path.exists():
            raise FileNotFoundError(f"Evaluation inflow file not found: {test_path}")

        self.train_inflow, self.train_years = load_monthly_inflows_with_calendar(
            train_path
        )
        if self.load_evaluation:
            self.test_inflow, self.test_years = load_monthly_inflows_with_calendar(
                test_path
            )
        else:
            self.test_inflow = None
            self.test_years = None
        train_values = self.train_inflow.to_numpy(dtype=float)
        self.inflow_lower_bound = float(np.min(train_values))
        self.inflow_upper_bound = float(np.max(train_values))


class ReplayBuffer:
    """
    High-performance circular replay buffer.
    """

    def __init__(self, capacity, n_features, n_actions):
        self.capacity = capacity
        self.n_features = n_features
        self.n_actions = int(n_actions)
        self.ptr = 0
        self.size = 0
        self.s = np.empty((capacity, n_features), dtype=np.float32)
        self.a = np.empty((capacity, 1), dtype=np.int64)
        self.r = np.empty((capacity, 1), dtype=np.float32)
        self.s_next = np.empty((capacity, n_features), dtype=np.float32)
        self.done = np.empty((capacity, 1), dtype=np.float32)
        self.next_action_masks = np.empty(
            (capacity, self.n_actions),
            dtype=np.bool_,
        )

    def push(self, s, a, r, s_, d, next_action_mask):
        i = self.ptr
        self.s[i] = s
        self.a[i, 0] = a
        self.r[i, 0] = r
        self.s_next[i] = s_
        self.done[i, 0] = float(d)
        mask = np.asarray(next_action_mask, dtype=np.bool_)
        if mask.shape != (self.n_actions,) or not np.any(mask):
            raise ValueError("next_action_mask must contain at least one feasible action")
        self.next_action_masks[i] = mask
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size, self.capacity - 1) + 1

    def sample(self, batch_size):
        idx = np.random.choice(self.size, size=batch_size, replace=False)
        return (
            self.s[idx],
            self.a[idx],
            self.r[idx],
            self.s_next[idx],
            self.done[idx],
            self.next_action_masks[idx],
        )

    def __len__(self):
        return self.size


class DQN(nn.Module):
    """
    Deep Q-network with LayerNorm.
    """

    def __init__(self, n_features, n_actions, config: Config):
        super().__init__()

        hidden_size = config.get_network_params()["hidden_size"]

        self.fc1 = nn.Linear(n_features, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.out = nn.Linear(hidden_size // 2, n_actions)

        self.ln1 = nn.LayerNorm(hidden_size * 2)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size // 2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        return self.out(x)


class DQNAgent:
    """
    DQN agent with normalized states and AMP support.
    """

    def __init__(self, n_actions, n_features, config: Config, basic_info: BasicInfo, device=None):
        self.config = config
        self.n_actions = n_actions
        self.n_features = n_features
        self.basic_info = basic_info

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"GPU memory: "
                f"{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB"
            )

        agent_cfg = self.config.agent_config
        self.learning_rate = agent_cfg["learning_rate"]
        self.gamma = agent_cfg["reward_decay"]
        self.replace_target_iter = agent_cfg["replace_target_iter"]
        self.memory_size = agent_cfg["memory_size"]
        self.batch_size = agent_cfg["batch_size"]

        self.epsilon_start = agent_cfg["epsilon_start"]
        self.epsilon_end = agent_cfg["epsilon_end"]
        if agent_cfg["epsilon_schedule"] != "linear_environment_steps":
            raise ValueError("Formal DQN requires a linear environment-step epsilon schedule")
        self.epsilon_decay_steps = int(agent_cfg["epsilon_decay_steps"])
        self.learning_starts = int(agent_cfg["learning_starts"])
        self.epsilon = self.epsilon_start
        self.env_steps = 0

        self.memory = ReplayBuffer(self.memory_size, n_features, n_actions)

        self.eval_net = DQN(n_features, n_actions, config).to(self.device)
        self.target_net = DQN(n_features, n_actions, config).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())

        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.SmoothL1Loss()

        self.use_amp = bool(self.config.train_config["use_amp"])
        self.scaler = amp.GradScaler(enabled=(self.use_amp and self.device.type == "cuda"))

        self.learn_step_counter = 0

        self.state_low = np.array(
            [1.0, basic_info.dead_level, basic_info.inflow_lower_bound],
            dtype=np.float32,
        )
        self.state_high = np.array(
            [12.0, basic_info.normal_level, basic_info.inflow_upper_bound],
            dtype=np.float32,
        )
        self._norm_scale = 1.0 / (self.state_high - self.state_low + 1e-8)

        self.debug = False
        self.loss_history = []
        self.epsilon_history = []

    def normalize_state(self, state):
        """
        Normalize state to [0, 1].
        """
        normalized = (state - self.state_low) * self._norm_scale
        return np.clip(normalized, 0, 1)

    def epsilon_for_environment_steps(self, steps: int) -> float:
        progress = np.clip(
            (int(steps) - self.learning_starts) / self.epsilon_decay_steps,
            0.0,
            1.0,
        )
        return float(max(
            self.epsilon_end,
            self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start),
        ))

    def choose_action(
        self,
        observation,
        env=None,
        evaluate=False,
        action_mask=None,
    ):
        if action_mask is None:
            if env is None:
                raise ValueError("A physical action mask is required for action selection")
            action_mask = env.get_feasible_action_mask()
        mask = np.asarray(action_mask, dtype=np.bool_)
        if mask.shape != (self.n_actions,) or not np.any(mask):
            raise ValueError("action_mask must contain at least one feasible action")
        feasible_actions = np.flatnonzero(mask)
        normalized_obs = self.normalize_state(observation)
        obs_tensor = torch.as_tensor(
            normalized_obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        greedy = evaluate or np.random.uniform() > self.epsilon
        if greedy:
            with torch.no_grad():
                q_values = self.eval_net(obs_tensor).squeeze(0)
                mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
                q_values = q_values.masked_fill(~mask_tensor, -torch.inf)
                action = int(torch.argmax(q_values).item())
        else:
            action = int(np.random.choice(feasible_actions))

        return action

    def store_transition(
        self,
        s,
        a,
        r,
        s_,
        done=False,
        next_action_mask=None,
    ):
        if next_action_mask is None:
            raise ValueError("next_action_mask is required for Double-DQN targets")
        s_norm = self.normalize_state(s)
        s_next_norm = self.normalize_state(s_)
        self.memory.push(s_norm, a, r, s_next_norm, done, next_action_mask)
        self.env_steps += 1
        self.epsilon = self.epsilon_for_environment_steps(self.env_steps)

    def learn(self):
        """
        Learn from replay buffer using Double DQN.
        """
        if len(self.memory) < self.batch_size:
            return None

        b_s, b_a, b_r, b_s_, b_done, b_next_mask = self.memory.sample(
            self.batch_size
        )

        b_s = torch.as_tensor(b_s, dtype=torch.float32, device=self.device)
        b_a = torch.as_tensor(b_a, dtype=torch.long, device=self.device).squeeze(1)
        b_r = torch.as_tensor(b_r, dtype=torch.float32, device=self.device).squeeze(1)
        b_s_ = torch.as_tensor(b_s_, dtype=torch.float32, device=self.device)
        b_done = torch.as_tensor(b_done, dtype=torch.float32, device=self.device).squeeze(1)
        b_next_mask = torch.as_tensor(
            b_next_mask,
            dtype=torch.bool,
            device=self.device,
        )

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", enabled=(self.use_amp and self.device.type == "cuda")):
            q_eval = self.eval_net(b_s).gather(1, b_a.unsqueeze(1))
            with torch.no_grad():
                next_online_q = self.eval_net(b_s_).masked_fill(
                    ~b_next_mask,
                    -torch.inf,
                )
                next_actions = next_online_q.argmax(1, keepdim=True)
                q_next = self.target_net(b_s_).gather(1, next_actions).squeeze(1)
                q_target = (b_r + self.gamma * q_next * (1.0 - b_done)).unsqueeze(1)
            loss = self.loss_func(q_eval, q_target)

        self.scaler.scale(loss).backward()
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        loss_value = loss.item()
        self.loss_history.append(loss_value)
        self.epsilon_history.append(self.epsilon)
        return loss_value

    def save_model(self, path, *, artifact_metadata):
        """
        Save model state.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "eval_net_state_dict": self.eval_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "learn_step_counter": self.learn_step_counter,
            "env_steps": self.env_steps,
            "state_low": self.state_low,
            "state_high": self.state_high,
            "device": str(self.device),
            "seed": getattr(self.config, "seed", 42),
            "artifact_metadata": dict(artifact_metadata),
        }

        torch.save(checkpoint, path)
        return True

    def load_model(self, path, *, expected_lambda=None):
        """
        Load model state.
        """
        path = Path(path)
        if not path.exists():
            print(f"Model file not found: {path}")
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            require_contract(
                checkpoint.get("artifact_metadata"),
                penalty_lambda=expected_lambda,
            )
            self.artifact_metadata = dict(checkpoint["artifact_metadata"])
            if self.artifact_metadata.get("state_order") != [
                "month", "xiaowan_level_m", "xiaowan_current_inflow_m3s"
            ]:
                raise ValueError("DQN checkpoint state order is incompatible")

            self.eval_net.load_state_dict(checkpoint["eval_net_state_dict"])
            self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)

            self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
            self.learn_step_counter = checkpoint.get("learn_step_counter", 0)
            self.env_steps = checkpoint.get("env_steps", 0)
            self.epsilon = self.epsilon_for_environment_steps(self.env_steps)

            if "state_low" in checkpoint:
                self.state_low = checkpoint["state_low"]
            if "state_high" in checkpoint:
                self.state_high = checkpoint["state_high"]
            self._norm_scale = 1.0 / (
                self.state_high - self.state_low + 1e-8
            )

            print(f"Model loaded successfully (device: {self.device})")
            return True

        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            return False
