"""Development/final training workflow for the single-reservoir Double DQN."""

from __future__ import annotations

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
    get_data_dir,
    get_formal_algorithm_dir,
)
from reservoir_benchmark.shared.experiment_contract import base_metadata


TRAINING_MODES = ("development", "final")


def _valid_january_start_year_indices(frame, years, episode_months):
    """Annual-table row indices whose January leaves a real bootstrap month."""
    if episode_months != 240:
        raise ValueError("The comparison protocol requires 240 monthly transitions")
    if frame.shape[1] != 12 or len(frame) != len(years):
        raise ValueError("Training data must contain 12 months per calendar year")
    if not np.all(np.diff(np.asarray(years, dtype=int)) == 1):
        raise ValueError("Training calendar years must be consecutive")
    candidates = np.arange(len(frame), dtype=np.int64)
    candidates = candidates[candidates * 12 + episode_months < len(frame) * 12]
    if len(candidates) == 0:
        raise ValueError("No January leaves 240 transitions plus a bootstrap month")
    return candidates


def _split_training_data(basic_info: BasicInfo, config: Config, mode: str):
    if mode not in TRAINING_MODES:
        raise ValueError(f"mode must be one of {TRAINING_MODES}")
    frame = basic_info.train_inflow.reset_index(drop=True)
    years = np.asarray(basic_info.train_years, dtype=int)
    development_years = int(config.validation_config["development_years"])
    validation_years = int(config.validation_config["validation_years"])
    if (development_years, validation_years) != (320, 80):
        raise ValueError("The comparison protocol requires the 320/80-year split")
    if len(frame) != development_years + validation_years:
        raise ValueError(
            "DQN requires exactly development_years + validation_years rows "
            "in the configured training workbook"
        )
    if mode == "final":
        return frame, years, None, None
    learning = frame.iloc[:development_years].reset_index(drop=True)
    validation = frame.iloc[development_years:].reset_index(drop=True)
    return (
        learning,
        years[:development_years].copy(),
        validation,
        years[development_years:].copy(),
    )


def _configure_env_data(
    env: ReservoirEnv,
    frame: pd.DataFrame,
    years: np.ndarray,
) -> None:
    env.train_data = frame.reset_index(drop=True).copy()
    env.train_years = np.asarray(years, dtype=int).copy()
    env.is_training = True
    env.current_year_index = 0


def _greedy_validation(
    agent: DQNAgent,
    basic_info: BasicInfo,
    validation_frame: pd.DataFrame,
    validation_years: np.ndarray,
) -> dict:
    env = ReservoirEnv(basic_info, debug=False)
    _configure_env_data(env, validation_frame, validation_years)
    env.max_steps = len(validation_frame) * 12
    observation, _ = env.reset()
    total_generation_gwh = 0.0
    total_penalized_objective_gwh = 0.0
    total_deficit_energy_gwh = 0.0
    violation_month_count = 0
    total_spill_volume_m3 = 0.0
    satisfied_months = 0
    total_months = 0

    eval_was_training = agent.eval_net.training
    target_was_training = agent.target_net.training
    agent.eval_net.eval()
    agent.target_net.eval()
    try:
        with torch.no_grad():
            while True:
                mask = env.get_feasible_action_mask()
                action = agent.choose_action(
                    observation,
                    env=env,
                    evaluate=True,
                    action_mask=mask,
                )
                next_observation, _, terminated, truncated, info = env.step(action)
                total_generation_gwh += float(info["electricity_kwh"]) / 1e6
                total_penalized_objective_gwh += float(
                    info["penalized_objective_gwh"]
                )
                total_deficit_energy_gwh += float(info["deficit_energy_gwh"])
                violation_month_count += int(info["firm_power_violation"])
                total_spill_volume_m3 += (
                    float(info["spill_discharge"]) * float(info["duration_days"]) * 86400.0
                )
                satisfied_months += int(
                    float(info["power_output"]) >= basic_info.firm_power
                )
                total_months += 1
                if terminated or truncated:
                    break
                observation = next_observation
    finally:
        agent.eval_net.train(eval_was_training)
        agent.target_net.train(target_was_training)

    years_count = total_months / 12.0
    return {
        "months": total_months,
        "total_generation_gwh": total_generation_gwh,
        "mean_annual_generation_gwh": total_generation_gwh / years_count,
        "total_penalized_objective_gwh": total_penalized_objective_gwh,
        "mean_annual_penalized_objective_gwh": (
            total_penalized_objective_gwh / years_count
        ),
        "guarantee_rate": satisfied_months / total_months,
        "violation_month_count": violation_month_count,
        "deficit_energy_gwh": total_deficit_energy_gwh,
        "total_spill_volume_m3": total_spill_volume_m3,
    }


def run_training(
    firm_penalty_lambda: float,
    seed: int,
    *,
    mode: str,
    final_budget: int | None = None,
) -> Path:
    """Train one frozen-hyperparameter seed in development or final mode."""
    if mode not in TRAINING_MODES:
        raise ValueError(f"mode must be one of {TRAINING_MODES}")
    config = Config()
    if int(config.train_config["num_envs"]) != 1:
        raise ValueError("The retained single-reservoir protocol requires num_envs=1")
    config.set_seed(seed)
    config.agent_config["firm_penalty_lambda"] = float(firm_penalty_lambda)
    set_global_seed(seed)

    # Training and internal validation never open the independent evaluation file.
    basic_info = BasicInfo(config, load_evaluation=False)
    learning_frame, learning_years, validation_frame, validation_years = (
        _split_training_data(basic_info, config, mode)
    )
    learning_values = learning_frame.to_numpy(dtype=float)
    basic_info.inflow_lower_bound = float(np.min(learning_values))
    basic_info.inflow_upper_bound = float(np.max(learning_values))

    env = ReservoirEnv(basic_info, debug=False)
    _configure_env_data(env, learning_frame, learning_years)
    valid_start_years = _valid_january_start_year_indices(
        learning_frame, learning_years, env.max_steps
    )
    # Dedicated RNG: exploration/replay/validation cannot advance this stream.
    window_rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(
        n_actions=env.action_space.n,
        n_features=env.observation_space.shape[0],
        config=config,
        basic_info=basic_info,
        device=device,
    )

    lam_str = format_lambda_for_filename(float(firm_penalty_lambda))
    root = get_formal_algorithm_dir("dqn")
    if mode == "development":
        run_dir = root / "development" / f"lambda_{lam_str}" / f"seed_{seed}"
        model_path = run_dir / "best_validation.pth"
    else:
        run_dir = root / "final" / f"lambda_{lam_str}" / f"seed_{seed}"
        model_path = root / "policy" / f"dqn_train_lambda_{lam_str}_seed_{seed}.pth"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    latest_model_path = run_dir / "latest.pth"

    if mode == "development":
        total_episodes = int(config.train_config["development_episode_budget"])
    else:
        if final_budget is None or int(final_budget) <= 0:
            raise ValueError("Final mode requires an explicit positive final_budget B")
        total_episodes = int(final_budget)
    validation_interval = int(
        config.train_config["validation_interval_episodes"]
    )
    show_progress = bool(config.display_config.get("show_progress", True))
    log_interval = max(
        1,
        int(config.display_config.get("progress_update_freq", 10)),
    )
    best_validation_score = float("-inf")
    training_records: list[dict] = []
    validation_records: list[dict] = []
    artifact_metadata = base_metadata(
        data_dir=get_data_dir(), split="train", penalty_lambda=firm_penalty_lambda,
        method="Double DQN", solver_variant="fixed_75_output_double_dqn_v2", grid_step=1.0)
    artifact_metadata.update({
        "state_order": ["month", "xiaowan_level_m", "xiaowan_current_inflow_m3s"],
        "normalization": "componentwise_minmax_training_subset_clip",
        "normalization_lower": agent.state_low.tolist(),
        "normalization_upper": agent.state_high.tolist(),
        "reward_scale_gwh": float(
            basic_info.max_power * basic_info.delta_t_days * 24.0 / 1e6
        ),
        "reward_clip": None,
        "gamma": float(agent.gamma),
        "epsilon_schedule": "linear_environment_steps",
        "learning_starts": int(agent.learning_starts),
        "epsilon_decay_steps": int(agent.epsilon_decay_steps),
        "epsilon_start": float(agent.epsilon_start),
        "epsilon_min": float(agent.epsilon_end),
        "learning_rate": float(agent.learning_rate),
        "replay_buffer_size": int(agent.memory_size),
        "batch_size": int(agent.batch_size),
        "target_update_interval_learning_steps": int(agent.replace_target_iter),
        "network_hidden_dimensions": [
            int(config.network_config["hidden_size"]) * 2,
            int(config.network_config["hidden_size"]),
            int(config.network_config["hidden_size"]) // 2,
        ],
        "use_amp": bool(agent.use_amp),
        "seed": int(seed),
        "training_mode": mode,
        "final_budget": None if mode == "development" else total_episodes,
        "selection_metric": "mean_annual_penalized_objective_gwh",
    })

    for episode in range(1, total_episodes + 1):
        start_year_index = int(window_rng.choice(valid_start_years))
        observation, _ = env.reset(
            options={"window_start_year_index": start_year_index}
        )
        action_mask = env.get_feasible_action_mask()
        episode_reward = 0.0
        episode_generation_gwh = 0.0
        episode_penalized_objective_gwh = 0.0
        episode_corrections = 0
        last_loss = None

        while True:
            action = agent.choose_action(
                observation,
                env=env,
                evaluate=False,
                action_mask=action_mask,
            )
            next_observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated != (env.current_step == env.max_steps):
                raise RuntimeError("Training window must truncate only after 240 steps")
            next_action_mask = env.get_feasible_action_mask()
            # Includes the real month-241 state/mask; an artificial window end
            # must retain its TD continuation rather than zeroing bootstrap.
            agent.store_transition(
                observation,
                action,
                reward,
                next_observation,
                done=False,
                next_action_mask=next_action_mask,
            )
            if (
                agent.env_steps >= agent.learning_starts
                and len(agent.memory) >= agent.batch_size
                and agent.env_steps % int(config.agent_config["learn_every_steps"]) == 0
            ):
                for _ in range(int(config.agent_config["updates_per_learning_event"])):
                    last_loss = agent.learn()
            episode_reward += float(reward)
            episode_generation_gwh += float(info["electricity_kwh"]) / 1e6
            episode_penalized_objective_gwh += float(
                info["penalized_objective_gwh"]
            )
            episode_corrections += int(info["was_corrected"])
            observation = next_observation
            action_mask = next_action_mask
            if terminated or truncated:
                break

        validation_score = None
        selected_best = False
        if mode == "development" and (
            episode % validation_interval == 0 or episode == total_episodes
        ):
            if validation_frame is None or validation_years is None:
                raise RuntimeError("Development mode is missing its validation split")
            validation = _greedy_validation(
                agent,
                basic_info,
                validation_frame,
                validation_years,
            )
            validation_score = float(
                validation["mean_annual_penalized_objective_gwh"]
            )
            selected_best = validation_score > best_validation_score
            if selected_best:
                best_validation_score = validation_score
                selected_metadata = {**artifact_metadata,
                                     "checkpoint_role": "best_validation",
                                     "completed_episode": episode,
                                     "env_steps": int(agent.env_steps)}
                agent.save_model(model_path, artifact_metadata=selected_metadata)
            validation_records.append(
                {
                    "episode": episode,
                    "selected_best": selected_best,
                    **validation,
                }
            )

        training_records.append(
            {
                "episode": episode,
                "episode_start_year_index": start_year_index,
                "episode_start_year": int(learning_years[start_year_index]),
                "episode_transitions": env.current_step,
                "normalized_episode_reward": episode_reward,
                "episode_generation_gwh": episode_generation_gwh,
                "episode_penalized_objective_gwh": (
                    episode_penalized_objective_gwh
                ),
                "epsilon": float(agent.epsilon),
                "env_steps": int(agent.env_steps),
                "learning_steps": int(agent.learn_step_counter),
                "last_loss": last_loss,
                "correction_count": episode_corrections,
                "validation_score": validation_score,
                "selected_best": selected_best,
            }
        )
        if episode % validation_interval == 0 or episode == total_episodes:
            latest_metadata = {
                **artifact_metadata,
                "checkpoint_role": "latest",
                "completed_episode": episode,
                "env_steps": int(agent.env_steps),
            }
            agent.save_model(latest_model_path, artifact_metadata=latest_metadata)
        if show_progress and (
            episode == 1
            or episode % log_interval == 0
            or episode == total_episodes
        ):
            print(
                f"mode={mode} seed={seed} episode={episode}/{total_episodes} "
                f"reward={episode_reward:.6f} epsilon={agent.epsilon:.6f} "
                f"validation={validation_score}"
            )

    if mode == "final":
        final_metadata = {**artifact_metadata, "checkpoint_role": "final",
                          "completed_episode": total_episodes,
                          "env_steps": int(agent.env_steps)}
        agent.save_model(model_path, artifact_metadata=final_metadata)
    elif not model_path.exists():
        raise RuntimeError("Development training produced no validation checkpoint")

    pd.DataFrame(training_records).to_csv(
        run_dir / "training_history.csv",
        index=False,
    )
    if validation_records:
        pd.DataFrame(validation_records).to_csv(
            run_dir / "validation_history.csv",
            index=False,
        )
    return model_path


def run_selected_training(mode: str, penalty_lambda: float, final_budget: int | None = None) -> list[Path]:
    config = Config()
    return [
        run_training(penalty_lambda, int(seed), mode=mode, final_budget=final_budget)
        for seed in config.get_seed_list()
    ]


def main_dqn(mode: str = "development", *, penalty_lambda: float, final_budget: int | None = None) -> None:
    run_selected_training(mode, penalty_lambda, final_budget)


if __name__ == "__main__":
    raise SystemExit("Use scripts/run_single_dqn.py with an explicit --lambda")
