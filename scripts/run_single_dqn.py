"""Run the frozen single-reservoir DQN development, final, or evaluation stage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reservoir_benchmark.evaluation.dqn.dqn_eval import main_dqn_eval
from reservoir_benchmark.policy_generation.dqn.dqn_runner import main_dqn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("development", "final", "evaluation"),
        help="Run exactly one strictly separated DQN experiment stage.",
    )
    parser.add_argument("--lambda", dest="penalty_lambda", type=float, required=True,
                        help="GWh penalty per failed month for this isolated run.")
    parser.add_argument("--final-budget", type=int, default=None,
                        help="Required common episode budget B in final mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage = args.stage
    if stage == "evaluation":
        if args.final_budget is not None:
            raise SystemExit("--final-budget is not used during evaluation")
        main_dqn_eval(penalty_lambda=args.penalty_lambda)
    else:
        if stage == "final" and (args.final_budget is None or args.final_budget <= 0):
            raise SystemExit("final requires --final-budget B selected after development")
        if stage == "development" and args.final_budget is not None:
            raise SystemExit("development does not accept --final-budget")
        main_dqn(mode=stage, penalty_lambda=args.penalty_lambda,
                 final_budget=args.final_budget)


if __name__ == "__main__":
    main()
