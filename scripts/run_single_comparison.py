"""Run one explicit non-DQN stage of the single-reservoir comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reservoir_benchmark.evaluation.ddp.ddp_eval import main_ddp_eval
from reservoir_benchmark.evaluation.iso_surf.iso_surf_eval import main_iso_surf_eval
from reservoir_benchmark.evaluation.sdp.sdp_eval import main_sdp_eval
from reservoir_benchmark.policy_generation.iso_surf.iso_surf_runner import (
    main_iso_surf,
)
from reservoir_benchmark.policy_generation.sdp.sdp_runner import main_sdp


STAGES = {
    "ddp-evaluation": main_ddp_eval,
    "sdp-training": main_sdp,
    "sdp-evaluation": main_sdp_eval,
    "iso-training": main_iso_surf,
    "iso-evaluation": main_iso_surf_eval,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=tuple(STAGES))
    args = parser.parse_args()
    if args.stage in ("iso-training", "iso-evaluation"):
        parser.error(
            "This ISO-RF working copy uses scripts/run_single_iso_rf.py; "
            "follow development -> lambda-scan -> select-lambda -> final-train -> evaluation. "
            "The legacy ISO-SURF stages are retained only as reference modules."
        )
    STAGES[args.stage]()


if __name__ == "__main__":
    main()
