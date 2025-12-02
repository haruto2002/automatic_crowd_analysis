import os
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

"""
pixi run python -m run_detection.py で実行
"""


@hydra.main(config_path="conf", config_name="tracking_config", version_base=None)
def main(cfg: DictConfig) -> None:
    os.environ["OMP_NUM_THREADS"] = "12"
    os.environ["MKL_NUM_THREADS"] = "12"

    print("Config:\n" + OmegaConf.to_yaml(cfg))

    tracker = instantiate(cfg.tracker)

    results_root_dir = Path(cfg.results_root_dir)
    results_dirs = list(sorted(results_root_dir.glob("*")))

    if "TASK_INDEX" in os.environ:
        start = int(os.environ["TASK_INDEX"]) - 1
        end = start + int(os.environ["TASK_STEPSIZE"])

    else:
        assert cfg.task_index is not None and cfg.task_stepsize is not None, "task_index and task_stepsize must be set"
        start = int(cfg.task_index) - 1
        end = start + int(cfg.task_stepsize)

    results_dirs = results_dirs[start:end]

    print("Running tracking...")
    tracker.run(results_dirs)


if __name__ == "__main__":
    main()
