import os
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

"""
pixi run python -m run_detection.py で実行
"""


@hydra.main(
    config_path="conf/video_conf", config_name="detection_config", version_base=None
)
def main(cfg: DictConfig) -> None:
    # os.environ["OMP_NUM_THREADS"] = "12"
    # os.environ["MKL_NUM_THREADS"] = "12"

    print("Config:\n" + OmegaConf.to_yaml(cfg))

    detector = instantiate(cfg.detector)

    video_dir = Path(cfg.video_dir)
    video_paths = list(sorted(video_dir.glob(f"*.{cfg.video_extension}")))

    if "TASK_INDEX" in os.environ:
        start = int(os.environ["TASK_INDEX"]) - 1
        end = start + int(os.environ["TASK_STEPSIZE"])

    else:
        assert cfg.task_index is not None and cfg.task_stepsize is not None, (
            "task_index and task_stepsize must be set"
        )
        start = int(cfg.task_index) - 1
        end = start + int(cfg.task_stepsize)

    video_paths = video_paths[start:end]

    root_save_dir = Path(cfg.output_dir)

    print("Running detection...")
    detector.run(video_paths, root_save_dir)


if __name__ == "__main__":
    main()
