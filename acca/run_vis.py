import os
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

"""
pixi run python -m run_detection.py で実行
"""


@hydra.main(config_path="conf", config_name="vis_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # os.environ["OMP_NUM_THREADS"] = "12"
    # os.environ["MKL_NUM_THREADS"] = "12"

    print("Config:\n" + OmegaConf.to_yaml(cfg))

    detection_visualizer = instantiate(cfg.detection_visualizer)
    tracking_visualizer = instantiate(cfg.tracking_visualizer)

    video_dir = Path(cfg.video_dir)
    video_paths = list(sorted(video_dir.glob(f"*.{cfg.video_extension}")))

    if "TASK_INDEX" in os.environ:
        start = int(os.environ["TASK_INDEX"]) - 1
        end = start + int(os.environ["TASK_STEPSIZE"])

    else:
        assert cfg.task_index is not None and cfg.task_stepsize is not None, "task_index and task_stepsize must be set"
        start = int(cfg.task_index) - 1
        end = start + int(cfg.task_stepsize)

    video_paths = video_paths[start:end]

    root_save_dir = Path(cfg.output_dir)

    print("Running visualizations...")
    for video_path in video_paths:
        detection_save_dir = root_save_dir.joinpath(video_path.stem, "detection")
        detection_vis_save_dir = root_save_dir.joinpath(video_path.stem, "vis", "detection")
        tracking_save_dir = root_save_dir.joinpath(video_path.stem, "track")
        tracking_vis_save_dir = root_save_dir.joinpath(video_path.stem, "vis", "track")
        tracking_vis_save_dir.mkdir(parents=True, exist_ok=True)
        tracking_mov_save_path = tracking_vis_save_dir.joinpath(f"{video_path.stem}.mp4")

        detection_visualizer.visualize(detection_save_dir, video_path, detection_vis_save_dir)
        print(tracking_save_dir)
        tracking_visualizer.visualize(tracking_save_dir, video_path, tracking_mov_save_path)


if __name__ == "__main__":
    main()
