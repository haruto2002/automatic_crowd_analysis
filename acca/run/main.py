import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

"""
python -m run.main で実行
"""


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print("Config:\n" + OmegaConf.to_yaml(cfg))

    detector = instantiate(cfg.detector)
    tracker = instantiate(cfg.tracker)

    detection_visualizer = instantiate(cfg.detection_visualizer)
    tracking_visualizer = instantiate(cfg.tracking_visualizer)

    path2video = cfg.video_path
    save_dir = cfg.output_dir

    print("Running detection...")
    detection_save_dir = save_dir + "/detection"
    detector.run(path2video, detection_save_dir)
    print("Running tracking...")
    tracker_save_dir = save_dir + "/track"
    tracker.run(detection_save_dir, tracker_save_dir)
    print("Visualizing detection...")
    detection_vis_save_dir = save_dir + "/vis/detection"
    detection_visualizer.visualize(
        detection_save_dir, path2video, detection_vis_save_dir
    )
    print("Visualizing tracking...")
    track_vis_save_path = save_dir + "/vis/track.mp4"
    tracking_visualizer.visualize(tracker_save_dir, path2video, track_vis_save_path)


if __name__ == "__main__":
    main()
