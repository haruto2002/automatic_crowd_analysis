import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm

"""
python -m run_detection.main で実行
"""


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.gpu_id == 0:
        print("Config:\n" + OmegaConf.to_yaml(cfg))

        print("Setting...")
    detector = instantiate(cfg.detector)
    io_setting = instantiate(cfg.io_setting)
    io_info_list = io_setting.get_io_info_list()

    tqdm_disable = cfg.gpu_id != 0

    for io_info in tqdm(io_info_list, desc="Running detection", disable=tqdm_disable):
        path2video = io_info.input_video_path.as_posix()
        detection_save_dir = io_info.detection_save_dir.as_posix()
        detector.run(path2video, detection_save_dir)


if __name__ == "__main__":
    main()
