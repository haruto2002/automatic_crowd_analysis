import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from run_tracking.tracker import Tracker
import time

"""
python -m run_tracking.main で実行
"""


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    start_time = time.time()
    print("Config:\n" + OmegaConf.to_yaml(cfg))

    tracker = instantiate(cfg.tracker)

    result_dirs = Path(cfg.result_dir).glob("*")

    pool_list = []
    for res_dir in result_dirs:
        detection_dir = res_dir.joinpath("detection").as_posix()
        track_save_dir = res_dir.joinpath("track").as_posix()
        pool_list.append((tracker, detection_dir, track_save_dir))

    pool_size = min(get_pool_size(cfg.node_type), len(pool_list))
    with Pool(pool_size) as p:
        for _ in tqdm(p.imap_unordered(run_tracking, pool_list), total=len(pool_list)):
            pass

    end_time = time.time()
    with open(Path(cfg.hydra_log_dir) / "tracking_time.txt", "w") as f:
        f.write(f"Tracking time: {int(end_time - start_time)} seconds")


def run_tracking(inputs: tuple[Tracker, str, str]):
    tracker, detection_data_dir, track_data_dir = inputs
    tracker.run(detection_data_dir, track_data_dir)


def get_pool_size(node_type: str) -> int:
    if node_type == "rt_HC":
        return 32
    elif node_type == "rt_HG":
        return 32
    elif node_type == "rt_HF":
        return 192
    else:
        raise ValueError(f"Invalid node type: {node_type}")


if __name__ == "__main__":
    main()
