from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from displayer import AllMapDisplayerMatplotlib, AreaConfig
from tqdm import tqdm


def create_video(
    output_img_list: list[np.ndarray],
    video_path: Path,
    fps: int = 1,
) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    height = output_img_list[0].shape[0]
    width = output_img_list[0].shape[1]
    out = cv2.VideoWriter(video_path.as_posix(), fourcc, fps, (width, height))
    for output_img in tqdm(output_img_list, desc="Creating video"):
        out.write(output_img)
    out.release()


def create_video_from_files(
    frame_save_dir: Path, video_path: Path, fps: int = 1
) -> None:
    path2frame_list = sorted(frame_save_dir.glob("*.jpg"))
    output_img_list = [
        cv2.imread(path2frame.as_posix()) for path2frame in path2frame_list
    ]
    create_video(output_img_list, video_path, fps=fps)


def load_config(yaml_path: Path) -> AreaConfig:
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return AreaConfig(
        path2homography_matrix=Path(cfg["path2homography_matrix"]),
        points_dir=Path(cfg["points_dir"]),
        original_area_size=[float(x) for x in cfg["original_area_size"]],
        all_map_left_top_coor=[float(x) for x in cfg["all_map_left_top_coor"]],
        scale=float(cfg["scale"]),
    )


def set_config_list(conf_dir: Path):
    config_list = []
    for conf_file in conf_dir.glob("*.yaml"):
        config_list.append(load_config(conf_file))
    return config_list


def main():
    all_map_path = Path("bcm/venue_data/202508_yokohama/all/all_map.jpg")
    conf_dir = Path("bcm/config/202508_yokohama")
    save_dir = Path("bcm/output/202508_yokohama")
    frame_save_dir = save_dir.joinpath("frames")
    save_dir.mkdir(parents=True, exist_ok=True)
    frame_save_dir.mkdir(parents=True, exist_ok=True)

    config_list = set_config_list(conf_dir)
    displayer = AllMapDisplayerMatplotlib(all_map_path, config_list)
    start_time = datetime(2025, 8, 4, 16, 30, 0)
    end_time = datetime(2025, 8, 4, 22, 30, 0)

    results = [
        displayer.run(time)
        for time in tqdm(
            pd.date_range(start_time, end_time, freq="1min"), desc="Displaying all maps"
        )
    ]
    for time, output_img in tqdm(results, desc="Saving frames"):
        time_str = time.strftime("%Y-%m-%d_%H:%M:%S")
        cv2.imwrite(
            frame_save_dir.joinpath(f"{time_str}.jpg").as_posix(),
            output_img,
        )

    sorted_results = sorted(results, key=lambda x: x[0])
    output_img_list = [output_img for _, output_img in sorted_results]

    video_path = save_dir.joinpath("all_map_video.mp4")
    create_video(output_img_list, video_path, fps=10)


if __name__ == "__main__":
    main()
