import glob
import os
import numpy as np
from vis_heatmap import display_heatmap_parallel
import cv2
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
import argparse


def get_back_img(
    img_dir, end_frame, crop_area=None, path2bev_matrix=None, path2bev_size=None
):
    path2img = sorted(glob.glob(f"{img_dir}/*.png"))[end_frame - 1]
    img = cv2.imread(path2img)
    if path2bev_matrix is not None:
        assert path2bev_size is not None
        img = bev_trans(path2bev_matrix, path2bev_size, img)

    if crop_area is not None:
        img = img[crop_area[1] : crop_area[3], crop_area[0] : crop_area[2], :]

    return img


def bev_trans(path2bev_matrix, path2bev_size, img):
    matrix = np.loadtxt(path2bev_matrix)
    img_w, img_h = np.loadtxt(path2bev_size).astype(int)
    bev_img = cv2.warpPerspective(
        img,
        matrix,
        (img_w, img_h),
        borderValue=(255, 255, 255),
    )
    return bev_img


def parallel_get_back_img(inputs):
    (img_dir, end_frame, crop_area, path2bev_matrix, path2bev_size) = inputs
    back_img = get_back_img(
        img_dir, end_frame, crop_area, path2bev_matrix, path2bev_size
    )
    return end_frame, back_img


def main(
    save_base_dir,
    img_dir,
    source_dir,
    frame_range=[801, 1500],
    crop_area=[180, 350, 460, 630],
    path2bev_matrix=None,
    path2bev_size=None,
    scale=5,
    smoothing=False,
):
    print("Setting data...")
    path2cfg = os.path.join(source_dir, "config.yaml")
    with open(path2cfg, "r") as f:
        cfg = yaml.safe_load(f)
    grid_size = cfg["grid_size"]
    freq = cfg["freq"]
    if smoothing:
        danger_source_dir = os.path.join(
            source_dir, "each_result", "danger_score_smoothed"
        )
    else:
        danger_source_dir = os.path.join(source_dir, "each_result", "danger_score")
    vec_source_dir = os.path.join(source_dir, "each_result", "vec_data")
    if smoothing:
        save_dir = f"{save_base_dir}/{frame_range[0]}_{frame_range[1]}_smoothed"
    else:
        save_dir = f"{save_base_dir}/{frame_range[0]}_{frame_range[1]}"
    os.makedirs(save_dir, exist_ok=True)
    map_crop_area = [
        crop_area[0] // grid_size,
        crop_area[1] // grid_size,
        crop_area[2] // grid_size,
        crop_area[3] // grid_size,
    ]

    danger_score_path_list = sorted(glob.glob(os.path.join(danger_source_dir, "*.txt")))
    vec_data_path_list = sorted(glob.glob(os.path.join(vec_source_dir, "*.txt")))

    target_danger_score_path_list = danger_score_path_list[
        frame_range[0] // freq : frame_range[1] // freq
    ]
    target_vec_data_path_list = vec_data_path_list[
        frame_range[0] // freq : frame_range[1] // freq
    ]

    danger_score_list = [
        crop_map(np.loadtxt(danger_score_path), map_crop_area)
        for danger_score_path in target_danger_score_path_list
    ]
    vec_data_list = [
        crop_vec_data(np.loadtxt(vec_data_path), crop_area)
        for vec_data_path in target_vec_data_path_list
    ]
    name_list = [
        os.path.basename(danger_score_path).split(".")[0]
        for danger_score_path in target_danger_score_path_list
    ]

    pool_size = min(len(danger_score_list), os.cpu_count())
    pool_list = []
    for name in name_list:
        last_frame = int(name.split("_")[-1])
        pool_list.append(
            (img_dir, last_frame, crop_area, path2bev_matrix, path2bev_size)
        )
    with Pool(pool_size) as p:
        result_list = list(
            tqdm(
                p.imap_unordered(parallel_get_back_img, pool_list), total=len(pool_list)
            )
        )

    print("Displaying heatmap...")
    sorted_result_list = sorted(result_list, key=lambda x: x[0])
    img_list = [img[1] for img in sorted_result_list]
    vis_result_list = display_heatmap_parallel(
        danger_score_list,
        vec_data_list,
        img_list,
        name_list,
        save_dir,
        grid_size,
        scale=scale,
        min_score=0,
    )

    print("Creating movie...")
    sorted_vis_result_list = sorted(vis_result_list, key=lambda x: x[0])
    vis_img_list = [img[1] for img in sorted_vis_result_list]

    create_movie_from_img_list(vis_img_list, save_dir, "heatmap")


def crop_map(map_data, crop_area):
    return map_data[crop_area[1] : crop_area[3], crop_area[0] : crop_area[2]]


def crop_vec_data(vec_data, crop_area):
    crop_vec_data = vec_data[
        (vec_data[:, 0] >= crop_area[0])
        & (vec_data[:, 0] <= crop_area[2])
        & (vec_data[:, 1] >= crop_area[1])
        & (vec_data[:, 1] <= crop_area[3])
    ]
    crop_vec_data[:, 0] -= crop_area[0]
    crop_vec_data[:, 1] -= crop_area[1]
    return crop_vec_data


def create_movie_from_img_list(img_list, save_dir, name):
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(
        os.path.join(save_dir, f"{name}.mp4"),
        fourcc,
        10.0,
        (img_list[0].shape[1], img_list[0].shape[0]),
    )
    for img in tqdm(img_list):
        out.write(img)
    out.release()


def create_movie_from_img_dir(img_dir, save_dir, name):
    img_list = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    img_list = [cv2.imread(img_path) for img_path in img_list]
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(
        os.path.join(save_dir, f"{name}.mp4"),
        fourcc,
        10.0,
        (img_list[0].shape[1], img_list[0].shape[0]),
    )
    for img in tqdm(img_list):
        out.write(img)
    out.release()


def search_vec_data(vec_data_list):
    norm_data = [np.linalg.norm(data[:, 2:], axis=1) for data in vec_data_list]
    plt.hist(norm_data, bins=100)
    plt.savefig("BMVC/vec_data_main.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_base_dir", type=str, default="demo/risk_heatmap")
    parser.add_argument(
        "--img_dir",
        type=str,
        default="demo/img",
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="demo/crs_results/debug",
    )
    parser.add_argument(
        "--smoothing",
        action="store_true",
        help="whether use smoothed map data",
    )
    parser.add_argument(
        "--path2bev_matrix", type=str, default="crs/bev/WorldPorters_8K_matrix.txt"
    )
    parser.add_argument(
        "--path2bev_size", type=str, default="crs/size/WorldPorters_8K_size.txt"
    )
    parser.add_argument(
        "--frame_range",
        type=int,
        nargs=2,
        default=None,
        help="frame range as two integers: start end",
    )
    parser.add_argument(
        "--crop_area",
        type=int,
        nargs=4,
        default=None,
        help="crop area as four integers: x1 y1 x2 y2",
    )
    parser.add_argument("--scale", type=int, default=5)
    return parser.parse_args()


def run_main():
    args = get_args()
    main(
        args.save_base_dir,
        args.img_dir,
        args.source_dir,
        frame_range=args.frame_range,
        crop_area=args.crop_area,
        path2bev_matrix=args.path2bev_matrix,
        path2bev_size=args.path2bev_size,
        scale=args.scale,
        smoothing=args.smoothing,
    )


if __name__ == "__main__":
    run_main()
