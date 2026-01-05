import cv2
import glob
from tqdm import tqdm
import argparse
import numpy as np
import os
from multiprocessing import Pool


def visualize_detection(
    detection_dir,
    path2video,
    save_dir,
    threshold=0.5,
    freq=1,
    downscale_ratio=1,
):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(path2video)
    path2detection_list = sorted(glob.glob(f"{detection_dir}/*.txt"))
    if freq > 1:
        path2detection_list = path2detection_list[::freq]
    pool_list = []
    for path2detection in path2detection_list:
        frame_name = path2detection.split("/")[-1].split(".")[0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_name))
        ret, img = cap.read()
        if not ret:
            break
        pool_list.append(
            [path2detection, img, save_dir, frame_name, threshold, downscale_ratio]
        )
    # pool_size = int(min(len(pool_list), os.cpu_count(), 32))
    # with Pool(pool_size) as p:
    #     list(
    #         tqdm(
    #             p.imap_unordered(plot_det, pool_list),
    #             total=len(pool_list),
    #             desc="Plotting Detection",
    #         )
    #     )
    for inputs in tqdm(pool_list, desc="Plotting Detection"):
        plot_det(inputs)


def plot_det(inputs):
    path2detection, img, save_dir, frame_name, threshold, downscale_ratio = inputs
    det = np.loadtxt(path2detection)
    det = det[det[:, 2] > threshold]
    img = cv2.resize(img, None, fx=downscale_ratio, fy=downscale_ratio)
    for d in det:
        x, y, _ = d
        x = int(x * downscale_ratio)
        y = int(y * downscale_ratio)
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    cv2.imwrite(f"{save_dir}/{frame_name}.jpg", img)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection_dir", type=str, required=True)
    parser.add_argument("--path2video", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--freq", type=int, default=1)
    parser.add_argument("--downscale_ratio", type=float, default=1)
    return parser.parse_args()


def main():
    args = get_args()
    detection_dir = args.detection_dir
    path2video = args.path2video
    save_dir = args.save_dir
    threshold = args.threshold
    freq = args.freq
    downscale_ratio = args.downscale_ratio
    visualize_detection(
        detection_dir,
        path2video,
        save_dir,
        threshold,
        freq,
        downscale_ratio,
    )


if __name__ == "__main__":
    main()
