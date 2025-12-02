import argparse
import glob
import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from bytetrack.track_utils.byte_tracker import BYTETracker
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Point Tracker")
    parser.add_argument(
        "--detection_data_dir",
        type=str,
        default="byte_track/full_detection/WorldPorters_noon",
    )
    parser.add_argument("--save_dir", type=str, default="byte_track/track_data")
    parser.add_argument("--img_h_size", type=int, default=4320)
    parser.add_argument("--img_w_size", type=int, default=7680)
    parser.add_argument("--track_thresh", type=float, default=0.6)
    parser.add_argument("--track_buffer", type=int, default=30)
    parser.add_argument("--match_thresh", type=float, default=10.0)
    parser.add_argument(
        "--distance_metric",
        type=str,
        default="euclidean",
        choices=["maha", "euclidean"],
    )
    return parser


def _load_txt_to_np(path: str) -> np.ndarray:
    # 空ファイルの場合は空の配列を返す（列数は3: x, y, score）
    if os.path.getsize(path) == 0:
        return np.empty((0, 3), dtype=np.float32)
    df = pd.read_csv(path, header=None, sep=r"\s+", dtype=np.float32)
    if df.empty:
        return np.empty((0, 3), dtype=np.float32)
    return df.to_numpy()


def load_all_detections(paths, workers: int, disable_tqdm: bool) -> list[np.ndarray]:
    if workers <= 1:
        return [_load_txt_to_np(p) for p in tqdm(paths, desc="Loading detections", disable=disable_tqdm)]
    dets = [None] * len(paths)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_load_txt_to_np, p): i for i, p in enumerate(paths)}
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Loading detections (parallel)",
            disable=disable_tqdm,
        ):
            i = futures[fut]
            dets[i] = fut.result()
    return dets


def interpolate_tracklets(tracklets):
    for track_id, tracklet in tracklets.items():
        frames = tracklet["frames"]
        points = tracklet["points"]
        i = 0
        while i < len(frames) - 1:
            frame1 = frames[i]
            frame2 = frames[i + 1]
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            if frame1 + 1 == frame2:
                i += 1
                continue
            if frame1 + 1 < frame2:
                num_missing_frames = frame2 - frame1 - 1
                for j in range(1, num_missing_frames + 1):
                    x = (x1 * (num_missing_frames + 1 - j) + x2 * j) / (num_missing_frames + 1)
                    y = (y1 * (num_missing_frames + 1 - j) + y2 * j) / (num_missing_frames + 1)
                    points.insert(i + j, (x, y))
                    frames.insert(i + j, frame1 + j)
                i += num_missing_frames
            i += 1
    return tracklets


def _write_frame_txt(path: str, rows: list[list[float]]) -> None:
    # np.savetxt tends to be slow with many small files. Use string concatenation for batch output.
    # Format: id(int), x, y separated by commas
    if not rows:
        open(path, "w").close()
        return
    # Sort & stringify
    rows.sort(key=lambda r: r[0])
    lines = [f"{int(r[0])},{float(r[1]):.6f},{float(r[2]):.6f}\n" for r in rows]
    with open(path, "w") as f:
        f.writelines(lines)


def create_frame_txt_files(
    interpolated_tracklets,
    save_dir: str,
    total_frames: int,
    workers: int,
    disable_tqdm: bool = False,
):
    os.makedirs(save_dir, exist_ok=True)
    frame_data = defaultdict(list)

    # 各トラックの (frame, point) をフレーム側に寄せ集め
    for track_id, tr in interpolated_tracklets.items():
        tid = int(track_id)
        frames = tr["frames"]
        points = tr["points"]
        append_local = frame_data.__getitem__  # ルックアップ短縮
        for f, (x, y) in zip(frames, points):
            append_local(f).append([tid, x, y])

    # 並列書き出し
    tasks = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for frame_num in range(1, total_frames + 1):
            filename = os.path.join(save_dir, f"{frame_num:04d}.txt")
            rows = frame_data.get(frame_num, [])
            tasks.append(ex.submit(_write_frame_txt, filename, rows))
        # 進捗バー
        for _ in tqdm(
            as_completed(tasks),
            total=len(tasks),
            desc="Writing frames",
            disable=disable_tqdm,
        ):
            pass


def main():
    parser = make_parser()
    args = parser.parse_args()

    # Set log level
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, args.log_level.upper()))
    disable_tqdm = logger.level >= logging.ERROR

    logger.info("=== ByteTrack Point Tracker started ===")

    tracking(args, disable_tqdm)


def tracking(args, detection_data_dir, save_dir, disable_tqdm):
    # Initialize tracker
    tracker = BYTETracker(args)
    tracker.distance_metric = args.distance_metric
    img_size = (int(args.img_h_size), int(args.img_w_size))

    loader_workers = 1
    path2det_list = sorted(glob.glob(f"{detection_data_dir}/*.txt"))
    dets_list = load_all_detections(path2det_list, workers=loader_workers, disable_tqdm=disable_tqdm)

    all_data = []
    append_all = all_data.append

    for detections in tqdm(dets_list, desc="Tracking", disable=disable_tqdm):
        online_targets = tracker.update(detections, img_size, img_size)

        n = len(online_targets)
        if n == 0:
            append_all(np.empty((0, 3), dtype=np.float32))
            continue

        out = np.empty((n, 3), dtype=np.float32)
        for i, t in enumerate(online_targets):
            dp = t.detection_point  # [x, y] (list/ndarray)
            out[i, 0] = t.track_id
            out[i, 1] = dp[0]
            out[i, 2] = dp[1]
        append_all(out)

    tracklets = {}
    for frame_idx, frame_arr in enumerate(all_data, start=1):
        if frame_arr.size == 0:
            continue
        # frame_arr: (N,3) = [id, x, y]
        tids = frame_arr[:, 0].astype(np.int64)
        xs = frame_arr[:, 1]
        ys = frame_arr[:, 2]
        for tid, x, y in zip(tids, xs, ys):
            tr = tracklets.get(tid)
            if tr is None:
                tr = {"frames": [], "points": []}
                tracklets[tid] = tr
            tr["frames"].append(frame_idx)
            tr["points"].append((float(x), float(y)))

    interpolated_tracklets = interpolate_tracklets(tracklets)

    save_workers = 1
    create_frame_txt_files(
        interpolated_tracklets,
        save_dir,
        total_frames=len(dets_list),
        workers=save_workers,
        disable_tqdm=disable_tqdm,
    )


if __name__ == "__main__":
    main()
