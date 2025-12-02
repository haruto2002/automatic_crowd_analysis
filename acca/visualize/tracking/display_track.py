import argparse
import glob

import cv2
import numpy as np
from tqdm import tqdm


def display_tracking_result(track, img, color_list, downscale_ratio=1):
    img_copy = cv2.resize(img, None, fx=downscale_ratio, fy=downscale_ratio)
    existing_id = track[track[:, 0] == np.max(track[:, 0])][:, 1]
    for id in existing_id:
        one_track = track[track[:, 1] == id]
        use_length = len(one_track)
        for i, data in enumerate(one_track[-use_length:]):
            p = data[2:]  # coordinate
            if i == use_length - 1:
                p_size = 7
            else:
                p_size = 1

            cv2.circle(
                img_copy,
                (int(p[0] * downscale_ratio), int(p[1] * downscale_ratio)),
                p_size,
                color_list[(int(id) - 1) % len(color_list)],
                -1,
                lineType=cv2.LINE_AA,
            )
    return img_copy


def create_mov(img_list, save_path, fps):
    img = img_list[0]
    h, w, c = img.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    for img in tqdm(img_list, desc="Creating Movie"):
        out.write(img)
    out.release()


def display_on_frame(inputs):
    track_dir, start_frame, end_frame, img, color_list, downscale_ratio, order = inputs
    track = get_all_track(track_dir, start_frame, end_frame)
    output = display_tracking_result(track, img, color_list, downscale_ratio)
    return output, order


def get_all_track(track_dir, start_frame, end_frame):
    # crop_area >> [xmin,ymin,xmax,ymax]
    txt_files = sorted(glob.glob(f"{track_dir}/*.txt"))
    track_list = [np.loadtxt(path2txt, delimiter=",") for path2txt in txt_files[start_frame - 1 : end_frame]]
    all_track = []
    for i, track in enumerate(track_list):
        frame = start_frame + i
        track = np.concatenate([np.full((len(track), 1), frame), track], axis=1)
        all_track += list(track)
    all_track = np.array(all_track)

    # all_track=[[frame,id,x,y],...]
    return all_track


def visualize_trajectory(
    track_dir,
    path2video,
    save_path,
    freq,
    vis_length=30,
    start_frame=None,
    end_frame=None,
    downscale_ratio=1,
):
    print("Loading video...")
    cap = cv2.VideoCapture(path2video)

    print("Generating color list...")
    np.random.seed(seed=32)
    color_list = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(10000)]

    if start_frame is None:
        start_frame = 1
    if end_frame is None:
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Generating display inputs...")
    display_inputs = []
    order = 0
    for e in range(start_frame, end_frame + 1, freq):
        s = max(1, e - vis_length)
        cap.set(cv2.CAP_PROP_POS_FRAMES, e)
        ret, img = cap.read()
        if not ret:
            break
        display_inputs.append((track_dir, s, e, img, color_list, downscale_ratio, order))
        order += 1

    print("Displaying frames...")
    # pool_size = int(min(len(display_inputs), os.cpu_count(), 32))
    # with Pool(pool_size) as p:
    #     tracking_vis = list(
    #         tqdm(
    #             p.imap_unordered(display_on_frame, display_inputs),
    #             total=len(display_inputs),
    #             desc="Displaying Frames",
    #         )
    #     )
    tracking_vis = [display_on_frame(inputs) for inputs in tqdm(display_inputs, desc="Displaying Frames")]

    img_list = [output for output, _ in sorted(tracking_vis, key=lambda x: x[-1])]
    fps = 30 / freq
    create_mov(img_list, save_path, fps)


def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_dir", type=str, default="demo/track")
    parser.add_argument("--img_dir", type=str, default="demo/img")
    parser.add_argument("--save_path", type=str, default="demo/track_vis")
    parser.add_argument("--freq", type=int, default=1, help="display frame interval")
    parser.add_argument("--vis_length", type=int, default=30, help="display frame length")
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--downscale_ratio", type=float, default=1)
    args = parser.parse_args()

    visualize_trajectory(
        args.track_dir,
        args.img_dir,
        args.save_path,
        args.freq,
        vis_length=args.vis_length,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        downscale_ratio=args.downscale_ratio,
    )


if __name__ == "__main__":
    run_main()
