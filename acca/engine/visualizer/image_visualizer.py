from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cv2
from tqdm import tqdm


class DetectionVisualizerForImage:
    def __init__(self, threshold=0.5, freq=1, downscale_ratio=1, img_extension="jpg", with_movie=False, fps=30):
        self.threshold = float(threshold)
        self.freq = int(freq)
        self.downscale_ratio = float(downscale_ratio)
        self.img_extension = img_extension
        self.plot_size = int(10 * self.downscale_ratio)
        self.plot_color = (0, 0, 255)

        self.with_movie = with_movie
        self.fps = fps

    def visualize(self, detection_dir: Path, img_dir: Path, save_dir: Path, movie_name="detection"):
        save_dir.mkdir(parents=True, exist_ok=True)
        img_list = self.plot_det(detection_dir, img_dir, save_dir, self.threshold, self.freq, self.downscale_ratio)
        if self.with_movie:
            self.create_movie(img_list, save_dir, self.fps, movie_name)

    def plot_det(self, detection_dir: Path, img_dir: Path, save_dir: Path, threshold=0.5, freq=1, downscale_ratio=1):
        save_dir = save_dir.joinpath("frames")
        save_dir.mkdir(parents=True, exist_ok=True)
        path2detection_list = sorted(detection_dir.glob("*.txt"))
        if freq > 1:
            path2detection_list = path2detection_list[::freq]
        img_list = []
        for path2detection in tqdm(path2detection_list, desc="Plotting Detection"):
            frame_name = path2detection.stem
            img = cv2.imread(img_dir.joinpath(f"{frame_name}.{self.img_extension}"))
            det = np.loadtxt(path2detection)
            det = det[det[:, 2] > threshold]
            img = cv2.resize(img, None, fx=downscale_ratio, fy=downscale_ratio)
            for d in det:
                x, y, _ = d
                x = int(x * downscale_ratio)
                y = int(y * downscale_ratio)
                cv2.circle(img, (x, y), self.plot_size, self.plot_color, -1)
            cv2.imwrite(save_dir.joinpath(f"{frame_name}.{self.img_extension}"), img)
            img_list.append(img)
        return img_list

    def create_movie(self, img_list: list[np.ndarray], save_dir: Path, fps=30, movie_name="detection"):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            save_dir.joinpath(f"{movie_name}.mp4"), fourcc, fps, (img_list[0].shape[1], img_list[0].shape[0])
        )
        for img in tqdm(img_list, desc="Creating Movie"):
            out.write(img)
        out.release()


class GraphCreator:
    def __init__(self, threshold=0.5):
        self.threshold = float(threshold)
        self.font_size = 25
        self.title_font_size = 30

    def create_graph(self, detection_dir: Path, save_dir: Path, title: str):
        save_dir.mkdir(parents=True, exist_ok=True)
        df = self.load_num_people(detection_dir, save_dir)
        self.plot_num_people(save_dir, df, title)

    def plot_num_people(self, save_dir: Path, df: pd.DataFrame, title: str):
        start_time = df["time"].iloc[0]
        end_time = df["time"].iloc[-1]
        df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S")
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.plot(df["time"], df["num_people"], linestyle="-")
        # ax.fill_between(df["time"], 0, df["num_people"], alpha=0.8, color="g", linewidth=0)
        ax.tick_params(labelsize=self.font_size)
        ax.set_xlabel("Date time", fontsize=self.font_size)
        ax.set_xlim(df["time"].min(), df["time"].max())
        ax.set_ylabel("Number of People", fontsize=self.font_size)
        max_people = df["num_people"].max()
        digit = 10 if max_people < 100 else 10 ** (len(str(int(max_people))) - 2)
        y_max = (max_people // digit + 2) * digit
        ax.set_ylim(0, y_max)
        ax.grid(True, axis="y", linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.title(title, fontsize=self.title_font_size)
        plt.tight_layout()
        plt.savefig(save_dir.joinpath(f"{title}_{start_time}_{end_time}.jpg"))

    def load_num_people(self, detection_dir: Path, save_dir: Path):
        path2txt_list = sorted(detection_dir.glob("*.txt"))
        cnt_data = []
        for path2txt in path2txt_list:
            result = np.loadtxt(path2txt)
            det_result = result[result[:, 2] > self.threshold]
            num_people = len(det_result)
            time = path2txt.stem.split("_")[-1]
            cnt_data.append([time, num_people])
        df = pd.DataFrame(cnt_data, columns=["time", "num_people"])
        df.to_csv(save_dir.joinpath("num_people.csv"), index=False)
        return df
