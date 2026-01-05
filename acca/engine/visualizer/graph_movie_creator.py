from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

import cv2
from engine.common.utils import HanabiImageData
from tqdm import tqdm


class GraphMovieCreator:
    def __init__(
        self,
        threshold: float = 0.5,
        downscale_ratio: float = 1,
        fps: int = 10,
        bar: bool = True,
    ):
        self.threshold = threshold
        self.downscale_ratio = downscale_ratio
        self.fps = fps
        self.disable_tqdm = not bar

        self.detection_plot_size = int(10 * self.downscale_ratio)
        self.detection_plot_color = (0, 0, 255)

    def __call__(self, hanabi_image_data: HanabiImageData):
        self.save_name = f"{hanabi_image_data.event_name}_{hanabi_image_data.place_name}"
        self.set_data(hanabi_image_data)
        frame_list = self.create_frames()
        self.create_video(frame_list)

    def set_data(self, hanabi_image_data: HanabiImageData):
        self.detection_save_dir = hanabi_image_data.get_output_dir_for_phase(["vis", "detection", "detection"])
        self.graph_save_dir = hanabi_image_data.get_output_dir_for_phase(["vis", "detection", "graph"])
        self.all_vis_save_dir = hanabi_image_data.get_output_dir_for_phase(["vis", "detection", "all_vis"])
        self.graph_movie_save_dir = hanabi_image_data.get_output_dir_for_phase(["vis", "detection", "graph_movie"])
        self.num_people_save_dir = hanabi_image_data.get_output_dir_for_phase(["vis", "detection", "num_people"])

        self.detection_dir = hanabi_image_data.phase_outputdir.detection
        self.detection_path_list = sorted(self.detection_dir.glob("*.txt"))

        self.img_path_list = hanabi_image_data.image_paths
        self.img_width, self.img_height = self.set_img_size()
        self.num_people_df = self.load_num_people()

    def create_frames(self):
        frame_list = []
        for index in tqdm(
            range(len(self.img_path_list)), desc="Creating Graph Movie Frames", disable=self.disable_tqdm
        ):
            img_name = self.img_path_list[index].stem
            det_img = self.plot_det(index, img_name)
            graph_img = self.plot_graph(index, img_name)
            all_vis_img = np.concatenate([det_img, graph_img], axis=0)
            cv2.imwrite(self.all_vis_save_dir.joinpath(f"{img_name}.jpg"), all_vis_img)
            frame_list.append(all_vis_img)
        return frame_list

    def create_video(self, frame_list: list[np.ndarray]):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        height = frame_list[0].shape[0]
        width = frame_list[0].shape[1]
        out = cv2.VideoWriter(
            self.graph_movie_save_dir.joinpath(f"{self.save_name}.mp4"),
            fourcc,
            self.fps,
            (width, height),
        )
        for frame in tqdm(frame_list, desc="Creating Graph Movie Video", disable=self.disable_tqdm):
            out.write(frame)
        out.release()

    def set_img_size(self):
        img = cv2.imread(self.img_path_list[0])
        img_width = int(img.shape[1] * self.downscale_ratio)
        img_height = int(img.shape[0] * self.downscale_ratio)
        return img_width, img_height

    def load_num_people(self):
        path2txt_list = sorted(self.detection_dir.glob("*.txt"))
        cnt_data = []
        for path2txt in path2txt_list:
            result = np.loadtxt(path2txt)
            if result.size == 0:
                num_people = 0
            else:
                det_result = result[result[:, 2] > self.threshold]
                num_people = len(det_result)
            time = path2txt.stem.split("_")[-1]
            cnt_data.append([time, num_people])
        df = pd.DataFrame(cnt_data, columns=["time", "num_people"])
        df.to_csv(
            self.num_people_save_dir.joinpath(f"{self.save_name}.csv"),
            index=False,
        )
        return df

    def plot_det(self, index: int, img_name: str):
        path2detection = self.detection_path_list[index]
        path2img = self.img_path_list[index]
        img = cv2.imread(path2img)
        det = np.loadtxt(path2detection)
        det = np.array([]) if det.size == 0 else det[det[:, 2] > self.threshold]
        img = cv2.resize(img, (self.img_width, self.img_height))
        for d in det:
            x, y, _ = d
            x = int(x * self.downscale_ratio)
            y = int(y * self.downscale_ratio)
            cv2.circle(img, (x, y), self.detection_plot_size, self.detection_plot_color, -1)

        cv2.imwrite(self.detection_save_dir.joinpath(f"{img_name}.jpg"), img)

        return img

    def plot_graph(self, index: int, img_name: str):
        fig_size = (16, 9)
        dpi = self.img_width / fig_size[0]
        assert self.img_height / fig_size[1] == dpi
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        canvas = FigureCanvas(fig)

        self.num_people_df["time"] = pd.to_datetime(self.num_people_df["time"], format="%H:%M:%S")

        total_time_range = self.num_people_df["time"].max() - self.num_people_df["time"].min()
        x_margin = total_time_range * 0.05
        max_people = self.num_people_df["num_people"].max()
        digit = 10 if max_people < 100 else 10 ** (len(str(int(max_people))) - 2)
        y_max = (max_people // digit + 2) * digit

        ax.plot(
            self.num_people_df["time"][: index + 1],
            self.num_people_df["num_people"][: index + 1],
            marker="o",
            linestyle="-",
        )
        ax.set_xlabel("Time", fontsize=22)
        ax.set_ylabel("Number of People", fontsize=22)
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlim(
            self.num_people_df["time"].min() - x_margin,
            self.num_people_df["time"].max() + x_margin,
        )
        ax.set_ylim(0, y_max)
        ax.tick_params(labelsize=20)
        fig.tight_layout()

        # figをimgに変換
        canvas.draw()
        img = np.array(canvas.buffer_rgba())
        img = img[:, :, :3]
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        plt.close(fig)

        cv2.imwrite(self.graph_save_dir.joinpath(f"{img_name}.jpg"), img_bgr)

        return img_bgr
