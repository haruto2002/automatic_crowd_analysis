from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2

# import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# matplotlib.use("Agg")  # GUIバックエンドを使わない
# import matplotlib.pyplot as plt


@dataclass
class AreaConfig:
    path2homography_matrix: Path
    points_dir: Path
    original_area_size: list[float]
    all_map_left_top_coor: list[float]
    scale: float
    homography_matrix: np.ndarray = field(init=False)

    def __post_init__(self):
        self.homography_matrix = np.loadtxt(self.path2homography_matrix)


class AllMapDisplayer:
    def __init__(
        self,
        path2all_map: Path,
        config_list: list[AreaConfig],
    ):
        self.path2all_map = path2all_map
        self.all_map_img = cv2.imread(path2all_map.as_posix())
        self.config_list = config_list

    def run(self, time: datetime):
        output_img = self.all_map_img.copy()
        for config in self.config_list:
            points, area = self.set_data(time, config)
            if points is None:
                continue
            output_img = self.display_on_all_map(points, area, output_img)
        output_img = self.add_time_text(output_img, time)
        return time, output_img

    def set_data(self, time, config):
        time_str = time.strftime("%Y-%m-%d_%H:%M:%S")
        path2points = list(config.points_dir.glob(f"*{time_str}.txt"))
        if len(path2points) == 0:
            return None, None
        points = np.loadtxt(path2points[0])
        if points.shape[1] == 3:
            points = points[points[:, 2] > 0.5][:, :2]

        points = self.project_points(points, config.homography_matrix)

        points_on_all_map = points * config.scale + np.array(
            config.all_map_left_top_coor
        )
        area = [
            config.all_map_left_top_coor[0],
            config.all_map_left_top_coor[1],
            config.all_map_left_top_coor[0]
            + config.original_area_size[0] * config.scale,
            config.all_map_left_top_coor[1]
            + config.original_area_size[1] * config.scale,
        ]
        return points_on_all_map, area

    def project_points(self, points, homography_matrix):
        points = cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography_matrix)
        return points.reshape(-1, 2)

    def display_on_all_map(self, points, area, all_map_img):
        for point in points:
            x, y = int(point[0]), int(point[1])
            cv2.circle(
                all_map_img,
                (x, y),
                radius=3,
                color=(0, 0, 255),
                thickness=-1,
            )
        top_left = (int(area[0]), int(area[1]))
        bottom_right = (int(area[2]), int(area[3]))
        cv2.rectangle(
            all_map_img,
            top_left,
            bottom_right,
            (0, 255, 0),
            5,
        )
        return all_map_img

    def add_time_text(self, all_map_img, time):
        time_str = time.strftime("%H:%M:%S")
        # 画面右上に白い四角、その上に黒の時刻表⽰（解像度対応）
        h, w = all_map_img.shape[:2]
        rect_width = int(w * 0.15)
        rect_height = int(h * 0.05)
        margin = int(h * 0.01)
        top_left = (w - rect_width - margin, margin)
        bottom_right = (w - margin, margin + rect_height)
        cv2.rectangle(
            all_map_img, top_left, bottom_right, (255, 255, 255), thickness=-1
        )
        font_scale = rect_height / 50
        thickness = max(1, int(rect_height / 30))
        text_size, baseline = cv2.getTextSize(
            time_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        text_x = top_left[0] + (rect_width - text_size[0]) // 2
        text_y = top_left[1] + (rect_height + text_size[1]) // 2
        cv2.putText(
            all_map_img,
            time_str,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            lineType=cv2.LINE_AA,
        )
        return all_map_img


class AllMapDisplayerMatplotlib:
    """
    matplotlibを使用して効率的に透明度を実装したAllMapDisplayer
    ポイントを一度に描画するため、処理コストが低い
    """

    def __init__(
        self,
        path2all_map: Path,
        config_list: list[AreaConfig],
        point_alpha: float = 0.3,
        point_size: int = 5,
        point_color: tuple = (1.0, 0.0, 0.0),  # RGB形式 (0-1の範囲)
    ):
        self.path2all_map = path2all_map
        self.all_map_img = cv2.imread(path2all_map.as_posix())
        # OpenCVはBGR、matplotlibはRGBなので変換
        self.all_map_img_rgb = cv2.cvtColor(self.all_map_img, cv2.COLOR_BGR2RGB)
        self.config_list = config_list
        self.point_alpha = point_alpha
        self.point_size = point_size
        self.point_color = point_color

    def run(self, time: datetime):
        # matplotlibのFigureを作成
        h, w = self.all_map_img_rgb.shape[:2]
        dpi = 100
        fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])  # フルサイズのaxes
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)  # 画像座標系に合わせてy軸を反転
        ax.axis("off")

        # 背景画像を表示
        ax.imshow(self.all_map_img_rgb, extent=[0, w, h, 0])

        # すべてのポイントを収集
        all_points = []
        all_areas = []

        for config in self.config_list:
            points, area = self.set_data(time, config)
            if points is None:
                continue
            all_points.append(points)
            all_areas.append(area)

        # すべてのポイントを一度に描画（効率的）
        if all_points:
            combined_points = np.vstack(all_points)
            ax.scatter(
                combined_points[:, 0],
                combined_points[:, 1],
                s=self.point_size**2,
                c=[self.point_color],
                alpha=self.point_alpha,
                edgecolors="none",
            )

        # エリアの矩形を描画
        # for area in all_areas:
        #     top_left = (area[0], area[1])
        #     bottom_right = (area[2], area[3])
        #     width = bottom_right[0] - top_left[0]
        #     height = bottom_right[1] - top_left[1]
        #     rect = plt.Rectangle(
        #         top_left,
        #         width,
        #         height,
        #         fill=False,
        #         edgecolor=(0, 1, 0),  # 緑色
        #         linewidth=5,
        #     )
        #     ax.add_patch(rect)

        # 時刻テキストを追加
        self.add_time_text_matplotlib(ax, time, w, h)

        # Figureから画像を取得（canvasのバッファから直接取得）
        fig.canvas.draw()
        # buffer_rgba()からRGBAバッファを取得
        buf = fig.canvas.buffer_rgba()
        # numpy配列に変換
        img_array = np.asarray(buf)
        plt.close(fig)

        # RGBAからRGBに変換
        img_rgb = img_array[:, :, :3]
        # RGBからBGRに変換（OpenCV形式）
        output_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        return time, output_img

    def set_data(self, time, config):
        time_str = time.strftime("%Y-%m-%d_%H:%M:%S")
        path2points = list(config.points_dir.glob(f"*{time_str}.txt"))
        if len(path2points) == 0:
            return None, None
        points = np.loadtxt(path2points[0])
        if points.shape[1] == 3:
            points = points[points[:, 2] > 0.5][:, :2]

        points = self.project_points(points, config.homography_matrix)

        points_on_all_map = points * config.scale + np.array(
            config.all_map_left_top_coor
        )
        area = [
            config.all_map_left_top_coor[0],
            config.all_map_left_top_coor[1],
            config.all_map_left_top_coor[0]
            + config.original_area_size[0] * config.scale,
            config.all_map_left_top_coor[1]
            + config.original_area_size[1] * config.scale,
        ]
        return points_on_all_map, area

    def project_points(self, points, homography_matrix):
        points = cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography_matrix)
        return points.reshape(-1, 2)

    def add_time_text_matplotlib(self, ax, time, w, h):
        time_str = time.strftime("%H:%M:%S")
        # 画面右上に白い四角、その上に黒の時刻表示（解像度対応）
        rect_width = int(w * 0.15)
        rect_height = int(h * 0.05)
        margin = int(h * 0.01)
        top_left_x = w - rect_width - margin
        top_left_y = margin

        # 白い矩形を描画
        rect = plt.Rectangle(
            (top_left_x, top_left_y),
            rect_width,
            rect_height,
            fill=True,
            facecolor=(1.0, 1.0, 1.0),  # 白色
            edgecolor="none",
        )
        ax.add_patch(rect)

        # テキストを追加
        text_x = top_left_x + rect_width / 2
        text_y = top_left_y + rect_height / 2
        ax.text(
            text_x,
            text_y,
            time_str,
            ha="center",
            va="center",
            fontsize=rect_height / 2,
            color=(0, 0, 0),  # 黒色
            weight="bold",
        )
