import glob
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PedestrianClassConfig:
    """
    class_name: str
    area: list[float] >> [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    angle_range: list[float] | None = None
    speed_range: list[float] | None = None
    """

    class_name: str
    area: np.ndarray
    angle_range: list[float] | None = None
    speed_range: list[float] | None = None


class PedestrianClassifier:
    def __init__(self, class_config_list: list[PedestrianClassConfig]):
        self.class_config_list = class_config_list
        self.class_memory = {}

    def classify(self, tracklet):
        """
        tracklet: [frame,id,x,y]
        """
        position = tracklet[-1, 2:]
        moving_vec = tracklet[-1, 2:] - tracklet[0, 2:]
        speed = np.linalg.norm(moving_vec) / len(tracklet)
        angle = np.arctan2(moving_vec[1], moving_vec[0])
        id = int(tracklet[-1, 1])
        for class_config in self.class_config_list:
            if self.check_class_config(
                class_config,
                position,
                angle,
                speed,
            ):
                if id not in self.class_memory:
                    self.class_memory[id] = []
                self.class_memory[id].append(class_config.class_name)
                return class_config.class_name
        if id not in self.class_memory:
            self.class_memory[id] = []
        self.class_memory[id].append("unknown")
        return "unknown"

    def check_class_config(
        self,
        class_config: PedestrianClassConfig,
        position: list,
        angle: float,
        speed: float,
    ) -> bool:
        in_area = cv2.pointPolygonTest(
            class_config.area.astype(np.float32),
            (position[0].astype(np.float32), position[1].astype(np.float32)),
            False,
        )
        if in_area < 0:
            return False
        in_angle = class_config.angle_range is None or self.check_angle(
            angle, class_config.angle_range
        )
        if not in_angle:
            return False
        in_speed = class_config.speed_range is None or (
            class_config.speed_range[0] < speed < class_config.speed_range[1]
        )
        if not in_speed:
            return False
        return True

    def check_angle(self, angle: float, angle_range: list[float]) -> bool:
        angle = angle % (2 * np.pi)
        if angle_range[0] < angle_range[1]:
            return angle_range[0] < angle and angle < angle_range[1]
        else:
            return angle_range[0] < angle or angle < angle_range[1]


class BevTrackGenerator:
    def __init__(self, path2homography_matrix: str, track_dir: str):
        self.homography_matrix = np.loadtxt(path2homography_matrix)
        self.track_dir = track_dir

    def run(self, start_frame, end_frame):
        track = self.get_all_track(self.track_dir, start_frame, end_frame)
        bev_track = self.transform_coordinate(track)
        return bev_track

    def get_all_track(self, track_dir, start_frame, end_frame):
        # crop_area >> [xmin,ymin,xmax,ymax]
        txt_files = sorted(glob.glob(f"{track_dir}/*.txt"))
        track_list = [
            np.loadtxt(path2txt, delimiter=",")
            for path2txt in txt_files[start_frame - 1 : end_frame]
        ]
        all_track = []
        for i, track in enumerate(track_list):
            if track.ndim == 1:
                track = np.expand_dims(track, axis=0)
            frame = start_frame + i
            track = np.concatenate([np.full((len(track), 1), frame), track], axis=1)
            all_track += list(track)
        all_track = np.array(all_track)

        # all_track=[[frame,id,x,y],...]
        return all_track

    def transform_coordinate(self, track):
        """
        track=[[frame,id,x,y],...]
        return [[frame,id,x_transformed,y_transformed],...]
        """
        frame_and_id_data = track[:, :2]  # [[frame,id],...]
        track_data = track[:, 2:]  # [[x,y],...]
        track_data_transformed = cv2.perspectiveTransform(
            track_data.reshape(-1, 1, 2), self.homography_matrix
        )
        track_data_transformed = track_data_transformed.reshape(-1, 2)
        track_transformed = np.concatenate(
            [frame_and_id_data, track_data_transformed], axis=1
        )
        return track_transformed
