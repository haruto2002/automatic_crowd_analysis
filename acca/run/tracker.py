from bytetrack.run_point_tracking import tracking
from pydantic import BaseModel
import os


class TrackerArgs(BaseModel):
    img_h_size: int = 4320
    img_w_size: int = 7680
    track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 10.0
    distance_metric: str = "euclidean"
    mot20: bool = False


class Tracker:
    def __init__(
        self,
        img_h_size: int,
        img_w_size: int,
        track_thresh: float,
        track_buffer: int,
        match_thresh: float,
        distance_metric: str,
    ):
        self.args = TrackerArgs(
            img_h_size=img_h_size,
            img_w_size=img_w_size,
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            distance_metric=distance_metric,
        )
        self.disable_tqdm = False

    def run(self, detection_data_dir: str, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        tracking(
            self.args, detection_data_dir, save_dir, disable_tqdm=self.disable_tqdm
        )
