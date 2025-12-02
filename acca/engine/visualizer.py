from pathlib import Path

from visualize.detection.plot_detection import visualize_detection
from visualize.tracking.display_track import visualize_trajectory


class DetectionVisualizer:
    def __init__(self, threshold=0.5, freq=1, downscale_ratio=1):
        self.threshold = float(threshold)
        self.freq = int(freq)
        self.downscale_ratio = float(downscale_ratio)

    def visualize(self, detection_dir: Path, video_path: Path, save_dir: Path):
        visualize_detection(
            detection_dir.as_posix(),
            video_path.as_posix(),
            save_dir.as_posix(),
            threshold=self.threshold,
            freq=self.freq,
            downscale_ratio=self.downscale_ratio,
        )


class TrackingVisualizer:
    def __init__(self, freq=30, vis_length=30, start_frame=None, end_frame=None, downscale_ratio=1):
        self.freq = int(freq)
        self.vis_length = int(vis_length)
        self.start_frame = int(start_frame) if (start_frame is not None and start_frame != "None") else None
        self.end_frame = int(end_frame) if (end_frame is not None and end_frame != "None") else None
        self.downscale_ratio = float(downscale_ratio)

    def visualize(self, track_dir: Path, video_path: Path, save_path: Path):
        visualize_trajectory(
            track_dir.as_posix(),
            video_path.as_posix(),
            save_path.as_posix(),
            freq=self.freq,
            vis_length=self.vis_length,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            downscale_ratio=self.downscale_ratio,
        )
