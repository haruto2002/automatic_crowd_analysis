from visualize.detection.plot_detection import visualize_detection
from visualize.tracking.display_track import visualize_trajectory


class DetectionVisualizer:
    def __init__(self, threshold=0.5, freq=1, downscale_ratio=1):
        self.threshold = float(threshold)
        self.freq = int(freq)
        self.downscale_ratio = float(downscale_ratio)

    def visualize(self, detection_dir, video_path, save_dir):
        visualize_detection(
            detection_dir,
            video_path,
            save_dir,
            threshold=self.threshold,
            freq=self.freq,
            downscale_ratio=self.downscale_ratio,
        )


class TrackingVisualizer:
    def __init__(self, freq=30, start_frame=None, end_frame=None, downscale_ratio=1):
        self.freq = int(freq)
        self.start_frame = (
            int(start_frame)
            if (start_frame is not None and start_frame != "None")
            else None
        )
        self.end_frame = (
            int(end_frame) if (end_frame is not None and end_frame != "None") else None
        )
        self.downscale_ratio = float(downscale_ratio)

    def visualize(self, track_dir, video_path, save_path):
        visualize_trajectory(
            track_dir,
            video_path,
            save_path,
            freq=self.freq,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            downscale_ratio=self.downscale_ratio,
        )
