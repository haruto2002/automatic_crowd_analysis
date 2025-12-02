from run.detector import Detector
from run_debug.tracker import Tracker
from run_debug.visualizer import DetectionVisualizer
from run_debug.visualizer import TrackingVisualizer


def set_detector():
    cfg_path = "p2pnet/conf/p2p.yaml"
    weight_path = "cutout.pth"
    gpu_id = 0
    image_size = [4320, 7680]
    dtype = "float16"
    detector = Detector(cfg_path, weight_path, gpu_id, image_size, dtype)
    return detector


def set_detection_visualizer():
    threshold = 0.5
    freq = 10
    downscale_ratio = 0.5
    detection_visualizer = DetectionVisualizer(threshold, freq, downscale_ratio)
    return detection_visualizer


def set_tracker():
    img_h_size = 4320
    img_w_size = 7680
    track_thresh = 0.6
    track_buffer = 30
    match_thresh = 10.0
    distance_metric = "euclidean"
    tracker = Tracker(
        img_h_size,
        img_w_size,
        track_thresh,
        track_buffer,
        match_thresh,
        distance_metric,
    )
    return tracker


def set_tracking_visualizer():
    freq = 5
    start_frame = None
    end_frame = None
    downscale_ratio = 0.5
    tracking_visualizer = TrackingVisualizer(
        freq, start_frame, end_frame, downscale_ratio
    )
    return tracking_visualizer


def main():
    print("start")
    video_path = "/groups/gaa50073/nakayama-haruto/hanabi_videos/202408_Yokohama_WorldPorters/DSC_7475.MOV"
    det_save_dir = "demo_outputs/202408_Yokohama_WorldPorters/DSC_7475/detection"
    # tracker_save_dir = "demo_outputs/demo_1s_1/track"
    print("set detector")
    detector = set_detector()
    print("Done")
    print("run detector")
    detector.run(video_path, det_save_dir)
    # tracker = set_tracker()
    # tracker.run(det_save_dir, tracker_save_dir)
    # det_vis_save_dir = "demo_outputs/demo_1s_1/vis/detection"
    # track_vis_save_path = "demo_outputs/demo_1s_1/vis/track.mp4"
    # detection_visualizer = set_detection_visualizer()
    # detection_visualizer.visualize(det_save_dir, video_path, det_vis_save_dir)
    # tracking_visualizer = set_tracking_visualizer()
    # tracking_visualizer.visualize(tracker_save_dir, video_path, track_vis_save_path)


def vis():
    print("start")
    video_path = "demo_videos/demo_1s_1.mov"
    det_save_dir = "demo_outputs/demo_videos/results/demo_1s_1/detection"
    det_vis_save_dir = "demo_outputs/demo_videos/results/demo_1s_1/vis/detection"
    detection_visualizer = set_detection_visualizer()
    detection_visualizer.visualize(det_save_dir, video_path, det_vis_save_dir)


if __name__ == "__main__":
    main()
    # vis()
