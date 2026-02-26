import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

HOST = "member"
IP = "192.168.0.10"
PORT = 554
PW = "AIST-rwdc"
RTSP_URL = f"rtsp://{HOST}:{PW}@{IP}:{PORT}/ONVIF/MediaInput?profile=def_profile4"


class RealTimeDetector:
    def __init__(self, cfg_path: Path):
        self.cfg_path = cfg_path
        self.detector = self.build_detector(cfg_path)

    def build_detector(self, conf_path: Path):
        cfg = OmegaConf.load(conf_path)
        detector = instantiate(cfg)
        return detector

    def run(self, images: List[np.ndarray]) -> List[np.ndarray]:
        time_start = time.time()
        results = self.detector(images)
        time_end = time.time()
        print(
            f"Time taken: {time_end - time_start} seconds, {sum(len(result) for result in results)} people detected"
        )
        return results


def display_results(
    img: np.ndarray,
    detection: np.ndarray,
    point_size: int = 5,
    color: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    for x, y, _ in detection:
        cv2.circle(img, (int(x), int(y)), point_size, color, -1)
    return img


def main():
    cfg_path = Path("acca/conf/rt_conf/p2pnet.yaml")
    detector = RealTimeDetector(cfg_path)
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = detector.run([frame])
        frame = display_results(frame, results[0])
        cv2.imshow("frame", frame)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
