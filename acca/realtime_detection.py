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
RTSP_URL = f"rtsp://{HOST}:{PW}@{IP}:{PORT}/ONVIF/MediaInput?profile=def_profile1"


class RealTimeDetector:
    def __init__(self, cfg_path: Path):
        self.cfg_path = cfg_path
        self.detector = self.build_detector(cfg_path)

    def build_detector(self, conf_path: Path):
        cfg = OmegaConf.load(conf_path)
        detector = instantiate(cfg)
        return detector

    def run(self, images: List[np.ndarray]) -> List[np.ndarray]:
        results = self.detector(images)
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
    print("Detector setting DONE")

    dummy=np.zeros((100,100,3),np.uint8)
    cv2.imshow("Detection", dummy)
    cv2.waitKey(1)

    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    print(f"RTSP connecting DONE: {RTSP_URL}")

    t0=time.perf_counter()
    ret, frame = cap.read()
    t1=time.perf_counter()
    cv2.imshow("Detection", frame)
    t2=time.perf_counter()
    cv2.waitKey(1)
    t3=time.perf_counter()
    # print("INIT", f"{t1-t0:03f}",f"{t2-t1:03f}",f"{t3-t2:03f}")
    # print("INIT",t3-t0)

    print("START")
    n=0
    while True:
        n+=1
        t0=time.perf_counter()
        ret, frame = cap.read()
        t1=time.perf_counter()

        if not ret:
            break
        results = detector.run([frame])
        frame = display_results(frame, results[0])
        t2=time.perf_counter()
        cv2.imshow("Detection", frame)
        t3=time.perf_counter()
        cv2.waitKey(1)
        t4=time.perf_counter()
        print(f"{t1-t0:03f}",f"{t2-t1:03f}",f"{t3-t2:03f}",f"{t4-t3:03f}")
        # print(t3-t0/
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
