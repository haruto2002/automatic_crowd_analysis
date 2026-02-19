from typing import Dict, List

import cv2
import numpy as np


class TrackletDisplayer:
    def __init__(
        self,
        path2back_img,
        scale_ratio=1,
        class_memory: Dict[int, List[str]] = None,
    ):
        self.back_img = cv2.imread(path2back_img)
        self.scale_ratio = scale_ratio
        self.back_img = cv2.resize(
            self.back_img, None, fx=self.scale_ratio, fy=self.scale_ratio
        )
        self.class_memory = class_memory
        self.color_dict = {
            "A": (0, 0, 255),
            "B": (255, 0, 0),
            "C": (0, 255, 0),
            "unknown": (0, 0, 0),
        }

    def display(self, track):
        img_copy = self.back_img.copy()
        existing_id = track[track[:, 0] == np.max(track[:, 0])][:, 1]
        for id in existing_id:
            one_track = track[track[:, 1] == id]
            class_name = self.class_memory[id][0]
            self.class_memory[id].pop(0)
            color = self.color_dict[class_name.split("_")[0]]
            use_length = len(one_track)
            for i, data in enumerate(one_track[-use_length:]):
                p = data[2:]  # coordinate
                if i == use_length - 1:
                    p_size = 3
                else:
                    p_size = 1

                cv2.circle(
                    img_copy,
                    (
                        int(p[0] * self.scale_ratio),
                        int(p[1] * self.scale_ratio),
                    ),
                    p_size,
                    color,
                    -1,
                    lineType=cv2.LINE_AA,
                )
        return img_copy
