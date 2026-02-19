import os
from collections import Counter
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
from pedestrian_classifier import (
    BevTrackGenerator,
    PedestrianClassConfig,
    PedestrianClassifier,
)
from tqdm import tqdm
from tracklet_displayer import TrackletDisplayer


def get_area(path2area: str):
    """
    CSVファイルから座標を読み込み、4点ずつareaとして分割する

    Args:
        path2area: CSVファイルのパス（x,y形式）

    Returns:
        area_list: 各areaが4点からなるnumpy配列のリスト
    """
    df = pd.read_csv(path2area)
    area_list = []

    # 座標データを取得（ヘッダー行を除く）
    points = df[["x", "y"]].values

    # 4点ずつに分割
    for i in range(0, len(points), 4):
        if i + 4 <= len(points):
            area = np.array(points[i : i + 4])
            area_list.append(area)

    return area_list


def set_angle(path2angle: str, margin: float):
    """
    CSVファイルから2点を読み込み、境界線の角度範囲を取得する

    Args:
        path2angle: CSVファイルのパス（x,y形式、2行で1本の線）

    Returns:
        angle_range_list: [min_angle, max_angle]の形式のリスト（ラジアン単位）
    """
    df = pd.read_csv(path2angle)
    angle_range_list = []
    angle_range_list_reverse = []
    angle_range_list_ambiguous_right = []
    angle_range_list_ambiguous_left = []

    points = df[["x", "y"]].values

    # 2点を取得（最初の2行）
    if len(points) < 2:
        raise ValueError("CSVファイルには少なくとも2点の座標が必要です")
    if len(points) % 2 != 0:
        raise ValueError("CSVファイルには偶数個の座標が必要です")

    for i in range(0, len(points), 2):
        point1 = points[i]
        point2 = points[i + 1]

        # 2点間のベクトルを計算
        vec = point1 - point2

        # 角度を計算（度単位）
        angle = np.arctan2(vec[1], vec[0])

        # 角度範囲を設定（線の方向±π、または線の方向から線の方向+π）
        # 境界線として、線の方向からπの範囲を返す（ラジアンのまま）
        min_angle = angle
        max_angle = angle + np.pi

        # 角度を0から2πの範囲に正規化
        angle_range_list.append(
            [(min_angle + margin) % (2 * np.pi), (max_angle - margin) % (2 * np.pi)]
        )
        angle_range_list_reverse.append(
            [(max_angle + margin) % (2 * np.pi), (min_angle - margin) % (2 * np.pi)]
        )
        angle_range_list_ambiguous_right.append(
            [(min_angle - margin) % (2 * np.pi), (min_angle + margin) % (2 * np.pi)]
        )
        angle_range_list_ambiguous_left.append(
            [(max_angle - margin) % (2 * np.pi), (max_angle + margin) % (2 * np.pi)]
        )

    return (
        angle_range_list,
        angle_range_list_reverse,
        angle_range_list_ambiguous_right,
        angle_range_list_ambiguous_left,
    )


def set_classifier(class_data_dir: str):
    path2area_data = os.path.join(class_data_dir, "area.csv")
    assert os.path.exists(path2area_data), f"Area data file not found: {path2area_data}"
    path2angle_data = os.path.join(class_data_dir, "angle.csv")
    assert os.path.exists(path2angle_data), (
        f"Angle data file not found: {path2angle_data}"
    )

    area_list = get_area(path2area_data)
    (
        angle_range_list,
        angle_range_list_reverse,
        angle_range_list_ambiguous_right,
        angle_range_list_ambiguous_left,
    ) = set_angle(path2angle_data, np.deg2rad(10))
    A_class_config_list = [
        PedestrianClassConfig(
            class_name=f"A_{i + 1}", area=area, angle_range=angle_range
        )
        for i, (area, angle_range) in enumerate(zip(area_list, angle_range_list))
    ]
    B_class_config_list = [
        PedestrianClassConfig(
            class_name=f"B_{i + 1}", area=area, angle_range=angle_range
        )
        for i, (area, angle_range) in enumerate(
            zip(area_list, angle_range_list_reverse)
        )
    ]
    C_right_class_config_list = [
        PedestrianClassConfig(
            class_name=f"C_{i + 1}_right", area=area, angle_range=angle_range
        )
        for i, (area, angle_range) in enumerate(
            zip(area_list, angle_range_list_ambiguous_right)
        )
    ]
    C_left_class_config_list = [
        PedestrianClassConfig(
            class_name=f"C_{i + 1}_left", area=area, angle_range=angle_range
        )
        for i, (area, angle_range) in enumerate(
            zip(area_list, angle_range_list_ambiguous_left)
        )
    ]
    return PedestrianClassifier(
        A_class_config_list
        + B_class_config_list
        + C_right_class_config_list
        + C_left_class_config_list
    )


def unify_class_by_majority(class_memory: Dict[int, List[str]]) -> Dict[int, List[str]]:
    """
    IDごとに、unknown はそのままにし、A/B/C のみで多数決を取り、
    それ以外（A,B,Cのいずれかだった位置）を多数決結果の1種類に統一する。

    Args:
        class_memory: { id: [class_name1, class_name2, ...], ... }

    Returns:
        unknown の位置は不変、A/B/C だった位置は多数決で得たラベルに統一した辞書
    """
    result = {}
    for _id, names in tqdm(class_memory.items(), desc="Unifying class by majority"):
        if not names:
            result[_id] = ["unknown"]
            continue
        normalized = []  # "unknown" or "A" or "B" or "C"
        for name in names:
            if name == "unknown" or not name:
                normalized.append("unknown")
            else:
                first = name.split("_")[0]
                if first in ("A", "B", "C"):
                    normalized.append(first)
                else:
                    normalized.append("unknown")
        # A,B,C のみで多数決（unknown は投票に含めない）
        abc_only = [x for x in normalized if x != "unknown"]
        majority = Counter(abc_only).most_common(1)[0][0] if abc_only else "unknown"
        # unknown の位置はそのまま、それ以外は多数決結果に統一
        result[_id] = [x if x == "unknown" else majority for x in normalized]
    return result


def main():
    path2homography_matrix = "cfa/data/homography.txt"
    path2back_img = "cfa/data/map.jpg"
    class_data_dir = "cfa/data/class_data"
    track_dir = "cfa/202408_Yokohama_WorldPorters/DSC_7511/track"
    save_dir = "cfa/output"
    os.makedirs(save_dir, exist_ok=True)

    bev_track_generator = BevTrackGenerator(path2homography_matrix, track_dir)
    ped_classifier = set_classifier(class_data_dir)

    start_frame = 1
    end_frame = 1000
    freq = 10
    vis_length = 30

    for i in tqdm(range(start_frame, end_frame, freq), desc="Classifying"):
        s = max(1, i - vis_length)
        e = i
        track = bev_track_generator.run(s, e)

        existing_id = track[track[:, 0] == np.max(track[:, 0])][:, 1]
        for id in existing_id:
            one_track = track[track[:, 1] == id]
            ped_classifier.classify(one_track)

    class_memory = unify_class_by_majority(ped_classifier.class_memory)

    tracklet_displayer = TrackletDisplayer(
        path2back_img=path2back_img,
        class_memory=class_memory,
        scale_ratio=2.0,
    )

    outputs = []
    for i in tqdm(range(start_frame, end_frame, freq), desc="Displaying"):
        s = max(1, i - vis_length)
        e = i
        track = bev_track_generator.run(s, e)
        out_img = tracklet_displayer.display(track)
        # cv2.imwrite(f"{save_dir}/{i:04d}.jpg", out_img)
        outputs.append(out_img)

    create_video(outputs, save_path=f"{save_dir}/output.mp4")


def create_video(outputs: List[np.ndarray], save_path: str):
    height, width, _ = outputs[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(save_path, fourcc, 30, (width, height))
    for output in outputs:
        video_writer.write(output)
    video_writer.release()


if __name__ == "__main__":
    main()
