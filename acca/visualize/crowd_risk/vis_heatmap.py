import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm


def save_colorbar(save_dir, min_val, max_val, height=400, width=50):
    """Function to generate and save colorbar"""
    # Create colorbar
    gradient = np.linspace(min_val, max_val, height)[:, np.newaxis]
    gradient = np.tile(gradient, (1, width))

    # 正規化して0-255の範囲に変換
    gradient_norm = ((gradient - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    gradient_norm = np.tile(gradient_norm[:, :, np.newaxis], (1, 1, 3))

    # Apply colormap
    colorbar = cv2.applyColorMap(gradient_norm, cv2.COLORMAP_JET)

    # 保存
    colorbar_dir = os.path.join(save_dir, "colorbar")
    os.makedirs(colorbar_dir, exist_ok=True)
    cv2.imwrite(os.path.join(colorbar_dir, "colorbar.png"), colorbar)

    # Colorbar with text (using matplotlib)
    plt.figure(figsize=(2, 8))
    plt.imshow(gradient, cmap="jet")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(
        os.path.join(colorbar_dir, "colorbar_with_text.png"),
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close()


def display_heatmap_parallel(
    map_data_list,
    vec_data_list,
    img_list,
    name_list,
    save_dir,
    grid_size,
    scale=1,
    max_score=None,
    min_score=None,
    resize_method="linear",
    display_vec=True,
    colorbar_img=False,
):
    # 全体の最小値と最大値を取得
    map_global_max, map_global_min = get_global_min_max(map_data_list)
    if max_score is None:
        max_score = map_global_max
    if min_score is None:
        min_score = map_global_min

    # Save colorbar
    if colorbar_img:
        save_colorbar(save_dir, map_global_min, map_global_max)

    pool_list = []
    for map_data, vec_data, img, name in zip(
        map_data_list, vec_data_list, img_list, name_list
    ):
        inputs = (
            map_data,
            vec_data,
            img,
            name,
            grid_size,
            max_score,
            min_score,
            save_dir,
            scale,
            display_vec,
            resize_method,
        )
        pool_list.append(inputs)

    pool_size = min(len(pool_list), os.cpu_count())
    with Pool(pool_size) as p:
        results = list(
            tqdm(p.imap_unordered(display_single, pool_list), total=len(pool_list))
        )

    return results


def get_global_min_max(map_data_list, dense=False):
    max_values = [hm.max() for hm in map_data_list]
    min_values = [hm.min() for hm in map_data_list]

    # 降順ソートして上位 100 個を取得
    top = sorted(max_values, reverse=True)[:50]
    bottom = sorted(min_values, reverse=False)[:50]

    # 上位 100 個の平均を計算
    max_mean = np.mean(top)
    min_mean = np.mean(bottom)

    return max_mean, min_mean


def display_single(inputs):
    (
        map_data,
        vec_data,
        img,
        name,
        grid_size,
        max_score,
        min_score,
        save_dir,
        scale,
        display_vec,
        resize_method,
    ) = inputs

    img = cv2.resize(img, None, fx=scale, fy=scale)
    img_height, img_width, _ = img.shape
    vec_data *= scale

    # ヒートマップをリサイズ
    if resize_method == "linear":
        map_data_resized = cv2.resize(
            map_data,
            (img_width, img_height),
            interpolation=cv2.INTER_LINEAR,
        )

    elif resize_method == "nearest":
        map_data_resized = np.zeros((img_height, img_width))
        for i in range(map_data.shape[0]):
            for j in range(map_data.shape[1]):
                map_data_resized[
                    i * grid_size * scale : (i + 1) * grid_size * scale,
                    j * grid_size * scale : (j + 1) * grid_size * scale,
                ] = map_data[i, j]
    else:
        raise ValueError(f"Invalid resize method: {resize_method}")

    min_score = min_score
    max_score = max_score
    # NaN を最小値に置き換え
    map_data_resized[np.isnan(map_data_resized)] = min_score
    # min_score, max_score を用いた正規化
    map_data_resized = np.clip(map_data_resized, min_score, max_score)
    map_norm_map = (
        (map_data_resized - min_score) / (max_score - min_score) * 255
    ).astype(np.uint8)
    map_heatmap = cv2.applyColorMap(map_norm_map, cv2.COLORMAP_JET)

    # 矢印の描画
    if display_vec:
        arrow_len = 5
        tipLength = 0.3
        arrow_scale = 50
        # Heatmap is in front
        for x, y, vx, vy in vec_data:
            # ベクトルを圧縮・スケール
            vec = np.array([vx, vy], dtype=float)
            vec = compress_vec_data(vec) * arrow_scale

            # 開始点・終了点
            start = (int(x), int(y))
            end = (int(x + vec[0]), int(y + vec[1]))

            # 角度を計算（-π～+π）
            angle = np.arctan2(vec[1], vec[0])

            # Hue を 0–179 にマッピング
            hue = ((angle + np.pi) / (2 * np.pi) * 179).astype(int)
            # HSV 画像（1×1）を作って BGR に変換
            hsv_pixel = np.uint8([[[hue, 255, 255]]])  # H, S=255, V=255
            bgr_color = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0].tolist()

            # 矢印を描画
            cv2.arrowedLine(
                img,
                start,
                end,
                color=bgr_color,
                thickness=arrow_len,
                tipLength=tipLength,
            )
    img = img.astype(np.uint8)
    output = cv2.addWeighted(img, 0.5, map_heatmap, 0.5, 0)

    # 保存
    hm_vis_save_dir = save_dir + "/hm_vis_" + str(img_height)
    hm_raw_save_dir = save_dir + "/hm_raw_" + str(img_height)
    os.makedirs(hm_vis_save_dir, exist_ok=True)
    os.makedirs(hm_raw_save_dir, exist_ok=True)
    hm_vis_save_path = os.path.join(hm_vis_save_dir, f"{name}.png")
    hm_raw_save_path = os.path.join(hm_raw_save_dir, f"{name}.png")
    cv2.imwrite(hm_vis_save_path, output)
    cv2.imwrite(hm_raw_save_path, map_heatmap)

    return name, output


def compress_vec_data(vel):
    norm = np.linalg.norm(vel)
    over_ratio = norm / 0.35
    if norm > 0.35:
        vel = vel / norm * 0.35 * (over_ratio * 0.5)
    return vel
