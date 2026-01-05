from pathlib import Path

import matplotlib.pyplot as plt

import torch

import cv2
from engine.detector.image_detector import ImageDetector
from sklearn.decomposition import PCA


def main():
    detector = ImageDetector(
        cfg_path="p2pnet/conf/p2p.yaml",
        weight_path="cutout.pth",
        gpu_id=0,
        image_size=(4320, 7680),
        batch_size=8,
        dtype="float16",
        img_extension="jpg",
    )
    model = detector.detector

    # img_list = [Path("/home/aag16599bn/research/dinov3/hanabi.jpg")]
    # img_list = [Path("/home/aag16599bn/research/dinov3/hanabi_night.jpg")]
    img_list = [Path("/home/aag16599bn/research/dinov3/hanabi_sea.jpg")]

    # 可視化用
    img_path = img_list[0].as_posix()
    name = img_path.split("/")[-1].split(".")[0]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (detector.resize_size[1], detector.resize_size[0]))

    # 特徴量取得用
    samples = detector.set_image_batch(img_list, detector.resize_size)

    with torch.inference_mode(), torch.autocast("cuda", dtype=detector.dtype):
        features = model.backbone(samples)
    print(len(features))
    for i, feature in enumerate(features):
        print(f"Feature {i} shape: {feature.shape}")
        pca_visualize(feature, img, name + f"_{i}")


def pca_visualize(feature_map, original_image, name):
    """
    特徴量マップをPCAで可視化して、特徴量が正しく取得できていることを確認

    Args:
        feature_map: torch.Tensor, shape (B, C, H, W) の特徴量マップ
        original_image: PIL.Image, 元の画像（オプション）
    """
    print("\n=== PCA可視化による特徴量検証 ===")

    # 特徴量マップの形状を確認
    if len(feature_map.shape) != 4:
        raise ValueError(f"Expected feature_map shape (B, C, H, W), got {feature_map.shape}")

    B, C, H, W = feature_map.shape
    print(f"特徴量マップ形状: (B={B}, C={C}, H={H}, W={W})")

    # バッチサイズが1の場合のみ処理
    if B != 1:
        print("警告: バッチサイズが1ではありません。最初のサンプルのみ処理します。")
        feature_map = feature_map[0:1]

    # 特徴量マップを (H*W, C) の形状に変換（ノートブックの実装を参考）
    # feature_map: (1, C, H, W) -> (C, H*W) -> (H*W, C)
    x = feature_map.squeeze(0)  # (C, H, W)
    dim = x.shape[0]  # C
    x = x.view(dim, -1).permute(1, 0)  # (H*W, C)

    print(f"変換後の特徴量形状: {x.shape} (パッチ数={x.shape[0]}, 特徴次元={x.shape[1]})")

    # 特徴量の統計情報を表示
    print("\n特徴量の統計情報:")
    print(f"  Mean: {x.mean().item():.6f}")
    print(f"  Std: {x.std().item():.6f}")
    print(f"  Min: {x.min().item():.6f}")
    print(f"  Max: {x.max().item():.6f}")

    # NaNやInfがないか確認
    has_nan = torch.isnan(x).any().item()
    has_inf = torch.isinf(x).any().item()
    print(f"  NaN: {has_nan}")
    print(f"  Inf: {has_inf}")

    if has_nan or has_inf:
        print("警告: 特徴量にNaNまたはInfが含まれています！")
        return

    # CPUに移動してnumpyに変換
    x_np = x.detach().cpu().numpy()

    # PCAを適用（3成分、whiteningあり）
    print("\nPCAを適用中...")
    pca = PCA(n_components=3, whiten=True)
    pca.fit(x_np)

    # 説明分散比を表示
    explained_variance = pca.explained_variance_ratio_
    print("PCA説明分散比:")
    for i, var in enumerate(explained_variance):
        print(f"  成分{i + 1}: {var:.4f} ({var * 100:.2f}%)")
    print(f"  累積: {explained_variance.sum():.4f} ({explained_variance.sum() * 100:.2f}%)")

    # PCAを適用して可視化用の画像を生成
    projected = pca.transform(x_np)  # (H*W, 3)
    projected_tensor = torch.from_numpy(projected).view(H, W, 3)  # (H, W, 3)

    # 色を鮮やかにするために2倍してシグモイドを適用（ノートブックの実装を参考）
    projected_tensor = torch.nn.functional.sigmoid(projected_tensor.mul(2.0)).permute(2, 0, 1)  # (3, H, W)

    # 可視化
    print("\nPCA可視化画像を生成中...")
    num_subplots = 3 if original_image is not None else 2
    plt.figure(figsize=(5 * num_subplots, 5), dpi=150)

    subplot_idx = 1

    # 元画像を表示
    if original_image is not None:
        plt.subplot(1, num_subplots, subplot_idx)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis("off")
        subplot_idx += 1

    # 元の特徴量マップの平均（チャネル方向）を可視化
    plt.subplot(1, num_subplots, subplot_idx)
    feature_mean = feature_map.squeeze(0).mean(dim=0).detach().cpu().numpy()  # (H, W)
    plt.imshow(feature_mean, cmap="viridis")
    plt.title(f"Feature Map Mean\n(Shape: {feature_map.shape})")
    plt.axis("off")
    subplot_idx += 1

    # PCA可視化
    plt.subplot(1, num_subplots, subplot_idx)
    projected_vis = projected_tensor.permute(1, 2, 0).numpy()  # (H, W, 3)
    plt.imshow(projected_vis)
    plt.title(f"PCA Visualization (3 components)\nExplained Variance: {explained_variance.sum() * 100:.1f}%")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"feature_map_{name}.png", dpi=150, bbox_inches="tight")
    print(f"可視化結果を 'feature_map_{name}.png' に保存しました。")
    plt.close()


if __name__ == "__main__":
    main()
