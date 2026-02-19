from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from p2pnet.utils.util_dnn import suggest_network


class RealtimeDetector:
    def __init__(
        self,
        cfg_path,
        weight_path,
        gpu_id,
        dtype,
        threshold,
    ):
        assert Path(cfg_path).exists(), (
            f"Config file does not exist: {cfg_path} Current working directory: {Path.cwd()}"
        )
        assert Path(weight_path).exists(), (
            f"Weight file does not exist: {weight_path} Current working directory: {Path.cwd()}"
        )
        self.cfg_path = cfg_path
        self.weight_path = weight_path
        self.device = f"cuda:{gpu_id}"
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.detector = self.set_model_on_gpu()
        self.threshold = threshold

    def __call__(self, images: list[np.ndarray]):
        results = self.run_inference(
            images,
            self.detector,
        )
        return results

    def setup_config(self, cfg_path, weight_path):
        cfg = OmegaConf.load(cfg_path)
        cfg.default.finetune = True
        cfg.network.init_weight = weight_path

        return cfg

    def set_model_on_gpu(self):
        cfg = self.setup_config(self.cfg_path, self.weight_path)
        model = suggest_network(cfg, self.device)
        model.to(self.device)
        model.eval()

        return model

    def transform(self, img, resize_size):
        new_height, new_width = resize_size
        transform = A.Compose(
            [
                A.Resize(new_height, new_width, interpolation=cv2.INTER_AREA),
                A.Normalize(p=1.0),
            ],
            keypoint_params=A.KeypointParams(format="xy"),
        )
        transformed = transform(image=img, keypoints=[])
        transformed_img = transformed["image"]

        return transformed_img

    def set_image_batch(self, images: list[np.ndarray], resize_size):
        transformed_images = [self.transform(image, resize_size) for image in images]
        batch = np.array(transformed_images).transpose((0, 3, 1, 2))
        batch = torch.from_numpy(batch).to(self.dtype).clone().to(self.device)
        return batch

    def run_inference(self, images: list[np.ndarray], detector: torch.nn.Module):
        w, h = images[0].shape[:2]
        image_size = (int(w), int(h))
        resize_size = (int(w // 128 * 128), int(h // 128 * 128))
        batch = self.set_image_batch(images, resize_size)
        with torch.no_grad(), torch.autocast("cuda", dtype=self.dtype):
            outputs = detector(batch)
            outputs_scores = (
                torch.nn.functional.softmax(outputs["pred_logits"], -1)[:, :, 1]
                .detach()
                .cpu()
            )
            outputs_points = outputs["pred_points"].detach().cpu().numpy()
            results = []
            for scores, points in zip(outputs_scores, outputs_points, strict=True):
                res = self.post_process(
                    scores,
                    points,
                    resize_size,
                    image_size,
                )
                results.append(res)
            return results

    def post_process(
        self,
        scores,
        points,
        resize_size,
        image_size,
    ):
        points = points[scores > self.threshold]
        scores = scores[scores > self.threshold]
        inverted_detect_points = self.invert_transform_points(
            image_size, resize_size, points
        )
        result = np.concatenate(
            [inverted_detect_points, np.expand_dims(scores, -1)], axis=1
        )
        return result

    def invert_transform_points(self, raw_size, resize_size, points):
        ratio_height = raw_size[0] / resize_size[0]
        ratio_width = raw_size[1] / resize_size[1]
        points[:, 0] = points[:, 0] * ratio_width
        points[:, 1] = points[:, 1] * ratio_height
        return points
