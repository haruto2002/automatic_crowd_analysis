from pathlib import Path

from omegaconf import OmegaConf

import numpy as np

import torch

import albumentations as A
import cv2
from engine.common.utils import HanabiImageData
from p2pnet.utils.util_dnn import suggest_network
from tqdm import tqdm


class ImageDetector:
    def __init__(
        self,
        cfg_path,
        weight_path,
        gpu_id,
        image_size,
        batch_size,
        dtype,
        img_extension,
        bar=True,
    ):
        self.cfg_path = cfg_path
        self.weight_path = weight_path
        self.device = f"cuda:{gpu_id}"
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.detector = self.set_model_on_gpu()
        self.raw_size = image_size
        self.resize_size = (image_size[0] // 128 * 128, image_size[1] // 128 * 128)
        self.batch_size = batch_size
        self.img_extension = img_extension
        self.disable_tqdm = not bar

    def __call__(self, hanabi_image_data: HanabiImageData):
        self.run_inference(
            hanabi_image_data.image_paths,
            hanabi_image_data.get_output_dir_for_phase("detection"),
            self.detector,
            self.raw_size,
            self.resize_size,
        )

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

    def set_image_batch(self, image_paths: list[Path], resize_size):
        images = [cv2.imread(image_path.as_posix()) for image_path in image_paths]
        transformed_images = [self.transform(image, resize_size) for image in images]
        batch = np.array(transformed_images).transpose((0, 3, 1, 2))
        batch = torch.from_numpy(batch).to(self.dtype).clone().to(self.device)
        return batch

    def run_inference(
        self,
        image_paths: list[Path],
        save_dir: Path,
        detector: torch.nn.Module,
        image_size: tuple[int, int],
        resize_size: tuple[int, int],
    ):
        batch_num = (len(image_paths) + self.batch_size - 1) // self.batch_size
        for i in tqdm(range(batch_num), desc="Processing batches", disable=self.disable_tqdm):
            batch_image_paths = image_paths[i * self.batch_size : (i + 1) * self.batch_size]
            image_names = [image_path.stem for image_path in batch_image_paths]
            batch = self.set_image_batch(batch_image_paths, resize_size)
            with torch.inference_mode(), torch.autocast("cuda", dtype=self.dtype):
                outputs = detector(batch)
                outputs_scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[:, :, 1].detach().cpu()
                outputs_points = outputs["pred_points"].detach().cpu().numpy()
                for img_name, scores, points in zip(image_names, outputs_scores, outputs_points, strict=True):
                    save_path = save_dir.joinpath(f"{img_name}.txt")
                    self.post_process(
                        scores,
                        points,
                        save_path,
                        resize_size,
                        image_size,
                    )

    def post_process(
        self,
        scores,
        points,
        save_path,
        resize_size,
        image_size,
    ):
        threshold = 0.1
        points = points[scores > threshold]
        scores = scores[scores > threshold]
        inverted_detect_points = self.invert_transform_points(image_size, resize_size, points)
        self.save_detection_data(save_path, inverted_detect_points, scores)

    def invert_transform_points(self, raw_size, resize_size, points):
        ratio_height = raw_size[0] / resize_size[0]
        ratio_width = raw_size[1] / resize_size[1]
        points[:, 0] = points[:, 0] * ratio_width
        points[:, 1] = points[:, 1] * ratio_height
        return points

    def save_detection_data(self, save_path, points, scores):
        if len(points) != 0:
            np.savetxt(
                save_path.as_posix(),
                np.concatenate([points, np.expand_dims(scores, -1)], axis=1),
            )
        else:
            np.savetxt(save_path.as_posix(), np.array([]))
