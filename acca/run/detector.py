import torch
from nvidia.dali import fn, types
from nvidia.dali.pipeline import pipeline_def
import numpy as np
from tqdm import tqdm
from p2pnet.utils.util_dnn import suggest_network
from omegaconf import OmegaConf
import torchvision.transforms as transforms
import os


class Detector:
    def __init__(self, cfg_path, weight_path, gpu_id, image_size, dtype):
        self.cfg_path = cfg_path
        self.weight_path = weight_path
        self.gpu_id = gpu_id
        self.image_size = image_size
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.detector = self.set_model_on_gpu(cfg_path, weight_path, gpu_id=gpu_id)
        self.transformer = self.set_transformer()
        self.disable_tqdm = False

    def run(self, video_path, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.run_inference(
            video_path,
            save_dir,
            self.detector,
            self.transformer,
            self.gpu_id,
            self.image_size,
            self.dtype,
        )

    def setup_config(self, cfg_path, weight_path):
        cfg = OmegaConf.load(cfg_path)
        cfg.default.finetune = True
        cfg.network.init_weight = weight_path

        return cfg

    def set_model_on_gpu(self, cfg_path, weight_path, gpu_id=0):
        device = f"cuda:{gpu_id}"
        cfg = self.setup_config(cfg_path, weight_path)
        model = suggest_network(cfg, device)
        model.to(device)
        model.eval()

        return model

    def set_transformer(self):
        transformer = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return transformer

    def invert_transform_points(self, raw_size, resize_size, points):
        ratio_height = raw_size[0] / resize_size[0]
        ratio_width = raw_size[1] / resize_size[1]
        points[:, 0] = points[:, 0] * ratio_width
        points[:, 1] = points[:, 1] * ratio_height
        return points

    def transform_frame(self, frame, transformer, dtype):
        frame = frame.div(255).permute(2, 0, 1).to(dtype)
        transformed = transformer(frame)
        return transformed

    @pipeline_def
    def video_pipe(self, fnames, sequence_length, resize_x, resize_y):
        seq = fn.readers.video_resize(
            device="gpu",
            filenames=fnames,
            sequence_length=sequence_length,
            random_shuffle=False,
            image_type=types.RGB,
            file_list_include_preceding_frame=False,
            resize_x=resize_x,
            resize_y=resize_y,
        )

        return seq

    def run_inference(
        self,
        video_path,
        save_dir,
        detector,
        transformer,
        gpu_id,
        image_size,
        dtype,
    ):

        resize_size = (image_size[0] // 128 * 128, image_size[1] // 128 * 128)
        resize_x = resize_size[1]
        resize_y = resize_size[0]

        pipeline = self.video_pipe(
            batch_size=1,
            num_threads=16,
            device_id=gpu_id,
            fnames=[video_path],
            sequence_length=1,
            resize_x=resize_x,
            resize_y=resize_y,
        )
        pipeline.build()
        n_samples = pipeline.epoch_size()["__VideoResize_0"]
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            for i in tqdm(
                range(n_samples), desc="Processing frames", disable=self.disable_tqdm
            ):
                frame_id = i + 1
                (sequence,) = pipeline.run()

                sequence = torch.utils.dlpack.from_dlpack(
                    sequence.as_tensor().__dlpack__()
                )
                frame = sequence[0, 0]
                transformed_frame = self.transform_frame(frame, transformer, dtype)
                outputs = detector(transformed_frame.unsqueeze(0))
                outputs_scores = (
                    torch.nn.functional.softmax(outputs["pred_logits"], -1)[:, :, 1]
                    .detach()
                    .cpu()
                )
                outputs_points = outputs["pred_points"].detach().cpu().numpy()
                save_path = save_dir + f"/{frame_id:04d}.txt"
                self.post_process(
                    outputs_scores,
                    outputs_points,
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
        inverted_detect_points = self.invert_transform_points(
            image_size, resize_size, points
        )
        self.save_detection_data(save_path, inverted_detect_points, scores)

    def save_detection_data(self, save_path, points, scores):
        if len(points) != 0:
            np.savetxt(
                save_path,
                np.concatenate([points, np.expand_dims(scores, -1)], axis=1),
            )
        else:
            np.savetxt(save_path, np.array([]))
