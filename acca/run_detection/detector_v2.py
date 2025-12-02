import torch
from nvidia.dali import fn, types
from nvidia.dali.pipeline import pipeline_def
import numpy as np
from tqdm import tqdm
from p2pnet.utils.util_dnn import suggest_network
from omegaconf import OmegaConf
import os


class Detector:
    def __init__(self, cfg_path, weight_path, gpu_id, image_size, batch_size, dtype):
        self.cfg_path = cfg_path
        self.weight_path = weight_path
        self.gpu_id = gpu_id
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.detector = self.set_model_on_gpu(cfg_path, weight_path, gpu_id=gpu_id)
        self.raw_size = image_size
        self.resize_size = (image_size[0] // 128 * 128, image_size[1] // 128 * 128)
        self.sequence_length = batch_size
        if gpu_id == 0:
            self.disable_tqdm = False
        else:
            self.disable_tqdm = True

    def run(self, video_path, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.run_inference(
            video_path,
            save_dir,
            self.detector,
            self.gpu_id,
            self.raw_size,
            self.resize_size,
            self.sequence_length,
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

    @pipeline_def
    def video_pipe(fnames, sequence_length, resize_size):
        seq = fn.readers.video(
            device="gpu",
            filenames=fnames,
            sequence_length=sequence_length,
            pad_sequences=True,
            image_type=types.RGB,
        )

        resized_seq = fn.resize(
            seq,
            device="gpu",
            resize_x=resize_size[1],
            resize_y=resize_size[0],
            interp_type=types.INTERP_LINEAR,
        )

        normalized_seq = fn.crop_mirror_normalize(
            resized_seq,
            device="gpu",
            output_layout="FCHW",
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )

        return normalized_seq

    def run_inference(
        self,
        video_path,
        save_dir,
        detector,
        gpu_id,
        image_size,
        resize_size,
        sequence_length,
        dtype,
    ):

        pipeline = self.video_pipe(
            batch_size=1,
            num_threads=8,
            device_id=gpu_id,
            filenames=[video_path],
            sequence_length=sequence_length,
            resize_size=resize_size,
        )
        pipeline.build()
        n_samples = pipeline.epoch_size()["'__Video_0'"]
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            for i in tqdm(
                range(n_samples),
                desc="Processing frames",
                leave=False,
                disable=self.disable_tqdm,
            ):
                frame_ids = [i * sequence_length + j for j in range(sequence_length)]
                (sequence,) = pipeline.run()

                sequence = torch.utils.dlpack.from_dlpack(
                    sequence.as_tensor().__dlpack__()
                )
                frames = sequence[0]
                outputs = detector(frames)
                outputs_scores = (
                    torch.nn.functional.softmax(outputs["pred_logits"], -1)[:, :, 1]
                    .detach()
                    .cpu()
                )
                outputs_points = outputs["pred_points"].detach().cpu().numpy()
                for frame_id, scores, points in zip(
                    frame_ids, outputs_scores, outputs_points, strict=True
                ):
                    save_path = save_dir + f"/{frame_id:04d}.txt"
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
        inverted_detect_points = self.invert_transform_points(
            image_size, resize_size, points
        )
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
                save_path,
                np.concatenate([points, np.expand_dims(scores, -1)], axis=1),
            )
        else:
            np.savetxt(save_path, np.array([]))
