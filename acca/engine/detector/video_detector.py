from pathlib import Path

from omegaconf import OmegaConf

import numpy as np

import torch

from nvidia.dali import fn, types
from nvidia.dali.pipeline import pipeline_def
from p2pnet.utils.util_dnn import suggest_network
from tqdm import tqdm


class VideoDetector:
    def __init__(
        self,
        cfg_path,
        weight_path,
        gpu_id,
        image_size,
        batch_size,
        dtype,
        bar=True,
    ):
        assert Path(cfg_path).exists(), f"Config file does not exist: {cfg_path} Current working directory: {Path.cwd()}"
        assert Path(weight_path).exists(), f"Weight file does not exist: {weight_path} Current working directory: {Path.cwd()}"
        self.cfg_path = cfg_path
        self.weight_path = weight_path
        self.gpu_id = gpu_id
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.detector = self.set_model_on_gpu(cfg_path, weight_path, gpu_id=gpu_id)
        self.raw_size = image_size
        self.resize_size = (image_size[0] // 128 * 128, image_size[1] // 128 * 128)
        self.sequence_length = batch_size
        self.disable_tqdm = not bar

    def run(self, video_paths: list[Path], root_save_dir: Path):
        self.run_inference(
            video_paths,
            root_save_dir,
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
    def video_pipe(self, filenames, sequence_length, resize_size):
        seq = fn.readers.video(
            device="gpu",
            filenames=filenames,
            sequence_length=sequence_length,
            pad_sequences=True,
            image_type=types.RGB,
            file_list_include_preceding_frame=True,
            # ``file_list_include_preceding_frame`` uses the default value False. In future releases, the default value will be changed to True.という警告がうるさいので設定
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

    def get_save_dir(self, video_path: Path, root_save_dir: Path):
        video_name = video_path.stem
        save_dir = root_save_dir.joinpath(video_name, "detection")
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def run_inference(
        self,
        video_paths: list[Path],
        root_save_dir: Path,
        detector: torch.nn.Module,
        gpu_id: int,
        image_size: tuple[int, int],
        resize_size: tuple[int, int],
        sequence_length: int,
        dtype: torch.dtype,
    ):
        for video_path in tqdm(video_paths, desc="Processing videos", disable=self.disable_tqdm):
            save_dir = self.get_save_dir(video_path, root_save_dir)
            pipeline = self.video_pipe(
                batch_size=1,
                num_threads=12,
                device_id=gpu_id,
                filenames=[video_path.as_posix()],
                sequence_length=sequence_length,
                resize_size=resize_size,
            )
            pipeline.build()
            n_samples = pipeline.epoch_size()["__Video_0"]
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                for i in tqdm(
                    range(n_samples),
                    desc="Processing frames",
                    leave=False,
                    disable=self.disable_tqdm,
                ):
                    frame_ids = [i * sequence_length + 1 + j for j in range(sequence_length)]
                    (sequence,) = pipeline.run()

                    sequence = torch.utils.dlpack.from_dlpack(sequence.as_tensor().__dlpack__())
                    frames = sequence[0]
                    outputs = detector(frames)
                    outputs_scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[:, :, 1].detach().cpu()
                    outputs_points = outputs["pred_points"].detach().cpu().numpy()
                    for frame_id, scores, points in zip(frame_ids, outputs_scores, outputs_points, strict=True):
                        save_path = save_dir.joinpath(f"{frame_id:04d}.txt")
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


if __name__ == "__main__":
    detector = VideoDetector(
        cfg_path="p2pnet/conf/p2p.yaml",
        weight_path="cutout.pth",
        gpu_id=0,
        image_size=(4320, 7680),
        batch_size=4,
        dtype="float16",
    )
    video_paths = [Path("demo_videos/demo_1s_1.mov")]
    root_save_dir = Path("outputs_1113/debug")
    detector.run(video_paths, root_save_dir)
