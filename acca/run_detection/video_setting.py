from pathlib import Path
from pydantic import BaseModel

VIDEO_INPUT_PATH_DATA_FILE_NAME = "input_video_path_{gpu_id}.json"


class InputVideoPath(BaseModel):
    gpu_id: int
    video_path_list: list[Path] = []


class VideoSetting:

    def __init__(self, video_root_dir: Path, save_dir: Path, num_gpu: int):
        self.video_root_dir = video_root_dir
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.video_path_list = sorted(self.video_root_dir.glob("*"))
        self.num_gpu = num_gpu

    def get_input_video_path(self, gpu_id: int, video_path_list: list[Path]):
        return InputVideoPath(
            gpu_id=gpu_id,
            video_path_list=video_path_list,
        )

    def set(self):
        sep_num = len(self.video_path_list) // self.num_gpu
        for gpu_id in range(self.num_gpu):
            start_idx = gpu_id * sep_num
            end_idx = start_idx + sep_num
            video_path_list = self.video_path_list[start_idx:end_idx]
            input_video_path_data = self.get_input_video_path(gpu_id, video_path_list)
            save_path = self.save_dir.joinpath(
                VIDEO_INPUT_PATH_DATA_FILE_NAME.format(gpu_id=gpu_id)
            )
            save_path.write_text(
                input_video_path_data.model_dump_json(indent=4), encoding="utf-8"
            )
