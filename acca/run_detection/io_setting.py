from pathlib import Path
import json
from run_detection.video_setting import InputVideoPath
from pydantic import BaseModel


class IOInfo(BaseModel):
    input_video_path: Path
    detection_save_dir: Path


class InputVideoPathLoader:
    def __init__(self, input_video_path_dir: str, gpu_id: int):
        self.input_video_path_dir = Path(input_video_path_dir)
        self.input_video_path_data_file = self.input_video_path_dir.joinpath(
            f"input_video_path_{gpu_id}.json"
        )

    def load_input_video_path(self):
        input_video_path_data = json.loads(self.input_video_path_data_file.read_text())
        input_video_path_data = InputVideoPath.model_validate_json(
            self.input_video_path_data_file.read_text(encoding="utf-8")
        )
        return input_video_path_data.video_path_list


class IOSetting:
    def __init__(self, input_video_path_dir: str, gpu_id: int, save_root_dir: str):
        self.inputs_video_path_list = InputVideoPathLoader(
            input_video_path_dir, gpu_id
        ).load_input_video_path()
        self.save_root_dir = Path(save_root_dir)

    def get_io_info_list(self):
        io_info_list = []
        for input_video_path in self.inputs_video_path_list:
            video_name = input_video_path.stem
            detection_save_dir = Path(self.save_root_dir).joinpath(
                video_name,
                "detection",
            )
            io_info_list.append(
                IOInfo(
                    input_video_path=input_video_path,
                    detection_save_dir=detection_save_dir,
                )
            )
        return io_info_list
