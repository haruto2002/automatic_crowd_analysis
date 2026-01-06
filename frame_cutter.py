import argparse
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


def get_frame(path2video, save_dir, img_extension):
    logger.info(f"Extracting frames from video file {path2video}...")
    if img_extension == "png":
        command = f"ffmpeg -i {path2video} -vcodec png {save_dir}/%04d.png"
    elif img_extension == "jpg":
        command = f"ffmpeg -i {path2video} -q:v 1 {save_dir}/%04d.jpg"
    else:
        raise ValueError(f"Invalid image extension: {img_extension}")
    logger.debug(f"Executing command: {command}")

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("Frame extraction completed successfully")
        else:
            logger.error(f"Error occurred during frame extraction: {result.stderr}")
            raise RuntimeError(result.stderr)
    except Exception as e:
        logger.error(f"Exception occurred during frame extraction: {e}")
        raise RuntimeError(e)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path2video", type=str, default="videos/DSC_6628.MOV")
    parser.add_argument("--img_extension", type=str, default="jpg")
    parser.add_argument("--save_dir", type=str, default="images")
    return parser.parse_args()


def main():
    args = get_args()

    path2video = args.path2video
    video_name = Path(path2video).stem
    save_dir = Path(args.save_dir).joinpath(video_name)
    img_extension = args.img_extension

    logger.info("=== Frame extraction process started ===")
    logger.info(f"Video file: {path2video}")
    logger.info(f"Save directory: {save_dir}")

    if not os.path.exists(path2video):
        raise FileNotFoundError(path2video)

    save_dir.mkdir(parents=True, exist_ok=True)
    get_frame(path2video, save_dir, img_extension)


if __name__ == "__main__":
    main()
