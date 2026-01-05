import os
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from engine.common.utils import HanabiImageData, load_previous_state_file


@hydra.main(config_path="conf/image_conf", config_name="detection_with_graph_movie", version_base=None)
def main(cfg: DictConfig) -> None:
    os.environ["OMP_NUM_THREADS"] = "12"
    os.environ["MKL_NUM_THREADS"] = "12"

    print("Config:\n" + OmegaConf.to_yaml(cfg))

    pipleline = [instantiate(cfg[phase]) for phase in cfg.pipeline]

    image_dirs = list(sorted(Path(cfg.image_dir).glob("*/*")))

    if "TASK_INDEX" in os.environ:
        start = int(os.environ["TASK_INDEX"]) - 1
        end = start + int(os.environ["TASK_STEPSIZE"])

    else:
        assert cfg.task_index is not None and cfg.task_stepsize is not None, "task_index and task_stepsize must be set"
        start = int(cfg.task_index) - 1
        end = start + int(cfg.task_stepsize)

    image_dirs = image_dirs[start:end]

    root_save_dir = Path(cfg.output_dir)

    print("Running pipeline...")
    for i, image_dir in enumerate(image_dirs):
        print(f"Processing {image_dir} ({i + 1} of {len(image_dirs)})")
        hanabi_image_data = HanabiImageData(
            img_dir=Path(image_dir), root_output_dir=Path(root_save_dir), img_extension=cfg.image_extension
        )
        existing_state_file = hanabi_image_data.set_data()
        print(hanabi_image_data.state_file)
        if existing_state_file and cfg.resume:
            print(f"Loading previous state file: {existing_state_file}")
            hanabi_image_data = load_previous_state_file(existing_state_file)

        for phase in pipleline:
            phase(hanabi_image_data)
            hanabi_image_data.save()


if __name__ == "__main__":
    main()
