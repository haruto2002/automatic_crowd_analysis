from pathlib import Path

import hydra
from engine.common.utils import HanabiImageData, load_previous_state_file
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    config_path="conf/image_conf",
    config_name="conf",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    print("Config:\n" + OmegaConf.to_yaml(cfg))

    pipleline = [instantiate(cfg[phase]) for phase in cfg.pipeline]

    image_dir = Path(cfg.image_dir)
    root_save_dir = Path(cfg.output_dir)

    print(f"Processing {image_dir}")
    hanabi_image_data = HanabiImageData(
        img_dir=Path(image_dir),
        root_output_dir=Path(root_save_dir),
        img_extension=cfg.image_extension,
    )
    existing_state_file = hanabi_image_data.set_data()
    print(hanabi_image_data.state_file)
    if existing_state_file and cfg.resume:
        print(f"Loading previous state file: {existing_state_file}")
        hanabi_image_data = load_previous_state_file(existing_state_file)

    for phase in pipleline:
        print(f"Running {phase.__class__.__name__}")
        phase(hanabi_image_data)
        hanabi_image_data.save()


if __name__ == "__main__":
    main()
