import cv2
import os
import torch
from utils.util_dnn import suggest_network
import click
from omegaconf import OmegaConf
from dataset.dataset_utils import copy_datasets, suggest_dataset_root_dir
from tqdm import tqdm
import glob
from prediction.predict_utils import predict_points, predict_slide


def predict_main(save_dir, img_list, model, device):
    for path2img in tqdm(img_list):
        img_name = path2img.split("/")[-1][:-4]
        raw_img = cv2.imread(path2img)
        img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        if img.shape == (4320, 7680, 3) or img.shape == (2160, 3840, 3):
            # print("slide")
            h_size = 720
            w_size = 1280
            detections = predict_slide(img, model, device, h_size=h_size, w_size=w_size)
        else:
            detections, scores = predict_points(img, model, device)

        save_plot(save_dir, raw_img, img_name, detections)


def save_plot(save_dir, img, img_name, detections):
    img_copy = img.copy()
    p_size = 10
    # p_size = 3
    for p in detections:
        cv2.circle(
            img_copy,
            (int(p[0]), int(p[1])),
            p_size,
            (0, 0, 255),
            -1,
            lineType=cv2.LINE_AA,
        )
    cv2.imwrite(
        save_dir + f"/{img_name}_pred{len(detections)}.png",
        img_copy,
    )


@click.command()
@click.argument("save_dir", type=str, default="vis/plot")
@click.argument(
    "weight_path",
    type=str,
    default="weight_hanabi/SHTechB+2023.pth",
)
@click.argument("weight_name", type=str, default="hanabi")
@click.argument(
    "dataset_name",
    type=str,
    default="WorldPorters2023",
)
@click.argument("resource", type=str, default="local")
@click.argument("gpu_id", type=str, default=0)
def main(save_dir, weight_path, weight_name, dataset_name, resource, gpu_id):
    if torch.cuda.is_available():
        device = f"cuda:{gpu_id}"

    save_dir = f"{save_dir}/{dataset_name}/{weight_name}"
    os.makedirs(save_dir, exist_ok=True)

    cfg_path = "p2pnet/conf/p2p.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.dataset.name = dataset_name
    cfg.default.resource = resource
    # cfg.network.dnn_task = "counting_MIRU2024"
    cfg.default.finetune = True
    cfg.network.init_weight = weight_path

    model = suggest_network(cfg, device)
    model.to(device)
    model.eval()

    # 2023
    datasets_2023 = ["Kokusaibashi2023", "WorldPorters2023"]
    # 2024
    datasets_2024 = [
        "WorldPorters2024_8K",
        "Kokusaibashi2024_4K-2",
        "WorldPorters2024_4K-2",
        "Chosha2024_4K-2",
        "Chosha2024_8K",
        "Chosha2024_4K-1",
        "Kishamichi2024_8K",
        "WorldPorters2024_4K-1",
        "Chosha2024_4K-3",
        "Kokusaibashi2024_4K-1",
        "Chuohiroba2024_4K",
    ]

    if dataset_name in datasets_2023:
        copy_datasets(cfg)
        root_dir = suggest_dataset_root_dir(cfg)
        img_list = sorted(glob.glob(root_dir + "*.jpg"))
    elif dataset_name in datasets_2024:
        copy_datasets_from_mac(cfg)
        root_dir = suggest_dataset_root_dir(cfg)
        img_list = sorted(glob.glob(root_dir + "*.jpg"))
    elif resource == "local":
        root_dir = suggest_dataset_root_dir(cfg)
        img_list = sorted(glob.glob(root_dir + "*.jpg"))
    # else:
    #     raise ValueError("add more code")

    predict_main(save_dir, img_list, model, device)


def main_custom(par_dir, weight_path, weight_name, img_list):
    if torch.cuda.is_available():
        device = "cuda"

    save_dir = f"{par_dir}/{weight_name}"
    os.makedirs(save_dir, exist_ok=True)

    cfg_path = "p2pnet/conf/p2p.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.default.finetune = True
    cfg.network.init_weight = weight_path

    model = suggest_network(cfg, device)
    model.to(device)
    model.eval()

    predict_main(save_dir, img_list, model, device)


if __name__ == "__main__":
    main()
    # par_dir = "demo"
    # weight_path = "outputs_hanabi2023/color_jitter/weights/check_points_epochs_0500.pth"
    # weight_path = "/homes/hnakayama/Fin_p2p/outputs_hanabi2023/500epochs/weights/check_points_epochs_0500.pth"
    # weight_name = "color_jitter"
    # weight_name = "normal"
    # img_list = [
    #     "DSC_7130_2024-08-05_19:13:14_1.jpg",
    #     "DSC_7136_2024-08-05_19:45:33_3.jpg",
    # ]
    # main_custom(par_dir, weight_path, weight_name, img_list)
