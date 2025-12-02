import time
import numpy as np
import os
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
from utils.util_config import setup_config
from utils.util_ddp import get_ddp_settings, init_ddp
from utils.util_dnn import (
    fixed_r_seed,
    suggest_network,
)

from prediction.patch_dataset import create_patch_dataloader
from prediction.setting_data import reconstruction


def main_ddp(cfg, input_settings, dist_settings, nprocs):
    mp.spawn(
        main,
        args=(cfg, input_settings, dist_settings),
        nprocs=nprocs,
        join=True,
    )


def main(local_rank, cfg, input_settings=None, dist_settings=None):
    # Fixed random seed
    fixed_r_seed(cfg)

    # init DDP
    global_rank, world_size = init_ddp(local_rank, dist_settings, cfg)

    # Show experimental settings
    if global_rank == 0:
        print(OmegaConf.to_yaml(cfg))

    # set device
    device = torch.device("cuda:%d" % local_rank)
    torch.cuda.set_device(device)

    img_dir, img_size, separate_size, padded_size, trans_data = input_settings
    # test_loader = create_dataloader(
    #     cfg, img_dir,
    #     img_size=img_size,
    #     separate_size=separate_size,
    #     padded_size=padded_size,
    # )
    test_loader = create_patch_dataloader(
        cfg,
        img_dir,
        global_rank,
        world_size,
        img_size=img_size,
        separate_size=separate_size,
        padded_size=padded_size,
    )

    model = suggest_network(cfg, device)
    model = model.to(device, non_blocking=True)

    # cache clear
    torch.cuda.empty_cache()

    # convert model to DDP
    if cfg.default.DDP:
        model = DDP(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    if global_rank == 0:
        print("Inferencing...")
    evaluate(cfg, model, test_loader, device, global_rank, trans_data)

    # end DDP
    if cfg.default.DDP:
        dist.barrier()
        dist.destroy_process_group()


@torch.no_grad()
def evaluate(cfg, model, data_loader, device, global_rank, trans_data):
    model.eval()
    if cfg.default.bar and global_rank == 0:
        bar = tqdm(total=len(data_loader), leave=False)

    threshold = 0.1
    for i, batch in enumerate(data_loader):
        imgs, info_list = batch["image"], batch["info"]
        imgs = imgs.to(device)
        with torch.cuda.amp.autocast(enabled=cfg.default.amp):
            outputs = model(imgs)
        outputs_points = outputs["pred_points"].detach().cpu().numpy()
        outputs_scores = (
            torch.nn.functional.softmax(outputs["pred_logits"], -1)[:, :, 1]
            .detach()
            .cpu()
            .numpy()
        )
        torch.cuda.empty_cache()
        for img, info, points, scores in zip(
            imgs, info_list, outputs_points, outputs_scores
        ):
            np_img = img.detach().cpu().numpy().transpose((1, 2, 0))
            points = points[scores > threshold]
            scores = scores[scores > threshold]
            back_transformed_points = reconstruction(np_img, points, trans_data)
            save_detection(
                cfg.detection_save_dir, back_transformed_points, scores, info
            )
        if cfg.default.bar and global_rank == 0:
            bar.update(1)
    if cfg.default.bar and global_rank == 0:
        bar.close()


def save_detection(save_dir, points, scores, info):
    image_id = info["img_name"]
    (x_num, y_num) = info["num_parts"]
    (x_pos, y_pos) = info["pos"]
    patch_save_dir = f"{save_dir}/{image_id}"
    data_save_dir = f"{patch_save_dir}/data"
    os.makedirs(data_save_dir, exist_ok=True)
    save_name = f"{x_pos}_{y_pos}"
    save_detection_data(data_save_dir, points, scores, save_name)


def save_detection_data(data_save_dir, points, scores, save_name):
    if len(points) != 0:
        np.savetxt(
            data_save_dir + "/" + save_name + ".txt",
            np.concatenate([points, np.expand_dims(scores, -1)], axis=1),
            delimiter=",",
        )
    else:
        np.savetxt(
            data_save_dir + "/" + save_name + ".txt",
            np.array([]),
            delimiter=",",
        )


def input_info(cfg):
    img_dir = cfg.img_dir
    img_size = (4320, 7680)
    # separate_size = (720, 1280)
    padded_size = (4320, 7680)
    padding_data = (padded_size[0] - img_size[0], padded_size[1] - img_size[1])

    # Check if resize is needed
    height, width = separate_size
    new_height = height // 128 * 128
    new_width = width // 128 * 128
    trans_data = (
        None
        if (height == new_height and width == new_width)
        else list((height, width, new_height, new_width))
    )

    input_info_dict = {
        "img_dir": img_dir,
        "img_size": list(img_size),
        "padded_size": list(padded_size),
        "separate_size": list(separate_size),
        "padding_data": list(padding_data),
        "trans_data": trans_data,
    }

    with open(cfg.out_dir + "/input_info.yaml", "w") as f:
        yaml.dump(input_info_dict, f, default_flow_style=False)
    return img_dir, img_size, separate_size, padded_size, trans_data


if __name__ == "__main__":
    start_time = time.time()
    cfg = setup_config()

    save_dir = f"{cfg.out_dir}"
    os.makedirs(save_dir, exist_ok=True)
    cfg.detection_save_dir = save_dir

    # DDP setting
    master, rank_offset, world_size, local_size = get_ddp_settings(cfg)
    dist_settings = [rank_offset, world_size, master]

    # INPUT setting
    img_dir, img_size, separate_size, padded_size, trans_data = input_info(cfg)
    print("IMG_DIR: ", img_dir)
    input_settings = [img_dir, img_size, separate_size, padded_size, trans_data]

    main_ddp(cfg, input_settings, dist_settings, nprocs=local_size)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    with open(f"{cfg.out_dir}/time.txt", "w") as f:
        f.write(f"Time taken: {end_time - start_time} seconds")
