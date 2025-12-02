import time
import numpy as np
import os
import glob
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from utils.util_config import setup_config
from utils.util_ddp import get_ddp_settings, init_ddp
from utils.util_dnn import (
    fixed_r_seed,
    suggest_network,
)

from prediction.fullsize_ddp_dataset import create_fullsize_dataloader
from prediction.setting_data import reconstruction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main_ddp(cfg, dist_settings, nprocs):
    mp.spawn(
        main,
        args=(cfg, dist_settings),
        nprocs=nprocs,
        join=True,
    )


def main(local_rank, cfg, dist_settings=None):
    # Fixed random seed
    fixed_r_seed(cfg)

    # init DDP
    global_rank, world_size = init_ddp(local_rank, dist_settings, cfg)

    # if global_rank == 0:
    #     logger.info(f"=== 推論処理開始 (Rank: {global_rank}/{world_size}) ===")
    #     logger.info(f"設定内容:")
    #     logger.info(OmegaConf.to_yaml(cfg))

    # set device
    device = torch.device("cuda:%d" % local_rank)
    torch.cuda.set_device(device)

    test_loader = create_fullsize_dataloader(
        cfg,
        cfg.img_dir,
        global_rank,
        world_size,
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

    evaluate(cfg, model, test_loader, device, global_rank)

    # end DDP
    if cfg.default.DDP:
        dist.barrier()
        dist.destroy_process_group()


@torch.no_grad()
def evaluate(cfg, model, data_loader, device, global_rank):
    model.eval()

    threshold = 0.1

    if cfg.default.bar and global_rank == 0:
        bar = tqdm(
            total=len(data_loader),
            leave=False,
            desc="Inferencing",
            disable=not cfg.default.bar,
        )

    for i, batch in enumerate(data_loader):
        (
            imgs,
            img_names,
            original_heights,
            original_widths,
            transformed_heights,
            transformed_widths,
        ) = (
            batch["image"],
            batch["img_name"],
            batch["original_height"],
            batch["original_width"],
            batch["transformed_height"],
            batch["transformed_width"],
        )
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

        for (
            img,
            img_name,
            points,
            scores,
            original_height,
            original_width,
            transformed_height,
            transformed_width,
        ) in zip(
            imgs,
            img_names,
            outputs_points,
            outputs_scores,
            original_heights,
            original_widths,
            transformed_heights,
            transformed_widths,
        ):
            trans_data = (
                original_height,
                original_width,
                transformed_height,
                transformed_width,
            )
            np_img = img.detach().cpu().numpy().transpose((1, 2, 0))
            points = points[scores > threshold]
            scores = scores[scores > threshold]
            original_points = reconstruction(np_img, points, trans_data)
            save_detection_data(cfg.full_det_dir, original_points, scores, img_name)

        if cfg.default.bar and global_rank == 0:
            bar.update(1)

    if cfg.default.bar and global_rank == 0:
        bar.close()


def save_detection_data(data_save_dir, points, scores, save_name):
    if len(points) != 0:
        np.savetxt(
            data_save_dir + "/" + save_name + ".txt",
            np.concatenate([points, np.expand_dims(scores, -1)], axis=1),
        )
    else:
        np.savetxt(data_save_dir + "/" + save_name + ".txt", np.array([]))


def run():
    start_time = time.time()

    cfg = setup_config()

    assert (len(sorted(glob.glob(os.path.join(cfg.img_dir, "*.jpg")))) != 0) or len(
        sorted(glob.glob(os.path.join(cfg.img_dir, "*.png")))
    ) != 0, "No jpg images found (img_dir: {cfg.img_dir})"

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, cfg.log_level.upper() or "INFO"))
    visible_tqdm = logger.level < logging.ERROR
    cfg.default.bar = visible_tqdm

    logger.info("=== P2PNet推論開始 ===")

    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.full_det_dir, exist_ok=True)

    # DDP setting
    master, rank_offset, world_size, local_size = get_ddp_settings(cfg)
    dist_settings = [rank_offset, world_size, master]

    logger.info(
        f"DDP設定: master={master}, rank_offset={rank_offset}, world_size={world_size}, local_size={local_size}"
    )

    main_ddp(cfg, dist_settings, nprocs=local_size)

    end_time = time.time()
    total_time = end_time - start_time

    time_log_path = f"{cfg.out_dir}/time.txt"
    with open(time_log_path, "w") as f:
        f.write(f"Time taken: {total_time:.2f} seconds")


if __name__ == "__main__":
    run()
