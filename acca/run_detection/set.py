from run_detection.video_setting import VideoSetting
import argparse
from pathlib import Path

"""
python run_detection/set.py --video_root_dir <video_root_dir> --save_root_dir <save_root_dir> --node_type <node_type> --pbs_walltime <pbs_walltime>

Example:

1) demo
python -m run_detection.set --video_root_dir demo_videos --save_root_dir demo_outputs/demo_videos --node_type rt_HF --pbs_walltime 1:00:00

2) full demo
python -m run_detection.set --video_root_dir /home/aag16599bn/research/hanabi_videos/202408_Yokohama_WorldPorters --save_root_dir demo_outputs/202408_Yokohama_WorldPorters --node_type rt_HF --pbs_walltime 1:00:00
"""

LOG_DIR = "/home/aag16599bn/research/Automatic-Crowd-Congestion-Analysis-System/log/"
CONDA_ENV_PATH = "~/miniconda3/bin/activate hnakayama"
WORKDIR = "~/research/automatic_crowd_analysis/acca"
PBS_PROJECT = "gaa50073"
PBS_SELECT = 1


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root_dir", type=str, required=True)
    parser.add_argument("--save_root_dir", type=str, required=True)
    parser.add_argument("--node_type", type=str, required=True)
    parser.add_argument("--pbs_walltime", type=str, required=True)
    return parser.parse_args()


def get_num_gpu(node_type: str) -> int:
    if node_type == "rt_HF":
        return 8
    elif node_type == "rt_HG":
        return 1
    elif node_type == "rt_HC":
        return 0
    else:
        raise ValueError(f"Invalid node type: {node_type}")


def main() -> None:
    args = get_args()
    video_root_dir = Path(args.video_root_dir)
    save_root_dir = Path(args.save_root_dir)
    node_type = args.node_type
    pbs_walltime = args.pbs_walltime
    num_gpu = get_num_gpu(node_type)
    video_setting = VideoSetting(
        video_root_dir=video_root_dir,
        save_dir=save_root_dir.joinpath("detection_run_info"),
        num_gpu=num_gpu,
    )
    video_setting.set()

    create_run_script(
        input_video_path_dir=video_setting.save_dir,
        save_root_dir=save_root_dir,
        num_gpu=num_gpu,
        pbs_queue=node_type,
        pbs_walltime=pbs_walltime,
    )


def create_run_script(
    input_video_path_dir: Path,
    save_root_dir: Path,
    num_gpu: int,
    pbs_queue: str = "rt_HF",
    pbs_walltime: str = "1:00:00",
) -> None:
    """
    Args:
        input_video_path_dir: 入力ビデオパスディレクトリ（JSONファイルが保存されているディレクトリ）
        save_root_dir: 出力ディレクトリ
        num_gpu: GPU数（0からnum_gpu-1まで並列処理）
        output_path: 生成するrun.shのパス
        pbs_queue: PBSキューの名前
        pbs_walltime: PBSのwalltime
    """

    script_lines = []

    magic_lines = [
        "#!/bin/bash",
        f"#PBS -q {pbs_queue}",
        f"#PBS -l select={PBS_SELECT}",
        f"#PBS -l walltime={pbs_walltime}",
        f"#PBS -P {PBS_PROJECT}",
        f"#PBS -e {LOG_DIR}",
        f"#PBS -o {LOG_DIR}",
        "",
        f"WORKDIR={WORKDIR}",
        "cd $WORKDIR",
        "",
    ]
    script_lines.extend(magic_lines)

    hydra_log_dir = save_root_dir.joinpath("detection_hydra_log")
    results_save_dir = save_root_dir.joinpath("results")
    fix_params = [
        f"HYDRA_LOG_DIR={hydra_log_dir.as_posix()}",
        f"INPUT_VIDEO_PATH_DIR={input_video_path_dir.as_posix()}",
        f"SAVE_ROOT_DIR={results_save_dir.as_posix()}",
        "",
    ]
    script_lines.extend(fix_params)

    # 各GPUごとの並列実行コマンド
    for gpu_id in range(num_gpu):
        run_cmd = [
            "pixi run python -m run_detection.main \\",
            "hydra_log_dir=${HYDRA_LOG_DIR} \\",
            "input_video_path_dir=${INPUT_VIDEO_PATH_DIR} \\",
            "output_dir=${SAVE_ROOT_DIR} \\",
            f"gpu_id={gpu_id} &",
            "",
        ]
        script_lines.extend(run_cmd)

    script_lines.extend(["wait"])
    script_lines.extend([""])
    script_lines.extend(
        [
            'echo "Detection processing time: $SECONDS seconds" >> ${HYDRA_LOG_DIR}/detection_time.txt'
        ]
    )

    # ファイルに書き込み
    script_path = save_root_dir.joinpath("detection_run_info/run.sh")
    script_path.write_text("\n".join(script_lines), encoding="utf-8")

    print(f"Created run script: {script_path.as_posix()}")


if __name__ == "__main__":
    main()
