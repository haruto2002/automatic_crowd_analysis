#!/bin/bash
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=3:00:00
#PBS -P gaa50073
#PBS -e /home/aag16599bn/research/automatic_crowd_analysis/acca/logs
#PBS -o /home/aag16599bn/research/automatic_crowd_analysis/acca/logs

WORK_DIR=/home/aag16599bn/research/automatic_crowd_analysis/acca
cd $WORK_DIR

IMAGE_DIR=/home/aag16599bn/research/hanabi_videos/1min_freq_images
SAVE_DIR=/home/aag16599bn/research/automatic_crowd_analysis/acca/outputs/1min_freq_images


pixi run aiaccel-job local --config=abci3.yaml gpu --n_tasks=27 --n_tasks_per_proc=1 ${SAVE_DIR}/logs/image_detection.log \
    -- python run_detection_for_image.py --config-name=detection_with_graph_movie \
    image_dir=${IMAGE_DIR} \
    output_dir=${SAVE_DIR}/results \
    image_extension=jpg \
    resume=False

time_log_file="${SAVE_DIR}/logs/time.log"
echo "${SECONDS} seconds" >> "${time_log_file}"