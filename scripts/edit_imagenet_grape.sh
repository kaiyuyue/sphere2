#!/usr/bin/env bash
#
# edit an image outside imagenet by reconstructing it conditioned on another class
#
# usage:
# ./scripts/edit_imagenet_grape.sh
#
# one grid per run, rows are trials with different noise,
# columns are [input | round trip | step 1 | ... | step N]
# images are saved to workspace/editing/<job_dir>

# models to edit with, one folder each under workspace/experiments
# checkpoints: https://huggingface.co/tomg-group-umd/sphere2
JOB_DIRS=(
  # "sphere2-base-imagenet-256px"
  # "sphere2-base-imagenet-512px"
  # "sphere2-large-imagenet-256px"
  "sphere2-large-imagenet-512px"
  # ----- [checkpoints trained with fd-lite loss]
  # "sphere2-base-imagenet-fd-lite-256px"
  # "sphere2-base-imagenet-fd-lite-512px"
  # "sphere2-large-imagenet-fd-lite-256px"
  # "sphere2-large-imagenet-fd-lite-512px"
)

# image to edit
INPUT_IMAGE="assets/grape.png"

# target class
# note: edit.py takes the training class ids, the class_id column of
# workspace/datasets/imagenet/folder_to_id_to_label.json, not the original imagenet ids
CLASS=597


# ========================= Class Editing =========================

for JOB_DIR in "${JOB_DIRS[@]}"; do
  ./run.sh edit.py \
    --job_dir $JOB_DIR \
    --out_dir editing \
    --use_ema_model True \
    --edit_mode reconstruct \
    --input_image $INPUT_IMAGE \
    --rec_class $CLASS \
    --rec_angles 85 \
    --sampling_loop_angle 85 \
    --forward_steps 8 \
    --num_trials 12 \
    --cache_sampling_noise True \
    --seed_mode random
done

# =================================================================
