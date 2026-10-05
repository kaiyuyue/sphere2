#!/usr/bin/env bash
#
# sample images from pretrained imagenet models
#
# usage:
# ./scripts/sample_imagenet.sh
#
# images are saved to workspace/visualization/<job_dir>

# models to sample from, one folder each under workspace/experiments
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

# imagenet class ids to sample:
# 17 jay, 88 macaw, 279 arctic fox, 387 red panda, 927 trifle, 607 jack-o'-lantern
CLASSES="17 387 88 279 927 607"


# =========================== Sampling ============================

for JOB_DIR in "${JOB_DIRS[@]}"; do

  # sampling angle per model size
  if [[ $JOB_DIR == sphere2-large-* ]]; then
    SAMPLING_ANGLE=84.0
  else
    SAMPLING_ANGLE=85.5
  fi

  ./run.sh sample.py \
    --job_dir $JOB_DIR \
    --out_dir visualization \
    --use_ema_model True \
    --num_gen_samples 64 \
    --batch_size_per_rank 16 \
    --forward_steps 1 2 \
    --sampling_init_angle $SAMPLING_ANGLE \
    --cache_sampling_noise True \
    --seed_sampling False \
    --use_cfg False \
    --cfg_min 0.0 \
    --cfg_max 5.0 \
    --cfg_gap 5.0 \
    --cfg_position angle \
    --grid_nrow 8 \
    --random_sample_classes False \
    --class_of_interests $CLASSES
done

# =================================================================
