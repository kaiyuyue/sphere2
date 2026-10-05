#!/usr/bin/env bash
#
# class interpolation with pretrained imagenet models
#
# usage:
# ./scripts/edit_imagenet.sh
#
# one grid per class pair, rows are trials and columns go from class a to class b
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

# class pairs to interpolate, "a b" per entry
# note: edit.py takes the training class ids, the class_id column of
# workspace/datasets/imagenet/folder_to_id_to_label.json, not the original imagenet ids
PAIRS=(
  "867 294"   # jaguar -> lion
  "445 294"   # snow leopard -> lion
  "445 634"   # snow leopard -> tiger
  "849 634"   # persian cat -> tiger
  "348 867"   # egyptian cat -> jaguar
  "348 445"   # egyptian cat -> snow leopard
  "976 528"   # giant panda -> american black bear
  "226 59"    # castle -> mosque
  "561 994"   # jay -> macaw
)


# ====================== Class Interpolation ======================

for JOB_DIR in "${JOB_DIRS[@]}"; do

  # sampling angle per model size
  if [[ $JOB_DIR == sphere2-large-* ]]; then
    EDIT_ANGLE=84.0
  else
    EDIT_ANGLE=85.5
  fi

  for PAIR in "${PAIRS[@]}"; do
    ./run.sh edit.py \
      --job_dir $JOB_DIR \
      --out_dir editing \
      --use_ema_model True \
      --edit_mode interp \
      --interp_classes $PAIR \
      --class_interp slerp \
      --interp_fix_latent True \
      --num_interp 12 \
      --num_trials 4 \
      --forward_steps 2 \
      --edit_angle $EDIT_ANGLE \
      --cache_sampling_noise True \
      --seed_mode random 
  done
done

# =================================================================
