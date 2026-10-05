#!/usr/bin/env bash
#
# evaluate pretrained oxford flowers models with fdr-6
#
# usage:
# ./scripts/eval_flowers.sh      # single node
# ./scripts/eval_flowers.sh -d   # slurm multi-node
#
# reference stats are read from workspace/fdr6_stats
# generated images go to workspace/evaluation/<job_dir>, removed after scoring
# result tables are saved to workspace/experiments/<job_dir>/eval

DIST_MODE="local"
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -d)
      DIST_MODE="distributed"
      shift
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}"

# models to evaluate, one folder each under workspace/experiments
# checkpoints: https://huggingface.co/tomg-group-umd/sphere2
JOB_DIRS=(
  # "sphere2-base-flowers-256px"
  # "sphere2-base-flowers-512px"
  "sphere2-large-flowers-256px"
  # "sphere2-large-flowers-512px"
)


# ========================== Evaluation ===========================

for JOB_DIR in "${JOB_DIRS[@]}"; do

  # sampling angle per model size
  if [[ $JOB_DIR == sphere2-large-* ]]; then
    SAMPLING_ANGLE=84.0
  else
    SAMPLING_ANGLE=85.5
  fi

  # lower both batch sizes if the gpu runs out of memory at 512px
  ./run.sh \
    --dist_mode=$DIST_MODE \
    fdr6/eval_fdr6.py \
    --job_dir $JOB_DIR \
    --out_dir evaluation \
    --use_ema_model True \
    --num_eval_samples 51000 \
    --batch_size_per_rank 25 \
    --metric_batch_size_per_rank 25 \
    --forward_steps 2 \
    --sampling_init_angle $SAMPLING_ANGLE \
    --cache_sampling_noise True \
    --seed_sampling False \
    --use_cfg False \
    --cfg_min 0.0 \
    --cfg_max 5.0 \
    --cfg_gap 5.0 \
    --cfg_position angle \
    --fid_stats_used_from extr \
    --report_fid gfid \
    --rm_folder_after_eval True
done

# =================================================================
