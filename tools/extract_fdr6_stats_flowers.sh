#!/usr/bin/env bash
#
# extract the fdr-6 reference stats of oxford flowers, needed by scripts/eval_flowers.sh
#
# usage (from the repo root):
# ./tools/extract_fdr6_stats_flowers.sh      # single node
# ./tools/extract_fdr6_stats_flowers.sh -d   # slurm multi-node
#
# the reference set is the train + val splits, the held-out set is the test split
# (val.json if there is no test.json, which then overlaps the reference)
# stats are saved to workspace/fdr6_stats:
#   fdr6_stats_extr_flowers-102_<size>px_<encoder>_t<target>.npz
#   fdr6_valfd_extr_flowers-102_<size>px.json

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

IMAGE_SIZES=(256 512)

# the six encoders; run a subset to split the work, the stats are merged
ENCODERS="inception convnext dinov2 mae siglip clip"


# ========================== Extraction ===========================

for IMAGE_SIZE in "${IMAGE_SIZES[@]}"; do
  ./run.sh \
    --dist_mode=$DIST_MODE \
    fdr6/extract_fdr6_stats.py \
    --dataset_name flowers-102 \
    --image_size $IMAGE_SIZE \
    --encoders $ENCODERS \
    --compute_valfd True \
    --dtype float32 \
    --batch_size_per_rank 50 \
    --num_workers 16 \
    --out_dir fdr6_stats
done

# =================================================================
