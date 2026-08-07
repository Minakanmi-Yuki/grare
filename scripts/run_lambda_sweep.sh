#!/usr/bin/env bash
# Evaluate the requested intermediate z-score fusion weights serially. Each
# evaluation uses 24 CPU processes and resumes per-scene evaluator shards.
set -euo pipefail

source "${GRARE_ASSET_WORKSPACE:-/root/autodl-tmp/grare-assets}/grare_paths.env"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
grare_run="${GRARE_RUN:-/root/miniconda3/envs/grare/bin/grare-run}"

configs=(
  gn_realsense
  gn_kinect
  sbg_realsense
  eg_realsense
  eg_kinect
  hggd_realsense
  hggd_kinect
  rngnet_realsense
  rngnet_kinect
)
lambdas=(0.2 0.4 0.5 0.6 0.7 0.8 0.9)

for config in "${configs[@]}"; do
  for lambda in "${lambdas[@]}"; do
    lambda_tag="${lambda/./p}"
    prediction_dir="${GRARE_OUTPUT_ROOT}/predictions/${config}_lambda_${lambda_tag}"
    evaluation_dir="${GRARE_OUTPUT_ROOT}/evaluation/${config}_lambda_${lambda_tag}"
    if [[ -f "${evaluation_dir}/per_scene_raw.json" ]]; then
      echo "[lambda] complete ${config} lambda=${lambda}"
      continue
    fi
    if [[ ! -f "${prediction_dir}/rerank_summary.json" ]]; then
      echo "[lambda] rerank ${config} lambda=${lambda}"
      "${grare_run}" --config "configs/${config}.yaml" \
        --set "rerank.lambda=${lambda}" \
        --set eval.proc=24 \
        --start-from rerank \
        --stop-after rerank
    fi
    echo "[lambda] eval ${config} lambda=${lambda}"
    "${grare_run}" --config "configs/${config}.yaml" \
      --set "rerank.lambda=${lambda}" \
      --set eval.proc=24 \
      --start-from eval \
      --stop-after eval
  done
done
