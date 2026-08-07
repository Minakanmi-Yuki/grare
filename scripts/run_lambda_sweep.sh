#!/usr/bin/env bash
# Evaluate the requested intermediate z-score fusion weights serially. Each
# evaluation resumes per-scene evaluator shards.
set -euo pipefail

source "${GRARE_ASSET_WORKSPACE:-/root/autodl-tmp/grare-assets}/grare_paths.env"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
grare_run="${GRARE_RUN:-/root/miniconda3/envs/grare/bin/grare-run}"

default_configs=(
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
default_lambdas=(0.2 0.4 0.5 0.6 0.7 0.8 0.9)

# Space-separated overrides allow a long sweep to be split across queues.
if [[ -n "${GRARE_LAMBDA_CONFIGS:-}" ]]; then
  read -r -a configs <<< "${GRARE_LAMBDA_CONFIGS}"
else
  configs=("${default_configs[@]}")
fi
if [[ -n "${GRARE_LAMBDAS:-}" ]]; then
  read -r -a lambdas <<< "${GRARE_LAMBDAS}"
else
  lambdas=("${default_lambdas[@]}")
fi
eval_proc="${GRARE_EVAL_PROC:-24}"
if ! [[ "${eval_proc}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GRARE_EVAL_PROC must be a positive integer, got: ${eval_proc}" >&2
  exit 2
fi

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
        --set "eval.proc=${eval_proc}" \
        --start-from rerank \
        --stop-after rerank
    fi
    echo "[lambda] eval ${config} lambda=${lambda}"
    "${grare_run}" --config "configs/${config}.yaml" \
      --set "rerank.lambda=${lambda}" \
      --set "eval.proc=${eval_proc}" \
      --start-from eval \
      --stop-after eval
  done
done
