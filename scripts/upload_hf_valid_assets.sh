#!/usr/bin/env bash
# Publish only completed, supported detector/camera releases. The publisher
# records immutable five-scene shards in the Hub index, so rerunning resumes.
set -euo pipefail

if [[ -f /etc/network_turbo ]]; then
  source /etc/network_turbo
fi
source "${GRARE_ASSET_WORKSPACE:-/root/autodl-tmp/grare-assets}/grare_paths.env"
export HF_TOKEN="$(<"${HF_TOKEN_FILE:-/root/hf-token.txt}")"
grare_python="${GRARE_PYTHON:-/root/miniconda3/envs/grare/bin/python}"

settings=(
  "graspnet_baseline realsense gn_realsense"
  "graspnet_baseline kinect gn_kinect"
  "scale_balanced_grasp realsense sbg_realsense"
  "economicgrasp realsense eg_realsense"
  "economicgrasp kinect eg_kinect"
  "hggd realsense hggd_realsense"
  "hggd kinect hggd_kinect"
  "rngnet realsense rngnet_realsense"
  "rngnet kinect rngnet_kinect"
)

for setting in "${settings[@]}"; do
  read -r detector camera config <<<"${setting}"
  echo "[upload] checkpoint ${config}"
  "${grare_python}" scripts/publish_hf_assets.py checkpoint --config "configs/${config}.yaml"
  echo "[upload] dumps ${detector}/${camera}"
  "${grare_python}" scripts/publish_hf_assets.py dumps --detector "${detector}" --camera "${camera}"
  echo "[upload] features ${detector}/${camera}"
  "${grare_python}" scripts/publish_hf_assets.py features --detector "${detector}" --camera "${camera}"
done
