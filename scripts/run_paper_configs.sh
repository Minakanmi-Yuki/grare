#!/usr/bin/env bash
set -euo pipefail

# Run the five main-result command graphs sequentially. Arguments are passed
# through to grare-run, e.g. --dry-run or --stop-after train.
configs=(
  configs/gn_realsense.yaml
  configs/gn_kinect.yaml
  configs/sbg_realsense.yaml
  configs/eg_realsense.yaml
  configs/eg_kinect.yaml
)

for config in "${configs[@]}"; do
  echo "==> ${config}"
  python -m grare.cli.run --config "$config" "$@"
done
