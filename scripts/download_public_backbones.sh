#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:-checkpoints}"
mkdir -p "$output_dir"

curl -L --fail --retry 5 \
  https://huggingface.co/dhkim2810/MobileSAM/resolve/main/mobile_sam.pt \
  -o "$output_dir/mobile_sam.pt"
curl -L --fail --retry 5 \
  https://github.com/Pang-Yatian/Point-MAE/releases/download/main/pretrain.pth \
  -o "$output_dir/point_mae_pretrain.pth"

printf 'Downloaded public backbone checkpoints to %s\n' "$output_dir"
