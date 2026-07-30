#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/download_public_backbones.sh [output-dir]

Download the public MobileSAM and Point-MAE checkpoints required for feature
construction. This script does not download GraspNet-1Billion or detector
outputs; see scripts/prepare_data_assets.sh for the full asset workflow.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi

command -v curl >/dev/null || {
  printf '%s\n' 'curl is required to download public backbone checkpoints.' >&2
  exit 127
}

output_dir="${1:-checkpoints}"
mkdir -p "$output_dir"

download() {
  local url="$1"
  local destination="$2"
  local temporary="${destination}.partial"
  rm -f "$temporary"
  curl -L --fail --retry 5 --retry-all-errors "$url" -o "$temporary"
  test -s "$temporary"
  mv "$temporary" "$destination"
}

download \
  https://huggingface.co/dhkim2810/MobileSAM/resolve/main/mobile_sam.pt \
  "$output_dir/mobile_sam.pt"
download \
  https://github.com/Pang-Yatian/Point-MAE/releases/download/main/pretrain.pth \
  "$output_dir/point_mae_pretrain.pth"

printf 'Downloaded public backbone checkpoints to %s\n' "$output_dir"
