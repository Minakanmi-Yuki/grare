#!/usr/bin/env bash
# Verify only the assets acquired in the README Downloads section.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/check_downloaded_assets.sh [options]

Verify GraspNet-1Billion, the MobileSAM and Point-MAE weights, and the five
published detector checkpoints. Candidate dumps are generated later and are
not checked here.

Options:
  --workspace DIR                  Asset workspace (default: $GRARE_ASSET_WORKSPACE or ./grare-assets)
  --graspnet-root DIR              GraspNet-1Billion root
  --detector-checkpoint-root DIR   Detector checkpoint root
  --sam-checkpoint FILE            MobileSAM checkpoint
  --point-mae-checkpoint FILE      Point-MAE checkpoint
  -h, --help                       Show this help text
EOF
}

workspace="${GRARE_ASSET_WORKSPACE:-$(pwd)/grare-assets}"
graspnet_root="${GRASPNET_ROOT:-}"
detector_checkpoint_root="${GRARE_DETECTOR_CKPT_ROOT:-}"
sam_checkpoint="${GRARE_SAM_CKPT:-}"
point_mae_checkpoint="${GRARE_POINT_MAE_CKPT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace)
      workspace="$2"
      shift 2
      ;;
    --graspnet-root)
      graspnet_root="$2"
      shift 2
      ;;
    --detector-checkpoint-root)
      detector_checkpoint_root="$2"
      shift 2
      ;;
    --sam-checkpoint)
      sam_checkpoint="$2"
      shift 2
      ;;
    --point-mae-checkpoint)
      point_mae_checkpoint="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

graspnet_root="${graspnet_root:-$workspace/graspnet}"
detector_checkpoint_root="${detector_checkpoint_root:-$workspace/detector_checkpoints}"
sam_checkpoint="${sam_checkpoint:-$workspace/backbones/mobile_sam.pt}"
point_mae_checkpoint="${point_mae_checkpoint:-$workspace/backbones/point_mae_pretrain.pth}"

failed=0
checked=0

check_dir() {
  local label="$1" path="$2"
  if [[ -d "$path" ]]; then
    printf 'ok: %s\n' "$label"
    checked=$((checked + 1))
  else
    printf 'MISSING: %s: %s\n' "$label" "$path" >&2
    failed=1
  fi
}

check_file() {
  local label="$1" path="$2"
  if [[ -s "$path" ]]; then
    printf 'ok: %s\n' "$label"
    checked=$((checked + 1))
  else
    printf 'MISSING: %s: %s\n' "$label" "$path" >&2
    failed=1
  fi
}

check_dir 'GraspNet scenes' "$graspnet_root/scenes"
check_dir 'GraspNet models' "$graspnet_root/models"
check_file 'MobileSAM weight' "$sam_checkpoint"
check_file 'Point-MAE weight' "$point_mae_checkpoint"
check_file 'GN RealSense checkpoint' "$detector_checkpoint_root/graspnet_baseline/checkpoint-rs.tar"
check_file 'GN Kinect checkpoint' "$detector_checkpoint_root/graspnet_baseline/checkpoint-kn.tar"
check_file 'SBG RealSense checkpoint' "$detector_checkpoint_root/scale_balanced_grasp/log_full_model/checkpoint.tar"
check_file 'EG RealSense checkpoint' "$detector_checkpoint_root/economicgrasp/economicgrasp_realsense.tar"
check_file 'EG Kinect checkpoint' "$detector_checkpoint_root/economicgrasp/economicgrasp_kinect.tar"

if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

printf 'Downloaded asset check passed (%d checks).\n' "$checked"
