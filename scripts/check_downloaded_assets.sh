#!/usr/bin/env bash
# Verify downloaded assets and, when requested, generated GraRe data.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/check_downloaded_assets.sh [options]

Verify GraspNet-1Billion, the MobileSAM and Point-MAE weights, and the five
released detector checkpoints. Select HGGD, RNGNet, or Generalizing-Grasp
explicitly with --detector to verify its optional checkpoint. --with-dumps
additionally verifies detector outputs. --with-features verifies the complete
GraRe feature tree for one detector-camera setting.

Options:
  --workspace DIR                  Asset workspace (default: $GRARE_ASSET_WORKSPACE or ./grare-assets)
  --graspnet-root DIR              GraspNet-1Billion root
  --detector-checkpoint-root DIR   Detector checkpoint root
  --sam-checkpoint FILE            MobileSAM checkpoint
  --point-mae-checkpoint FILE      Point-MAE checkpoint
  --dump-root DIR                  Frozen-detector dump root
  --data-root DIR                  GraRe generated-data root
  --detector NAME                  Check one detector checkpoint instead of all
  --camera NAME                    Camera for --detector (realsense or kinect)
  --with-dumps                     Also require frozen-detector .npy dumps
  --with-features                  Also require complete labels, object-cloud,
                                   Point-MAE, and manifest outputs for the
                                   selected --detector and --camera
  -h, --help                       Show this help text
EOF
}

workspace="${GRARE_ASSET_WORKSPACE:-$(pwd)/grare-assets}"
graspnet_root="${GRASPNET_ROOT:-}"
detector_checkpoint_root="${GRARE_DETECTOR_CKPT_ROOT:-}"
sam_checkpoint="${GRARE_SAM_CKPT:-}"
point_mae_checkpoint="${GRARE_POINT_MAE_CKPT:-}"
dump_root="${GRARE_DUMP_ROOT:-}"
data_root="${GRARE_DATA_ROOT:-}"
with_dumps=false
with_features=false
detector=""
camera="realsense"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dump-root)
      dump_root="$2"
      shift 2
      ;;
    --with-dumps)
      with_dumps=true
      shift
      ;;
    --with-features)
      with_features=true
      shift
      ;;
    --data-root)
      data_root="$2"
      shift 2
      ;;
    --detector)
      detector="$2"
      shift 2
      ;;
    --camera)
      camera="$2"
      shift 2
      ;;
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
dump_root="${dump_root:-$workspace/detector_dumps}"
data_root="${data_root:-$workspace/grare_data}"

if [[ "$with_features" == true && -z "$detector" ]]; then
  printf '%s\n' '--with-features requires --detector and --camera.' >&2
  exit 2
fi

failed=0
checked=0

check_dir() {
  local label="$1" path="$2" pattern="${3:-*}"
  if [[ ! -d "$path" ]]; then
    printf 'MISSING: %s: %s\n' "$label" "$path" >&2
    failed=1
    return
  fi
  # An empty directory is not a usable asset: the workspace script creates the
  # tree up front, so existence alone says nothing about the download. -L follows
  # symlinks, since these directories are often links to another disk.
  if ! find -L "$path" -mindepth 1 -name "$pattern" -print -quit 2>/dev/null | grep -q .; then
    printf 'MISSING: %s is empty (expected %s): %s\n' "$label" "$pattern" "$path" >&2
    failed=1
    return
  fi
  printf 'ok: %s\n' "$label"
  checked=$((checked + 1))
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

count_files() {
  local path="$1" pattern="$2"
  if [[ ! -d "$path" ]]; then
    printf '0\n'
    return
  fi
  find -L "$path" -type f -name "$pattern" -print 2>/dev/null | wc -l
}

count_camera_dumps() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    printf '0\n'
    return
  fi
  find -L "$path" -type f -path "*/$camera/*.npy" -print 2>/dev/null | wc -l
}

check_archive_tree() {
  local label="$1" path="$2" expected="$3"
  local actual
  actual="$(count_files "$path" '*.npz')"
  if [[ "$actual" -ne "$expected" ]]; then
    printf 'INCOMPLETE: %s: %s / %s archives under %s\n' \
      "$label" "$actual" "$expected" "$path" >&2
    failed=1
    return
  fi
  if find -L "$path" -type f -name '*.npz' -size 0 -print -quit 2>/dev/null | grep -q .; then
    printf 'INVALID: %s contains an empty archive: %s\n' "$label" "$path" >&2
    failed=1
    return
  fi
  printf 'ok: %s (%s archives)\n' "$label" "$actual"
  checked=$((checked + 1))
}

check_prepared_split() {
  local split="$1" expected="$2"
  local prepared_root="$data_root/relabeled/$detector/$camera"
  local local_root="$prepared_root/local_cloud/$split"
  local manifest="$local_root/manifest.jsonl"
  local summary="$local_root/manifest.summary.json"
  local manifest_count

  check_archive_tree "$detector $camera $split local_cloud" "$local_root" "$expected"
  check_archive_tree "$detector $camera $split object_cloud" \
    "$prepared_root/object_cloud/$split" "$expected"
  check_archive_tree "$detector $camera $split object_pooled" \
    "$prepared_root/object_pooled/$split" "$expected"

  if [[ ! -s "$manifest" ]]; then
    printf 'MISSING: %s manifest: %s\n' "$split" "$manifest" >&2
    failed=1
  else
    manifest_count="$(awk 'END { print NR + 0 }' "$manifest")"
    if [[ "$manifest_count" -ne "$expected" ]]; then
      printf 'INCOMPLETE: %s manifest: %s / %s records: %s\n' \
        "$split" "$manifest_count" "$expected" "$manifest" >&2
      failed=1
    else
      printf 'ok: %s manifest (%s records)\n' "$split" "$manifest_count"
      checked=$((checked + 1))
    fi
  fi
  check_file "$detector $camera $split manifest summary" "$summary"
}

check_dir 'GraspNet scenes' "$graspnet_root/scenes" 'scene_*'
check_dir 'GraspNet models' "$graspnet_root/models" 'nontextured.ply'
# grare-prepare needs the prebuilt Dex-Net caches. Without them the official API
# falls back to a code path that uses np.int, which NumPy 2.x removed.
check_dir 'GraspNet dex_models' "$graspnet_root/dex_models" '*.pkl'
check_file 'MobileSAM weight' "$sam_checkpoint"
check_file 'Point-MAE weight' "$point_mae_checkpoint"

check_selected_checkpoint() {
  case "$detector:$camera" in
    graspnet_baseline:realsense)
      check_file 'GN RealSense checkpoint' "$detector_checkpoint_root/graspnet_baseline/checkpoint-rs.tar"
      ;;
    graspnet_baseline:kinect)
      check_file 'GN Kinect checkpoint' "$detector_checkpoint_root/graspnet_baseline/checkpoint-kn.tar"
      ;;
    scale_balanced_grasp:realsense)
      check_file 'SBG RealSense checkpoint' "$detector_checkpoint_root/scale_balanced_grasp/log_full_model/checkpoint.tar"
      ;;
    economicgrasp:realsense)
      check_file 'EG RealSense checkpoint' "$detector_checkpoint_root/economicgrasp/economicgrasp_realsense.tar"
      ;;
    economicgrasp:kinect)
      check_file 'EG Kinect checkpoint' "$detector_checkpoint_root/economicgrasp/economicgrasp_kinect.tar"
      ;;
    hggd:realsense)
      check_file 'HGGD RealSense checkpoint' "$detector_checkpoint_root/hggd/realsense_checkpoint"
      ;;
    hggd:kinect)
      check_file 'HGGD Kinect checkpoint' "$detector_checkpoint_root/hggd/kinect_checkpoint"
      ;;
    rngnet:realsense)
      check_file 'RNGNet RealSense checkpoint' "$detector_checkpoint_root/rngnet/realsense.pth"
      ;;
    rngnet:kinect)
      check_file 'RNGNet Kinect checkpoint' "$detector_checkpoint_root/rngnet/kinect.pth"
      ;;
    generalizing_grasp:realsense)
      check_file 'Generalizing-Grasp RealSense checkpoint' "$detector_checkpoint_root/generalizing_grasp/checkpoint.tar"
      ;;
    generalizing_grasp:kinect)
      printf 'UNSUPPORTED: Generalizing-Grasp publishes a RealSense checkpoint only.\n' >&2
      failed=1
      ;;
    scale_balanced_grasp:kinect)
      printf 'UNSUPPORTED: Scale-Balanced-Grasp has no published Kinect checkpoint.\n' >&2
      failed=1
      ;;
    *)
      printf 'INVALID: unsupported --detector/--camera selection.\n' >&2
      failed=1
      ;;
  esac
}

if [[ -n "$detector" ]]; then
  check_selected_checkpoint
else
  check_file 'GN RealSense checkpoint' "$detector_checkpoint_root/graspnet_baseline/checkpoint-rs.tar"
  check_file 'GN Kinect checkpoint' "$detector_checkpoint_root/graspnet_baseline/checkpoint-kn.tar"
  check_file 'SBG RealSense checkpoint' "$detector_checkpoint_root/scale_balanced_grasp/log_full_model/checkpoint.tar"
  check_file 'EG RealSense checkpoint' "$detector_checkpoint_root/economicgrasp/economicgrasp_realsense.tar"
  check_file 'EG Kinect checkpoint' "$detector_checkpoint_root/economicgrasp/economicgrasp_kinect.tar"
fi

if [[ "$with_dumps" == true ]]; then
  if [[ -n "$detector" ]]; then
    dump_train="$(count_camera_dumps "$dump_root/$detector/train")"
    dump_test="$(count_camera_dumps "$dump_root/$detector/test")"
    if [[ "$dump_train" -eq 25600 && "$dump_test" -eq 23040 ]]; then
      printf 'ok: frozen-detector dumps (%s train, %s test)\n' "$dump_train" "$dump_test"
      checked=$((checked + 1))
    else
      printf 'INCOMPLETE: frozen-detector dumps for %s %s: train %s / 25600, test %s / 23040\n' \
        "$detector" "$camera" "$dump_train" "$dump_test" >&2
      failed=1
    fi
  elif find -L "$dump_root" -type f -name '*.npy' -print -quit 2>/dev/null | grep -q .; then
    printf 'ok: %s\n' 'frozen-detector dumps'
    checked=$((checked + 1))
  else
    printf 'MISSING: frozen-detector .npy dumps under: %s\n' "$dump_root" >&2
    failed=1
  fi
fi

if [[ "$with_features" == true ]]; then
  check_prepared_split train 25600
  check_prepared_split test 23040
fi

if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

printf 'Asset check passed (%d checks).\n' "$checked"
