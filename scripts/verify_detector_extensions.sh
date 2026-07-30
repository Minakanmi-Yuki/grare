#!/usr/bin/env bash
# Verify imports provided by scripts/build_detector_extensions.sh.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/verify_detector_extensions.sh [--skip-economic] [--import-only]

Imports every compiled detector extension. --import-only does not require a
working NVIDIA driver, which is useful immediately after a build.
EOF
}

skip_economic=false
import_only=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-economic) skip_economic=true ;;
    --import-only) import_only=true ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

PROJECT_ROOT="$project_root" SKIP_ECONOMIC="$skip_economic" IMPORT_ONLY="$import_only" python - <<'PY'
import importlib
import os
import sys
from pathlib import Path

import torch

root = Path(os.environ["PROJECT_ROOT"])
skip_economic = os.environ["SKIP_ECONOMIC"] == "true"
import_only = os.environ["IMPORT_ONLY"] == "true"
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if not torch.cuda.is_available() and not import_only:
    raise SystemExit("CUDA is unavailable. Re-run with --import-only to verify imports only.")


def reset_modules() -> None:
    prefixes = ("pointnet2", "knn_pytorch", "MinkowskiEngine", "MinkowskiEngineBackend", "libs.pointnet2", "libs.knn")
    for name in list(sys.modules):
        if name.startswith(prefixes):
            sys.modules.pop(name, None)


def import_with_paths(label: str, paths: list[Path], modules: list[str]) -> None:
    reset_modules()
    original_path = list(sys.path)
    sys.path[:0] = [str(path) for path in paths]
    try:
        for module in modules:
            importlib.import_module(module)
            print("ok", label, module)
    finally:
        sys.path = original_path


import_with_paths(
    "graspnet_baseline",
    [root / "external/graspnet-baseline/pointnet2", root / "external/graspnet-baseline/knn"],
    ["pointnet2_utils", "knn_modules"],
)
import_with_paths(
    "scale_balanced_grasp",
    [root / "external/Scale-Balanced-Grasp/pointnet2", root / "external/graspnet-baseline/knn"],
    ["pointnet2_utils", "knn_modules"],
)
if not skip_economic:
    import_with_paths(
        "economicgrasp",
        [root / "external/EconomicGrasp", root / "external/EconomicGrasp/libs/MinkowskiEngine"],
        ["MinkowskiEngine", "libs.pointnet2.pointnet2_utils", "libs.knn.knn_modules"],
    )

print("Detector extension import verification complete.")
PY
