#!/usr/bin/env bash
# Build CUDA/C++ extensions required to regenerate frozen-detector dumps.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/build_detector_extensions.sh [--skip-economic]

Builds the PointNet2 and KNN operators for GraspNet-Baseline and
Scale-Balanced-Grasp, plus MinkowskiEngine, PointNet2, and KNN for
EconomicGrasp. Clone the three pinned detector sources into external/ first.

Environment variables:
  CUDA_HOME              CUDA toolkit root (default: /usr/local/cuda)
  TORCH_CUDA_ARCH_LIST   Optional CUDA architecture list for PyTorch extensions
  MAX_JOBS               Parallel compile jobs (default: 2)
  CUDA_THRUST_INCLUDE    Thrust/CUB include root for MinkowskiEngine
EOF
}

skip_economic=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-economic) skip_economic=true ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
max_jobs="${MAX_JOBS:-2}"
openblas_include="${OPENBLAS_INCLUDE_DIRS:-${CONDA_PREFIX:-}/include}"
openblas_library="${OPENBLAS_LIBRARY_DIRS:-${CONDA_PREFIX:-}/lib}"

if [[ -z "${CUDA_THRUST_INCLUDE:-}" ]]; then
  if [[ -f /usr/include/thrust/host_vector.h ]]; then
    cuda_thrust_include=/usr/include
  else
    cuda_thrust_include="$cuda_home/include/cccl"
  fi
else
  cuda_thrust_include="$CUDA_THRUST_INCLUDE"
fi

require_dir() {
  if [[ ! -d "$1" ]]; then
    printf 'Missing detector source directory: %s\n' "$1" >&2
    exit 1
  fi
}

run_setup_install() {
  local label="$1"
  local directory="$2"
  printf '== Building %s ==\n' "$label"
  (cd "$directory" && python setup.py install)
}

apply_economicgrasp_patch() {
  local patch_file="$project_root/scripts/patches/economicgrasp_minkowski_cuda13.patch"
  local me_root="$project_root/external/EconomicGrasp/libs/MinkowskiEngine"
  local concurrent="$me_root/src/3rdparty/concurrent_unordered_map.cuh"
  local functors="$me_root/src/coordinate_map_functors.cuh"
  local ranges="$me_root/src/3rdparty/cudf/detail/nvtx/ranges.hpp"
  local gpu_cu="$me_root/src/coordinate_map_gpu.cu"
  local spmm_cu="$me_root/src/spmm.cu"

  if grep -q 'cudaMemLocation location' "$concurrent" && \
     grep -q '#define CUDF_FUNC_RANGE()' "$ranges" && \
     ! grep -q 'thrust::unary_function' "$functors" && \
     grep -q '#include <thrust/remove.h>' "$gpu_cu" && \
     grep -q '#include <thrust/reduce.h>' "$spmm_cu"; then
    printf '%s\n' 'EconomicGrasp CUDA 13 compatibility patch already applied.'
    return
  fi
  printf '%s\n' '== Applying EconomicGrasp CUDA 13 compatibility patch =='
  (cd "$project_root" && patch -p0 < "$patch_file")
}

apply_graspnet_baseline_dump_patch() {
  local patch_file="$project_root/scripts/patches/graspnet_baseline_dump_fastpath.patch"
  local dataset="$project_root/external/graspnet-baseline/dataset/graspnet_dataset.py"
  local collision="$project_root/external/graspnet-baseline/utils/collision_detector.py"

  if grep -q 'return_raw_cloud_with_sample' "$dataset" && \
     grep -q 'downsample=True' "$collision"; then
    printf '%s\n' 'GraspNet-Baseline dump fast-path patch already applied.'
    return
  fi
  # The upstream checkout uses CRLF line endings.  Normalize the two patched
  # files so the repository patch remains portable across Linux hosts.
  sed -i 's/\r$//' "$dataset" "$collision"
  printf '%s\n' '== Applying GraspNet-Baseline dump fast-path patch =='
  (cd "$project_root" && patch -p0 < "$patch_file")
}

require_dir "$project_root/external/graspnet-baseline/pointnet2"
require_dir "$project_root/external/graspnet-baseline/knn"
require_dir "$project_root/external/Scale-Balanced-Grasp/pointnet2"
if [[ "$skip_economic" == false ]]; then
  require_dir "$project_root/external/EconomicGrasp/libs/MinkowskiEngine"
  require_dir "$project_root/external/EconomicGrasp/libs/pointnet2"
  require_dir "$project_root/external/EconomicGrasp/libs/knn"
fi
if [[ ! -x "$cuda_home/bin/nvcc" ]]; then
  printf 'nvcc not found at %s/bin/nvcc. Install a CUDA toolkit and set CUDA_HOME.\n' "$cuda_home" >&2
  exit 1
fi

export CUDA_HOME="$cuda_home"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS="$max_jobs"
export FORCE_CUDA=1
export OPENBLAS_INCLUDE_DIRS="$openblas_include"
export OPENBLAS_LIBRARY_DIRS="$openblas_library"
if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  export TORCH_CUDA_ARCH_LIST
fi

"$CUDA_HOME/bin/nvcc" --version
printf 'MAX_JOBS=%s\n' "$MAX_JOBS"
if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  printf 'TORCH_CUDA_ARCH_LIST=%s\n' "$TORCH_CUDA_ARCH_LIST"
fi

apply_graspnet_baseline_dump_patch
run_setup_install 'GraspNet-Baseline PointNet2' "$project_root/external/graspnet-baseline/pointnet2"
run_setup_install 'GraspNet-Baseline KNN' "$project_root/external/graspnet-baseline/knn"
run_setup_install 'Scale-Balanced-Grasp PointNet2' "$project_root/external/Scale-Balanced-Grasp/pointnet2"
printf '%s\n' 'Scale-Balanced-Grasp reuses the GraspNet-Baseline KNN extension on PyTorch 2.x.'

if [[ "$skip_economic" == false ]]; then
  apply_economicgrasp_patch
  printf '%s\n' '== Building EconomicGrasp MinkowskiEngine =='
  (
    cd "$project_root/external/EconomicGrasp/libs/MinkowskiEngine"
    python setup.py install \
      --force_cuda \
      --cuda_home="$CUDA_HOME" \
      --blas_include_dirs="$OPENBLAS_INCLUDE_DIRS,$cuda_thrust_include" \
      --blas_library_dirs="$OPENBLAS_LIBRARY_DIRS" \
      --blas=openblas
  )
  run_setup_install 'EconomicGrasp PointNet2' "$project_root/external/EconomicGrasp/libs/pointnet2"
  run_setup_install 'EconomicGrasp KNN' "$project_root/external/EconomicGrasp/libs/knn"
fi

printf '%s\n' 'Detector extension build complete.'
