# GraRe: Grasp Candidate Re-Ranking for Frozen 6-DoF Grasp Detectors

<p align="center">
  Jibao Yuan · Yuhui Zhao · Yinzhen Lv · Chao Xu · Shun Li · Chenxi Deng · Shaofei Chen*
</p>

## Abstract

Existing 6-DoF grasp detectors typically rank grasp candidates by detector
confidence. However, our analysis on GraspNet-1Billion shows that detector
confidence is often poorly aligned with grasp quality, causing successful
grasp candidates to be ranked too low during execution. Motivated by this
observation, we formulate grasp candidate re-ranking as a separate task for
frozen detectors, aiming to improve candidate ordering without changing the
detector or its grasp candidates. We propose GraRe, which estimates grasp
quality from candidate attributes, shell-stratified local geometry, and object
context. Candidate attributes condition the local geometric and object-context
representations, and a Transformer fuses all three feature types. The
predicted quality is combined with detector confidence to produce the final
ranking. Experiments on GraspNet-1Billion with three frozen detectors show
consistent improvements, with gains of up to 13.60 points in Average AP.
Real-robot experiments further demonstrate robust grasping in cluttered
scenes. These results show that improving candidate ranking provides a
practical way to enhance frozen 6-DoF grasp detectors.

## Overview

<p align="center">
  <img src="assets/grare-architecture.png" alt="GraRe architecture: frozen detector, candidate encoder, feature fusion, and re-ranking." width="100%" />
</p>

Given the grasp candidates produced by a frozen detector, the task keeps all
candidates unchanged and predicts a new ordering. GraRe evaluates each
candidate using three complementary types of information: candidate
attributes, shell-stratified local geometry, and object context. The
shell-stratified representation preserves geometry across different distances
from the gripper, while object context describes the candidate relative to the
visible object. Candidate attributes condition both geometric representations
before a Transformer fuses all three to predict grasp quality. GraRe combines
the predicted quality with detector confidence for final ranking.

## Environment

Ubuntu 22.04.5 LTS · Python 3.12.3 · PyTorch 2.12.1+cu130 (CUDA 13.0)

NumPy 2.4.6 · SciPy 1.18.0 · PyYAML 6.0.3 · OpenCV 4.13.0.92 · timm 1.0.27 · MobileSAM 1.0

## Installation

Clone GraRe, create the validated Conda environment, and install the project
dependencies:

```bash
git clone https://github.com/Minakanmi-Yuki/grare.git
cd grare
conda create -n grare python=3.12 -y
conda activate grare
python -m pip install --upgrade pip
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.12.1+cu130 torchvision==0.27.1+cu130
python -m pip install -e .
python -m pip install "timm>=0.9" "pytest>=7"
```

Install [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), which GraRe
uses to construct visible-object masks during feature preparation:

```bash
python -m pip install \
  "mobile-sam @ git+https://github.com/ChaoningZhang/MobileSAM.git"
```

To regenerate frozen-detector candidate dumps, install the shared detector
runtime and build dependencies. This requires a CUDA toolkit with `nvcc` that
is compatible with the PyTorch build above:

```bash
conda install -y -c anaconda openblas-devel
conda install -y -c conda-forge ninja
python -m pip install tensorboard open3d Pillow
```

Clone the detector repositories:
[GraspNet-Baseline](https://github.com/graspnet/graspnet-baseline),
[Scale-Balanced-Grasp](https://github.com/mahaoxiang822/Scale-Balanced-Grasp),
and [EconomicGrasp](https://github.com/iSEE-Laboratory/EconomicGrasp):

```bash
mkdir -p external
git clone https://github.com/graspnet/graspnet-baseline external/graspnet-baseline
git clone https://github.com/mahaoxiang822/Scale-Balanced-Grasp external/Scale-Balanced-Grasp
git clone https://github.com/iSEE-Laboratory/EconomicGrasp external/EconomicGrasp
```

Build the CUDA extensions and verify their imports before generating detector
dumps:

```bash
./scripts/build_detector_extensions.sh
./scripts/verify_detector_extensions.sh
```

The build script applies the included CUDA 13 compatibility patch to the
[EconomicGrasp](https://github.com/iSEE-Laboratory/EconomicGrasp)
MinkowskiEngine and reuses the
[GraspNet-Baseline](https://github.com/graspnet/graspnet-baseline) KNN
extension for
[Scale-Balanced-Grasp](https://github.com/mahaoxiang822/Scale-Balanced-Grasp)
on PyTorch 2.x. Do not install the upstream Scale-Balanced-Grasp
`requirements.txt`, which pins an incompatible historic PyTorch release.

Install the official [GraspNet API](https://github.com/graspnet/graspnetAPI)
and its `grasp_nms` extension only when running official AP evaluation:

```bash
git clone https://github.com/graspnet/graspnetAPI ../graspnetAPI
python -m pip install -e ../graspnetAPI
python -m pip install grasp_nms
```

For a different CUDA version, install the matching PyTorch wheel before
installing GraRe and the dependencies listed above. Other public upstream
projects are listed in [DEPENDENCIES.md](DEPENDENCIES.md).

## Downloads

GraRe requires the original GraspNet-1Billion dataset, detector checkpoints,
and the MobileSAM and Point-MAE backbone weights. Set the asset directory,
then initialize its paths; this does not download assets:

```bash
export GRARE_ASSET_WORKSPACE=/path/to/grare-assets
./scripts/prepare_data_assets.sh --workspace "$GRARE_ASSET_WORKSPACE"
source "$GRARE_ASSET_WORKSPACE/grare_paths.env"
```

Manually download GraspNet-1Billion from the
[official GraspNet page](https://graspnet.net/datasets.html), then extract the
archives to `$GRASPNET_ROOT` (default:
`$GRARE_ASSET_WORKSPACE/graspnet`). Then download the public
[MobileSAM weight](https://huggingface.co/dhkim2810/MobileSAM) and
[Point-MAE weight](https://github.com/Pang-Yatian/Point-MAE/releases/tag/main)
into the names expected by GraRe:

```bash
./scripts/prepare_data_assets.sh \
  --workspace "$GRARE_ASSET_WORKSPACE" \
  --download-backbones
```

To regenerate candidate dumps, download the five published detector
checkpoints: [GN RealSense](https://drive.google.com/file/d/1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk/view?usp=sharing),
[GN Kinect](https://drive.google.com/file/d/1vK-d0yxwyJwXHYWOtH1bDMoe--uZ2oLX/view?usp=sharing),
[SBG RealSense](https://drive.google.com/drive/folders/1Y2o0uAbhS6-yZhPkKMAAnL0tCpbiJfk0?usp=share_link),
and the [EG v1 release](https://github.com/iSEE-Laboratory/EconomicGrasp/releases/tag/v1)
for both cameras. Place them under `$GRARE_DETECTOR_CKPT_ROOT` as shown below.

The downloaded assets should be arranged as follows:

```text
$GRARE_ASSET_WORKSPACE/
├── graspnet/                                      GraspNet-1Billion
│   ├── scenes/
│   └── models/
├── detector_checkpoints/                          frozen detector weights
│   ├── graspnet_baseline/
│   │   ├── checkpoint-rs.tar                       GN RealSense
│   │   └── checkpoint-kn.tar                       GN Kinect
│   ├── scale_balanced_grasp/
│   │   └── log_full_model/checkpoint.tar           SBG RealSense
│   └── economicgrasp/
│       ├── economicgrasp_realsense.tar             EG RealSense
│       └── economicgrasp_kinect.tar                EG Kinect
└── backbones/                                     frozen feature backbones
    ├── mobile_sam.pt
    └── point_mae_pretrain.pth
```

The five detector-camera checkpoints above correspond to the five paper
configurations.

Verify that all downloaded assets are in place before generating candidate
dumps:

```bash
./scripts/check_downloaded_assets.sh
```

## Prepare

GraRe uses the three inputs described in the paper: unchanged candidate
attributes, shell-stratified local geometry, and object context. Run each
frozen detector with its downloaded checkpoint on the GraspNet train and test
splits, retaining every `(K, 17)` GraspGroup output under
`$GRARE_DUMP_ROOT`:

```text
$GRARE_DUMP_ROOT/$DETECTOR/$CAMERA/
├── train/scene_0000/$CAMERA/0000.npy
└── test/scene_0100/$CAMERA/0000.npy
```

Set the detector once, then prepare one split. The defaults in
`grare-prepare` reproduce the paper's four shells, per-shell sampling budgets,
and 512-point local and object clouds:

```bash
DETECTOR=graspnet_baseline
CAMERA=realsense
SPLIT=train

grare-prepare \
  --input-root "$GRARE_DUMP_ROOT/$DETECTOR/$CAMERA/$SPLIT" \
  --input-format detector-dump \
  --output-root "$GRARE_DATA_ROOT/relabeled/$DETECTOR/$CAMERA/local_cloud/$SPLIT" \
  --object-cloud-root "$GRARE_DATA_ROOT/relabeled/$DETECTOR/$CAMERA/object_cloud/$SPLIT" \
  --detector "$DETECTOR" \
  --dataset-root "$GRASPNET_ROOT" \
  --camera "$CAMERA" \
  --split "$SPLIT" \
  --sam-checkpoint "$GRARE_SAM_CKPT" \
  --omit-object-cloud
```

This preserves candidate attributes in `local_cloud/`, samples four radial
shells for local geometry, and writes MobileSAM object clouds to the
`object_cloud/` sidecar. Add `--limit 1` for a one-frame check, then set
`SPLIT=test` and run the same command again.

Precompute the frozen Point-MAE object features for the same split:

```bash
grare-precompute-object \
  --archive-root "$GRARE_DATA_ROOT/relabeled/$DETECTOR/$CAMERA/local_cloud/$SPLIT" \
  --object-cloud-root "$GRARE_DATA_ROOT/relabeled/$DETECTOR/$CAMERA/object_cloud/$SPLIT" \
  --output-root "$GRARE_DATA_ROOT/relabeled/$DETECTOR/$CAMERA/object_pooled/$SPLIT" \
  --pmae-ckpt "$GRARE_POINT_MAE_CKPT"
```

Set `SPLIT=test` and run it again. The resulting three input assets are:

```text
$GRARE_DATA_ROOT/relabeled/$DETECTOR/$CAMERA/
├── local_cloud/       candidate attributes and shell-wise local geometry
├── object_cloud/      MobileSAM object point clouds
└── object_pooled/     frozen Point-MAE object features
```

Set `DETECTOR` and `CAMERA` to the detector-camera pair used by the selected
paper configuration.

## Release Scope

This repository contains GraRe source code, paper configurations, tests, and
documentation. It does not redistribute datasets, detector repositories or
weights, GraRe checkpoints, or generated predictions. See
[docs/PUBLICATION_SCOPE.md](docs/PUBLICATION_SCOPE.md) for the release boundary
and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution.

## Reported Main Results

The following offline GraspNet-1Billion results use the official evaluation
protocol. Values are AP (%); GraRe re-ranks the unchanged candidate set from
each frozen detector. `—` denotes an unavailable result. See
[docs/RESULTS.md](docs/RESULTS.md) for the reproduction context and compact
gain summary.

### RealSense

| Frozen detector | Ranking | Seen | Similar | Novel | Average |
| --- | --- | ---: | ---: | ---: | ---: |
| [GraspNet-Baseline](https://github.com/graspnet/graspnet-baseline) | Detector | 47.83 | 42.79 | 16.94 | 35.85 |
| [GraspNet-Baseline](https://github.com/graspnet/graspnet-baseline) | **GraRe** | **64.48** | **58.78** | **25.10** | **49.45** |
| [Scale-Balanced-Grasp](https://github.com/mahaoxiang822/Scale-Balanced-Grasp) | Detector | 62.27 | 56.92 | 23.80 | 47.66 |
| [Scale-Balanced-Grasp](https://github.com/mahaoxiang822/Scale-Balanced-Grasp) | **GraRe** | **68.76** | **62.64** | **27.51** | **52.97** |
| [EconomicGrasp](https://github.com/iSEE-Laboratory/EconomicGrasp) | Detector | 69.30 | 61.50 | 25.28 | 52.02 |
| [EconomicGrasp](https://github.com/iSEE-Laboratory/EconomicGrasp) | **GraRe** | **75.12** | **64.39** | **28.34** | **55.95** |

### Kinect

| Frozen detector | Ranking | Seen | Similar | Novel | Average |
| --- | --- | ---: | ---: | ---: | ---: |
| [GraspNet-Baseline](https://github.com/graspnet/graspnet-baseline) | Detector | 41.97 | 37.56 | 12.24 | 30.59 |
| [GraspNet-Baseline](https://github.com/graspnet/graspnet-baseline) | **GraRe** | **53.94** | **46.39** | **16.04** | **38.79** |
| [Scale-Balanced-Grasp](https://github.com/mahaoxiang822/Scale-Balanced-Grasp) | Detector | — | — | — | — |
| [Scale-Balanced-Grasp](https://github.com/mahaoxiang822/Scale-Balanced-Grasp) | **GraRe** | **—** | **—** | **—** | **—** |
| [EconomicGrasp](https://github.com/iSEE-Laboratory/EconomicGrasp) | Detector | 63.75 | 52.43 | 19.61 | 45.26 |
| [EconomicGrasp](https://github.com/iSEE-Laboratory/EconomicGrasp) | **GraRe** | **69.90** | **58.00** | **22.04** | **49.98** |

## Reproduce GraRe Step by Step

Work through the following stages in order. Each stage has a completion gate;
do not proceed when its gate fails.

| Stage | Goal | Needs external data or GPU? | Completion gate |
| --- | --- | --- | --- |
| 1 | Install and exercise the package | No | `grare-smoke` completes |
| 2 | Download and place required assets | Downloads only | GraspNet, backbones, and detector checkpoints are present |
| 3 | Prepare three input assets for one setting | GraspNet + detector dumps; GPU recommended | local archives and object-pooled sidecars exist |
| 4 | Reproduce one paper setting | Full assets + GPU | train, rerank, and evaluation artifacts exist |
| 5 | Repeat the five reported settings | Full assets + GPU | all five configuration graphs complete |
| 6 | Interpret the comparison | Complete official test evaluations | AP is compared with the reported table above |

Start with `gn_realsense`: it has the shortest supported path and does not
need mmap packing. The Kinect GN and EG settings require the additional
packing step described in Stage 4.

### 1. Verify installation

Confirm the package before downloading data:

```bash
grare-smoke
python -m pytest -q
```

### 2. Download and place assets

Follow [Downloads](#downloads) to acquire and place the dataset, detector
checkpoints, and backbone weights. Source the generated environment file before
continuing:

```bash
source "$GRARE_ASSET_WORKSPACE/grare_paths.env"
```

Generate candidate dumps as described in [Prepare](#prepare) before feature
construction. If the dataset is stored elsewhere, pass `--graspnet-root`; use
`./scripts/prepare_data_assets.sh --help` for all options.

### 3. Prepare three input assets for one paper setting

Follow [Prepare](#prepare) for candidate dumps, shell-wise local clouds, and
Point-MAE object features. The `object_pooled/` sidecar is the Stage 3
completion gate.

### 4. Reproduce one complete paper setting

The package provides five main configurations:

| Config | Frozen detector | Camera |
| --- | --- | --- |
| `configs/gn_realsense.yaml` | [GraspNet-Baseline](https://github.com/graspnet/graspnet-baseline) | RealSense |
| `configs/gn_kinect.yaml` | [GraspNet-Baseline](https://github.com/graspnet/graspnet-baseline) | Kinect |
| `configs/sbg_realsense.yaml` | [Scale-Balanced-Grasp](https://github.com/mahaoxiang822/Scale-Balanced-Grasp) | RealSense |
| `configs/eg_realsense.yaml` | [EconomicGrasp](https://github.com/iSEE-Laboratory/EconomicGrasp) | RealSense |
| `configs/eg_kinect.yaml` | [EconomicGrasp](https://github.com/iSEE-Laboratory/EconomicGrasp) | Kinect |

Every configuration uses batch size `2048`. The environment variables were
written in Stage 2; load them with
`source "$GRARE_ASSET_WORKSPACE/grare_paths.env"` before invoking a
configuration.

The Kinect GN and EG configurations use mmap-packed training features to
preserve the reported 2048-candidate batch construction. Build the matching
packed tree after object-feature precomputation (substitute the detector and
camera names for the selected configuration):

```bash
grare-pack \
  --input-root "$GRARE_DATA_ROOT/relabeled/graspnet_baseline/kinect/local_cloud/train" \
  --object-pooled-root "$GRARE_DATA_ROOT/relabeled/graspnet_baseline/kinect/object_pooled/train" \
  --output-root "$GRARE_DATA_ROOT/packed/graspnet_baseline/kinect/train" \
  --archive-manifest auto \
  --require-archive-manifest \
  --require-object-pooled
```

Inspect the resolved GN-RealSense sequence before using a GPU:

```bash
grare-run --config configs/gn_realsense.yaml --dry-run
```

Run the three stages in sequence. Stopping after training gives a convenient
checkpoint gate before creating prediction files and evaluating AP:

```bash
grare-run --config configs/gn_realsense.yaml --stop-after train
grare-run --config configs/gn_realsense.yaml --start-from rerank --stop-after rerank
grare-run --config configs/gn_realsense.yaml --start-from eval
```

The expected artifacts are `best.pt`, a re-ranking summary and records, then
`per_scene_raw.npy` and `per_scene_raw.json` in `$GRARE_OUTPUT_ROOT`. The
official test evaluation is valid only when all 90 test scenes and 256 frames
per scene have been re-ranked.

### 5. Repeat the reported configurations

After preparing the corresponding features for all five settings and
completing GN-RealSense, run every main-result configuration sequentially:

```bash
./scripts/run_paper_configs.sh
```

Use `./scripts/run_paper_configs.sh --dry-run` to inspect all five command
graphs. Use `--set train.seed=11` for a different initialization seed. Test
AP is not used for model or hyperparameter selection.

### 6. Compare and understand the results

Compare complete official evaluations with the AP tables above and the
interpretation in [docs/RESULTS.md](docs/RESULTS.md). For paired scene-level
confidence intervals, use the command in
[docs/REPRODUCTION.md](docs/REPRODUCTION.md#statistical-comparison). The
real-robot results are a display-only showcase in
[docs/REAL_ROBOT_RESULTS.md](docs/REAL_ROBOT_RESULTS.md), not a robot-control
reproduction target.

## Method Components

- Candidate features: pose, gripper width, and detector confidence.
- Local features: four radial shells with boundaries `(0, 5, 15, 25, 40)` mm
  and per-shell FPS budgets `(64, 128, 128, 192)`.
- Object context: MobileSAM mask prompting followed by a frozen Point-MAE
  encoder and a trainable projection adapter.
- Conditioning and fusion: candidate-conditioned FiLM for local and object
  features, followed by a three-token Transformer.
- Objective: continuous friction-margin quality prediction with collision,
  empty-grasp, and object-classification auxiliary losses.
- Re-ranking: candidate-set z-score normalization and score fusion with
  `lambda = 1.0` in the paper configurations.

## Reference Guides

- [Configuration reference](configs/README.md)
- [Detailed offline reproduction](docs/REPRODUCTION.md)
- [Expected benchmark results](docs/RESULTS.md)
- [Real-robot result showcase](docs/REAL_ROBOT_RESULTS.md)
