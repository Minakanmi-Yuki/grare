# GraRe

GraRe re-ranks the unchanged grasp candidates produced by a frozen 6-DoF
grasp detector. For each candidate, it combines candidate attributes,
shell-stratified local geometry, and visible-object context to predict grasp
quality. The predicted quality and detector confidence are normalized within
the candidate set and combined to produce the final order.

This repository contains source code only. It does not include datasets,
detector repositories, detector weights, GraRe checkpoints, or generated
predictions. See [docs/PUBLICATION_SCOPE.md](docs/PUBLICATION_SCOPE.md) for
the release boundary and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for
upstream attribution.

## Overview

<img src="assets/grare-architecture.png" alt="GraRe architecture: frozen detector, candidate encoder, feature fusion, and re-ranking." width="75%" />

GraRe preserves the frozen detector and its candidate set. Candidate
attributes condition the local-geometry and object-context features through
FiLM; a three-token Transformer then predicts a quality score for each
candidate. Candidate-set z-score normalization fuses that score with the
detector confidence to obtain the final ranking.

## Reproducibility Guide

- [Paper configurations](configs/README.md) map the five reported settings to
  runnable YAML files.
- [Offline reproduction](docs/REPRODUCTION.md) documents the complete
  `prepare → precompute → train → rerank → evaluate` workflow.
- [Expected benchmark results](docs/RESULTS.md) records the reported
  GraspNet-1Billion values and what constitutes a valid comparison.
- [Real-robot results](docs/REAL_ROBOT_RESULTS.md) presents the physical
  evaluation as result evidence only; this repository does not provide a
  robot-control reproduction stack.

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

## Installation

Python 3.10 or newer is required. Install a PyTorch build compatible with the
intended CUDA version before installing GraRe.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
grare-smoke
```

Feature construction additionally requires MobileSAM:

```bash
python -m pip install -e '.[prepare]'
```

Official GraspNet evaluation requires a separate installation of the public
GraspNet API and its `grasp_nms` extension:

```bash
git clone https://github.com/graspnet/graspnetAPI /path/to/graspnetAPI
python -m pip install -e /path/to/graspnetAPI
```

The other public upstream projects are listed in
[DEPENDENCIES.md](DEPENDENCIES.md).

## Public Backbones

Download the public MobileSAM and Point-MAE checkpoints with:

```bash
./scripts/download_public_backbones.sh checkpoints
export GRARE_SAM_CKPT="$PWD/checkpoints/mobile_sam.pt"
export GRARE_POINT_MAE_CKPT="$PWD/checkpoints/point_mae_pretrain.pth"
```

The download script uses only the public upstream checkpoint URLs. Verify the
upstream licenses before redistributing those files.

## Input Contract

GraRe starts from frozen-detector candidate dumps in GraspNet `(K, 17)` array
format. It does not require detector source code during feature preparation,
training, re-ranking, or evaluation. See [DATA_FORMAT.md](DATA_FORMAT.md) for
the complete schema and directory layout.

Generate labels and shell-stratified local/object features from detector dumps:

```bash
grare-prepare \
  --input-root /path/to/detector/dumps/train \
  --input-format detector-dump \
  --output-root /path/to/grare-data/relabeled/graspnet_baseline/realsense/local_cloud/train \
  --object-cloud-root /path/to/grare-data/relabeled/graspnet_baseline/realsense/object_cloud/train \
  --detector graspnet_baseline \
  --benchmark graspnet \
  --dataset-root /path/to/graspnet \
  --camera realsense \
  --split train \
  --cloud-sampler stratified_fps \
  --shell-edges 0,0.005,0.015,0.025,0.040 \
  --shell-budgets 64,128,128,192 \
  --local-cloud-max-points 512 \
  --object-cloud-points 512 \
  --sam-checkpoint "$GRARE_SAM_CKPT" \
  --omit-object-cloud
```

For the recommended sidecar layout, first retain object clouds in an
`object_cloud` tree, then precompute frozen Point-MAE features:

```bash
grare-precompute-object \
  --archive-root /path/to/local_cloud/train \
  --object-cloud-root /path/to/object_cloud/train \
  --output-root /path/to/object_pooled/train \
  --pmae-ckpt "$GRARE_POINT_MAE_CKPT" \
  --object-cloud-points 512 \
  --device cuda
```

Repeat feature preparation for the test split. Analytical test labels are
needed only by the official evaluator and analyses; they are not used for
checkpoint selection.

## Paper Configurations

The package provides five main configurations:

| Config | Frozen detector | Camera |
| --- | --- | --- |
| `configs/gn_realsense.yaml` | GraspNet-Baseline | RealSense |
| `configs/gn_kinect.yaml` | GraspNet-Baseline | Kinect |
| `configs/sbg_realsense.yaml` | Scale-Balanced-Grasp | RealSense |
| `configs/eg_realsense.yaml` | EconomicGrasp | RealSense |
| `configs/eg_kinect.yaml` | EconomicGrasp | Kinect |

Every configuration uses batch size `2048`. Set the data and output roots:

```bash
export GRASPNET_ROOT=/path/to/graspnet
export GRARE_DATA_ROOT=/path/to/grare-data
export GRARE_OUTPUT_ROOT=/path/to/grare-output
export GRARE_POINT_MAE_CKPT=/path/to/point_mae_pretrain.pth
```

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

Inspect a full command sequence without executing it:

```bash
grare-run --config configs/gn_realsense.yaml --dry-run
```

Run training, re-ranking, and official evaluation:

```bash
grare-run --config configs/gn_realsense.yaml
```

Stages can be run separately:

```bash
grare-run --config configs/gn_realsense.yaml --stop-after train
grare-run --config configs/gn_realsense.yaml --start-from rerank --stop-after rerank
grare-run --config configs/gn_realsense.yaml --start-from eval
```

Use `--set train.seed=11` for a different initialization seed. Test AP is not
used for model or hyperparameter selection.

## Verification

Run the self-contained pipeline smoke test and the focused unit suite:

```bash
grare-smoke
python -m pytest -q
```

The smoke test uses generated inputs and does not require a dataset or trained
GraRe checkpoint. Exact paper results require training with the public data and
backbones described above.
