# GraRe: Grasp Candidate Re-Ranking for Frozen 6-DoF Grasp Detectors

<p align="center">
  Jibao Yuan · Yuhui Zhao · Yinzhen Lv · Chao Xu · Shun Li · Chenxi Deng · Shaofei Chen
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
| GraspNet-Baseline | Detector | 47.83 | 42.79 | 16.94 | 35.85 |
| GraspNet-Baseline | **GraRe** | **64.48** | **58.78** | **25.10** | **49.45** |
| Scale-Balanced-Grasp | Detector | 62.27 | 56.92 | 23.80 | 47.66 |
| Scale-Balanced-Grasp | **GraRe** | **68.76** | **62.64** | **27.51** | **52.97** |
| EconomicGrasp | Detector | 69.30 | 61.50 | 25.28 | 52.02 |
| EconomicGrasp | **GraRe** | **75.12** | **64.39** | **28.34** | **55.95** |

### Kinect

| Frozen detector | Ranking | Seen | Similar | Novel | Average |
| --- | --- | ---: | ---: | ---: | ---: |
| GraspNet-Baseline | Detector | 41.97 | 37.56 | 12.24 | 30.59 |
| GraspNet-Baseline | **GraRe** | **53.94** | **46.39** | **16.04** | **38.79** |
| Scale-Balanced-Grasp | Detector | — | — | — | — |
| Scale-Balanced-Grasp | **GraRe** | **—** | **—** | **—** | **—** |
| EconomicGrasp | Detector | 63.75 | 52.43 | 19.61 | 45.26 |
| EconomicGrasp | **GraRe** | **69.90** | **58.00** | **22.04** | **49.98** |

## Reproduce GraRe Step by Step

Work through the following stages in order. Each stage has a completion gate;
do not proceed when its gate fails.

| Stage | Goal | Needs external data or GPU? | Completion gate |
| --- | --- | --- | --- |
| 1 | Install and exercise the package | No | `grare-smoke` completes |
| 2 | Download and place GraspNet-1Billion | Dataset download only | `$GRASPNET_ROOT/scenes` exists |
| 3 | Build train/test features for one setting | GraspNet + detector dumps; GPU recommended | local archives and object-pooled sidecars exist |
| 4 | Reproduce one paper setting | Full assets + GPU | train, rerank, and evaluation artifacts exist |
| 5 | Repeat the five reported settings | Full assets + GPU | all five configuration graphs complete |
| 6 | Interpret the comparison | Complete official test evaluations | AP is compared with the reported table above |

Start with `gn_realsense`: it has the shortest supported path and does not
need mmap packing. The Kinect GN and EG settings require the additional
packing step described in Stage 4.

### 1. Install and validate the package

Python 3.10 or newer is required. Install a PyTorch build compatible with the
intended CUDA version before installing GraRe.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
grare-smoke
```

`grare-smoke` uses generated inputs and checks a synthetic train → checkpoint
reload → re-ranking loop. It needs neither a dataset, checkpoint, nor GPU.
Run the unit suite before investing in full preprocessing:

```bash
python -m pytest -q
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

### 2. Download and place GraspNet-1Billion

Start by creating the local workspace from the repository root. This command
does not download any dataset:

```bash
./scripts/prepare_data_assets.sh \
  --workspace "$PWD/grare-assets"
source "$PWD/grare-assets/grare_paths.env"
```

Download the original **GraspNet-1Billion** dataset from the
[official GraspNet download page](https://graspnet.net/datasets.html), accept
its terms, and extract it into the workspace so that this directory exists:

```text
grare-assets/
  graspnet/
    scenes/
      scene_0000/
      ...
```

The default dataset root is therefore `$PWD/grare-assets/graspnet`; the
generated environment file exposes the same path as `$GRASPNET_ROOT`. Confirm
this first data-acquisition milestone with:

```bash
test -d "$GRASPNET_ROOT/scenes" && echo "GraspNet-1Billion is ready"
```

The workspace also reserves `detector_dumps/`, `backbones/`, `grare_data/`,
and `grare_output/` for later stages. Do not run the full `--check` yet: it
also requires candidate dumps and the two public backbone checkpoints needed
only when feature construction begins.

Before Stage 3, add frozen-detector candidate dumps and download the public
MobileSAM and Point-MAE checkpoints:

```bash
./scripts/prepare_data_assets.sh \
  --workspace "$PWD/grare-assets" \
  --download-backbones \
  --check
```

If either input is stored elsewhere, pass `--graspnet-root` or
`--detector-dumps`; see [DEPENDENCIES.md](DEPENDENCIES.md) for their original
public sources. Use `./scripts/prepare_data_assets.sh --help` for all options.

### 3. Build features for one paper setting

GraRe starts from frozen-detector candidate dumps in GraspNet `(K, 17)` array
format. It does not require detector source code during feature preparation,
training, re-ranking, or evaluation. See [DATA_FORMAT.md](DATA_FORMAT.md) for
the complete schema and directory layout. For the recommended first setting,
put the train and test dumps under:

```text
$GRARE_DUMP_ROOT/graspnet_baseline/realsense/train/
$GRARE_DUMP_ROOT/graspnet_baseline/realsense/test/
```

First generate the training archives and sidecar object clouds:

```bash
grare-prepare \
  --input-root "$GRARE_DUMP_ROOT/graspnet_baseline/realsense/train" \
  --input-format detector-dump \
  --output-root "$GRARE_DATA_ROOT/relabeled/graspnet_baseline/realsense/local_cloud/train" \
  --object-cloud-root "$GRARE_DATA_ROOT/relabeled/graspnet_baseline/realsense/object_cloud/train" \
  --detector graspnet_baseline \
  --benchmark graspnet \
  --dataset-root "$GRASPNET_ROOT" \
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

For a low-cost format check, append `--limit 1` to this command first. Then
run it again without `--limit`. A successful full run writes a
`manifest.jsonl` plus matching archives under `local_cloud/` and
`object_cloud/`. Repeat the same command for the `test` input and output
directories, changing only `train` to `test`.

Next precompute the frozen Point-MAE features for the training split:

```bash
grare-precompute-object \
  --archive-root "$GRARE_DATA_ROOT/relabeled/graspnet_baseline/realsense/local_cloud/train" \
  --object-cloud-root "$GRARE_DATA_ROOT/relabeled/graspnet_baseline/realsense/object_cloud/train" \
  --output-root "$GRARE_DATA_ROOT/relabeled/graspnet_baseline/realsense/object_pooled/train" \
  --pmae-ckpt "$GRARE_POINT_MAE_CKPT" \
  --object-cloud-points 512 \
  --device cuda
```

Repeat Point-MAE precomputation for the test split. Its `object_pooled/`
sidecar is the Stage 3 completion gate. Analytical test labels are needed only
by the official evaluator and analyses; they are not used for checkpoint
selection.

### 4. Reproduce one complete paper setting

The package provides five main configurations:

| Config | Frozen detector | Camera |
| --- | --- | --- |
| `configs/gn_realsense.yaml` | GraspNet-Baseline | RealSense |
| `configs/gn_kinect.yaml` | GraspNet-Baseline | Kinect |
| `configs/sbg_realsense.yaml` | Scale-Balanced-Grasp | RealSense |
| `configs/eg_realsense.yaml` | EconomicGrasp | RealSense |
| `configs/eg_kinect.yaml` | EconomicGrasp | Kinect |

Every configuration uses batch size `2048`. The environment variables were
written in Stage 2; load them with
`source "$PWD/grare-assets/grare_paths.env"` before invoking a configuration.

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
