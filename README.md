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

Use a shell that has not sourced a ROS environment. ROS 2 puts its own
`site-packages` on `sys.path`, which shadows the NumPy 2.x build GraRe needs and
injects pytest plugins that fail on unrelated ROS dependencies.

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
python -m pip install -e '.[test]'
```

Verify the installation now, before downloading any data. Both commands run on
CPU and need no dataset:

```bash
grare-smoke
python -m pytest -q
```

Feature preparation and official AP evaluation additionally require the
official [GraspNet API](https://github.com/graspnet/graspnetAPI) and
[MobileSAM](https://github.com/ChaoningZhang/MobileSAM). Neither is on PyPI, so
clone both and install them from the checkout:

```bash
git clone https://github.com/graspnet/graspnetAPI ../graspnetAPI
python -m pip install -e ../graspnetAPI

git clone https://github.com/ChaoningZhang/MobileSAM.git ../MobileSAM
python -m pip install -e ../MobileSAM

python -m pip install grasp_nms
python -m pip install -e '.[prepare]'
```

`graspnetAPI` supplies the analytical force-closure, collision, and
empty-grasp labels used by `grare-prepare`, and the official evaluator used by
`grare-evaluate`. MobileSAM supplies the visible-object masks.

Clone both explicitly rather than letting pip fetch them: the MobileSAM
repository carries a 40 MB weight file, and a slow clone inside pip shows no
progress and cannot be retried on its own. If GitHub is slow or unreachable,
enable your proxy or mirror for the two `git clone` commands only.

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

Download every asset yourself from its official source, under that project's own
license and terms. GraRe never downloads or redistributes them.

| Asset | Official source | Place at |
| --- | --- | --- |
| GraspNet-1Billion | [graspnet.net/datasets.html](https://graspnet.net/datasets.html) | `$GRASPNET_ROOT` |
| MobileSAM weight | [MobileSAM repository](https://github.com/ChaoningZhang/MobileSAM), `weights/mobile_sam.pt` | `$GRARE_SAM_CKPT` |
| Point-MAE weight | [Point-MAE release](https://github.com/Pang-Yatian/Point-MAE/releases/tag/main), `pretrain.pth` | `$GRARE_POINT_MAE_CKPT` |
| GN checkpoints | [GraspNet-Baseline](https://github.com/graspnet/graspnet-baseline#training-and-testing), `checkpoint-rs.tar` and `checkpoint-kn.tar` | `$GRARE_DETECTOR_CKPT_ROOT/graspnet_baseline/` |
| SBG checkpoint | [Scale-Balanced-Grasp](https://github.com/mahaoxiang822/Scale-Balanced-Grasp#test), the `log_full_model/checkpoint.tar` in its Drive folder | `$GRARE_DETECTOR_CKPT_ROOT/scale_balanced_grasp/log_full_model/` |
| EG checkpoints | [EconomicGrasp v1 release](https://github.com/iSEE-Laboratory/EconomicGrasp/releases/tag/v1), `economicgrasp_realsense.tar` and `economicgrasp_kinect.tar` | `$GRARE_DETECTOR_CKPT_ROOT/economicgrasp/` |

The detector checkpoints are not on the front page of their repositories.
GraspNet-Baseline links them under "Training and Testing" and Scale-Balanced-Grasp
under "Train&Test → Test"; both offer Google Drive and Baidu Pan mirrors. The
Scale-Balanced-Grasp link opens a Drive folder rather than a file, and the
`log_full_model/checkpoint.tar` name comes from its own `command_test.sh`.
EconomicGrasp attaches both weights to its v1 release page directly.

### Extracting GraspNet-1Billion

The download page offers one archive per split rather than a single dataset
tree, so extract them into a shared `$GRASPNET_ROOT` and let their `scenes/`
directories merge. GraRe needs these five archives:

| Archive | Extracts to | Contents |
| --- | --- | --- |
| `train_1.zip` … `train_4.zip` | `scenes/scene_0000` … `scene_0099` | train scenes |
| `test_seen.zip` | `scenes/scene_0100` … `scene_0129` | Seen test scenes |
| `test_similar.zip` | `scenes/scene_0130` … `scene_0159` | Similar test scenes |
| `test_novel.zip` | `scenes/scene_0160` … `scene_0189` | Novel test scenes |
| `models.zip` | `models/` | object meshes |
| `dex_models.zip` | `dex_models/` | Dex-Net caches |

For example, with every archive in one directory:

```bash
mkdir -p "$GRASPNET_ROOT"
for archive in train_1 train_2 train_3 train_4 test_seen test_similar test_novel models dex_models; do
  unzip -q -n "$archive.zip" -d "$GRASPNET_ROOT"
done
```

The result must be `$GRASPNET_ROOT/scenes` with 190 `scene_XXXX` directories,
plus `models/` and `dex_models/`:

```bash
ls "$GRASPNET_ROOT"                     # scenes  models  dex_models
ls "$GRASPNET_ROOT/scenes" | wc -l      # 190
```

If an archive expands into a nested directory such as `train_1/scenes/`, move
the `scenes/` contents up so all 190 scenes share one `scenes/` directory.

`grasp_label.zip`, `collision_label.zip`, and `rect_labels.zip` are needed only
to train a detector from scratch. GraRe re-ranks frozen detector outputs, so it
does not read them.

The download page marks `dex_models.zip` as optional, but GraRe requires it: it
holds the prebuilt Dex-Net caches the official API uses for the analytical
force-closure labels. Without them the API rebuilds each model through a code
path that calls `np.int`, which NumPy 2.x removed, so `grare-prepare` fails.

`grare-prepare` also constructs the official API over the whole split, which
reads `scenes/scene_XXXX/object_id_list.txt` for **every** scene in that split
even when you only process a few frames. A partial dataset therefore needs at
least that file present for all 100 train or 90 test scenes.

Rename the two backbone weights to `mobile_sam.pt` and
`point_mae_pretrain.pth`; the Point-MAE release asset is named `pretrain.pth`
upstream. The detector checkpoints are needed only to regenerate candidate
dumps, and each upstream project documents its own download location for them.

The downloaded assets should be arranged as follows:

```text
$GRARE_ASSET_WORKSPACE/
├── graspnet/                                      GraspNet-1Billion
│   ├── scenes/                                     190 scene_XXXX directories
│   │   ├── scene_0000/                              train: 0000-0099
│   │   │   ├── realsense/                            rgb, depth, label, meta
│   │   │   ├── kinect/
│   │   │   └── object_id_list.txt
│   │   └── scene_0189/                              test: 0100-0189
│   ├── models/                                     object meshes, 000-087
│   └── dex_models/                                 000.pkl - 087.pkl
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

These five detector-camera checkpoints correspond to the five reported
settings.

Verify that all downloaded assets are in place before generating candidate
dumps:

```bash
./scripts/check_downloaded_assets.sh
```

## Prepare

First, run each frozen detector with its downloaded checkpoint on the GraspNet
train and test splits, retaining every unchanged `(K, 17)` GraspGroup output.
This requires the detector sources and CUDA extensions from
[Installation](#installation).

`grare-dump` runs a frozen detector over one split and writes the dumps that
`grare-prepare` consumes. Repeat it for each detector, camera, and split used by
the setting you want to reproduce:

```bash
grare-dump --detector graspnet_baseline --camera realsense --split train
grare-dump --detector graspnet_baseline --camera realsense --split test
```

Add `--deterministic` for bit-reproducible dumps on a fixed machine: the
upstream detectors use nondeterministic CUDA kernels, so repeated runs
otherwise vary the confidence column by roughly `1e-4` without changing grasp
poses. Use `--dry-run` to preview the upstream command.

The recommended dump settings are stored in the detector configs, so the
optimized command needs no tuning flags. They use batch size 24, eight data
workers, 32 post-process workers, prefetch factor 4, pinned memory, and resume
from existing files:

For the small container CPU quota, also limit NumPy/BLAS thread pools once per
shell (these variables are not detector-config parameters):

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

```bash
grare-dump --detector graspnet_baseline --camera realsense --split test
```

For SBG and EG, replace the detector name. GN, SBG, and EG now all use the
configured post-processing worker pool. GN and SBG should report
`"shared_collision_cloud": true` in their startup JSON after
`scripts/build_detector_extensions.sh` has been run.

GN and SBG can also be split across GPUs with `--index-shard-count` and
`--index-shard-id`; EG does not support sharding and rejects those flags:

```bash
CUDA_VISIBLE_DEVICES=0 grare-dump --detector graspnet_baseline --camera realsense \
  --split train --index-shard-count 2 --index-shard-id 0 &
CUDA_VISIBLE_DEVICES=1 grare-dump --detector graspnet_baseline --camera realsense \
  --split train --index-shard-count 2 --index-shard-id 1 &
wait
```

These flags change throughput only, not which candidates the detector emits.
Use the same values for the train and test splits of one setting, though: dumps
regenerated under different conditions vary slightly, so mixing them
introduces avoidable inconsistency between the two splits.

Regenerated dumps are not expected to match another machine's dumps exactly.
GPU model, driver, and CUDA version shift the detector's float outputs
slightly, and Scale-Balanced-Grasp additionally applies a score threshold, so
its candidate count can differ by a few grasps between machines. GraRe
re-ranks whichever candidate set the detector produces, so AP reproduced from
freshly generated dumps can differ marginally from the reported values.

Dumps are written per detector and split, with the camera below each scene:

```text
$GRARE_DUMP_ROOT/$DETECTOR/
├── train/scene_0000/$CAMERA/0000.npy
└── test/scene_0100/$CAMERA/0000.npy
```

The five reported settings need these dump groups:

```text
graspnet_baseline     realsense + kinect
scale_balanced_grasp  realsense
economicgrasp         realsense + kinect
```

Once the dumps exist, confirm the complete feature-construction input set:

```bash
./scripts/check_downloaded_assets.sh --with-dumps
```

Then set the detector once and prepare one split. The recommended command
separates the CPU labels pass from the single-GPU SAM pass. It reproduces the
reported four shells, per-shell sampling budgets, and 512-point clouds:

```bash
DETECTOR=graspnet_baseline
CAMERA=realsense
SPLIT=train
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# CPU: analytic labels and local geometry
grare-prepare --stage labels --num-workers 20 \
  --input-root "$GRARE_DUMP_ROOT/$DETECTOR/$SPLIT" \
  --pattern "scene_*/$CAMERA/*.npy" --input-format detector-dump \
  --output-root "$GRARE_DATA_ROOT/relabeled/$DETECTOR/$CAMERA/local_cloud/$SPLIT" \
  --detector "$DETECTOR" --dataset-root "$GRASPNET_ROOT" \
  --camera "$CAMERA" --split "$SPLIT" --omit-object-cloud

# GPU: add MobileSAM object clouds (one worker on a single GPU)
grare-prepare --stage object --num-workers 1 \
  --input-root "$GRARE_DATA_ROOT/relabeled/$DETECTOR/$CAMERA/local_cloud/$SPLIT" \
  --output-root "$GRARE_DATA_ROOT/relabeled/$DETECTOR/$CAMERA/local_cloud/$SPLIT" \
  --object-cloud-root "$GRARE_DATA_ROOT/relabeled/$DETECTOR/$CAMERA/object_cloud/$SPLIT" \
  --detector "$DETECTOR" --dataset-root "$GRASPNET_ROOT" \
  --camera "$CAMERA" --split "$SPLIT" \
  --sam-checkpoint "$GRARE_SAM_CKPT" --omit-object-cloud
```

The process is resumable: rerunning the same commands skips archives already
written. Add `--limit 1` to the labels command for a one-frame smoke test, then
set `SPLIT=test` for the test split.

This stage runs over every frame of every scene and dominates preparation time.
For the current container (25 CPU quota but roughly 96 GB cgroup memory), use
twenty labels workers: each worker owns a GraspNet/Dex-Net scene cache, so CPU
quota is not a safe memory limit. Keep the NumPy/BLAS pools at one thread per
process. The stage is resumable: an interrupted run can be repeated with the
same command and skips the frames it already wrote.

Progress is reported every 30 seconds as a JSON line with the archive count,
percentage, rate, and ETA, counted from the archives already on disk so it
advances even mid-batch. Set `GRARE_PREPARE_LOG_EVERY_SEC` to report more often.
The default `GRARE_PREPARE_CHUNK_SIZE=256` keeps each standard GraspNet scene in
one task; smaller values increase repeated scene-cache loading.

`--stage labels` keeps SAM off even if a checkpoint is passed, so it never
touches the GPU. `--stage object` reuses the labels already on disk and only adds
the object cloud. The two stages together produce byte-identical archives to the
single-pass command, so pick whichever fits the host.

Each object-stage worker holds its own MobileSAM model and CUDA context, so on a
single-GPU machine `--num-workers 12` creates twelve model copies and can OOM or
appear to hang. The prepare command now caps labels workers at both the
effective CPU quota and a memory-safe default of twenty (override deliberately
with `GRARE_PREPARE_MAX_LABEL_WORKERS`), and caps CUDA SAM workers at the number
of visible GPUs. MobileSAM prompt decoding defaults to a batch of 64; lower
`--sam-prompt-batch-size` if a smaller GPU runs out of VRAM. Switching
scenes evicts that cache and re-reads the scene's `dex_models` entries and object
meshes, which are far larger than the frames themselves; frames of the same scene
are grouped into scene-sized chunks (256 frames for the standard GraspNet layout)
to keep the cache alive and avoid reloading the same meshes in multiple workers.
Override this with `GRARE_PREPARE_CHUNK_SIZE` only for an unusual layout or a
smoke test; smaller values increase repeated scene loading.

On a mechanical disk those cache refills are the limiting factor, because several
workers reading different scenes at once turn them into random I/O. Watch
`%util` and `await` while a run is in progress and lower `--num-workers` if the
disk is saturated:

```bash
iostat -x 2
```

Putting `dex_models/` and `models/` on an SSD addresses this directly and is
cheaper than moving the whole dataset, since they are small next to the frames
but account for most of the repeated reads:

```bash
# Copy first and verify before removing the original: a cross-filesystem move
# that is interrupted leaves the data in neither place.
cp -r "$GRASPNET_ROOT/dex_models" /path/on/ssd/dex_models
diff -r "$GRASPNET_ROOT/dex_models" /path/on/ssd/dex_models && \
  rm -rf "$GRASPNET_ROOT/dex_models" && \
  ln -s /path/on/ssd/dex_models "$GRASPNET_ROOT/dex_models"
```

With those on an SSD, only the frame reads stay on the mechanical disk.

Precompute the frozen Point-MAE object features for the same split:

```bash
grare-precompute-object \
  --archive-root "$GRARE_DATA_ROOT/relabeled/$DETECTOR/$CAMERA/local_cloud/$SPLIT" \
  --object-cloud-root "$GRARE_DATA_ROOT/relabeled/$DETECTOR/$CAMERA/object_cloud/$SPLIT" \
  --output-root "$GRARE_DATA_ROOT/relabeled/$DETECTOR/$CAMERA/object_pooled/$SPLIT" \
  --pmae-ckpt "$GRARE_POINT_MAE_CKPT"
```

The precompute command fuses adjacent archives into GPU batches (default
`--batch-size 4096`) and runs the frozen backbone in inference mode. Use
`--num-shards N --shard K` to split the work across separate GPUs; keep one
process per GPU.

Set `SPLIT=test` and run it again. The resulting three input assets are:

```text
$GRARE_DATA_ROOT/relabeled/$DETECTOR/$CAMERA/
├── local_cloud/       candidate attributes and shell-wise local geometry
├── object_cloud/      MobileSAM object point clouds
└── object_pooled/     frozen Point-MAE object features
```

Set `DETECTOR` and `CAMERA` to the detector-camera pair used by the selected
configuration you are reproducing.

## Train

Training and everything after it need a GPU. The package provides five main
configurations, one per reported setting:

| Config | Frozen detector | Camera |
| --- | --- | --- |
| `configs/gn_realsense.yaml` | [GraspNet-Baseline](https://github.com/graspnet/graspnet-baseline) | RealSense |
| `configs/gn_kinect.yaml` | [GraspNet-Baseline](https://github.com/graspnet/graspnet-baseline) | Kinect |
| `configs/sbg_realsense.yaml` | [Scale-Balanced-Grasp](https://github.com/mahaoxiang822/Scale-Balanced-Grasp) | RealSense |
| `configs/eg_realsense.yaml` | [EconomicGrasp](https://github.com/iSEE-Laboratory/EconomicGrasp) | RealSense |
| `configs/eg_kinect.yaml` | [EconomicGrasp](https://github.com/iSEE-Laboratory/EconomicGrasp) | Kinect |

Start with `gn_realsense`: it has the shortest supported path and does not need
mmap packing. Every configuration uses batch size `2048`. Load the environment
file written during [Downloads](#downloads) before invoking a configuration:

```bash
source "$GRARE_ASSET_WORKSPACE/grare_paths.env"
```

The Kinect GN and EG configurations read mmap-packed training features to
preserve the reported 2048-candidate batch construction. Build the packed tree
after object-feature precomputation, substituting the detector and camera of
the configuration you are running:

```bash
grare-pack \
  --input-root "$GRARE_DATA_ROOT/relabeled/graspnet_baseline/kinect/local_cloud/train" \
  --object-pooled-root "$GRARE_DATA_ROOT/relabeled/graspnet_baseline/kinect/object_pooled/train" \
  --output-root "$GRARE_DATA_ROOT/packed/graspnet_baseline/kinect/train" \
  --archive-manifest auto \
  --require-archive-manifest \
  --require-object-pooled
```

Inspect the resolved command graph before using a GPU:

```bash
grare-run --config configs/gn_realsense.yaml --dry-run
```

Then train and re-rank. Stopping after training gives a checkpoint gate before
prediction files are written:

```bash
grare-run --config configs/gn_realsense.yaml --stop-after train
grare-run --config configs/gn_realsense.yaml --start-from rerank --stop-after rerank
```

Training writes `best.pt` to `$GRARE_OUTPUT_ROOT/checkpoints/<name>/`, selected
by held-out scene-level validation loss; test AP is never used to choose an
epoch or a hyperparameter. Re-ranking writes evaluator-compatible `.npy` files
plus a summary and per-frame records to `$GRARE_OUTPUT_ROOT/predictions/<name>/`.

`grare-rerank` consumes the relabeled `.npz` archives from `grare-prepare`, not
raw detector `.npy` dumps. It keeps every candidate and changes only the order.

Each of the other four settings runs the same way once its features exist.
Substitute its configuration name, and expect each one to take hours:

```bash
grare-run --config configs/eg_realsense.yaml --stop-after train
```

Use `--dry-run` to inspect a resolved command graph without touching the GPU,
or `--set train.seed=11` for a different initialization seed.
## Evaluate

Run the official GraspNet evaluator on the re-ranked predictions:

```bash
grare-run --config configs/gn_realsense.yaml --start-from eval
```

This writes `per_scene_raw.npy` and `per_scene_raw.json` to
`$GRARE_OUTPUT_ROOT/evaluation/<name>/`.

Official AP requires the complete test protocol: all 90 test scenes and all 256
frames per scene must have been re-ranked. `grare-evaluate` checks this before
calling the evaluator and refuses an incomplete dump. A one-frame or one-scene
run is useful for debugging the pipeline but must not be reported as an AP.

To obtain the detector baseline that the reported gains are measured against,
re-run the same configuration with `rerank.lambda=0.0`, which keeps the
detector's own ranking. Its artifacts go to separate `*_detector_baseline`
directories, so they cannot overwrite the GraRe results:

```bash
grare-run --config configs/gn_realsense.yaml --start-from rerank --set rerank.lambda=0.0
```

For paired scene-level confidence intervals between the two complete
evaluations:

```bash
python scripts/paired_scene_bootstrap.py \
  --baseline "$GRARE_OUTPUT_ROOT/evaluation/gn_realsense_detector_baseline/per_scene_raw.npy" \
  --treatment "$GRARE_OUTPUT_ROOT/evaluation/gn_realsense/per_scene_raw.npy" \
  --output paired_bootstrap.json
```

Compare the resulting AP with [Reported Results](#reported-results) and the
interpretation in [docs/RESULTS.md](docs/RESULTS.md).
## Reported Results

The following offline GraspNet-1Billion results use the official evaluation
protocol. Values are AP (%); GraRe re-ranks the unchanged candidate set from
each frozen detector. `—` denotes an unavailable result. The Detector rows are the baseline described in [Evaluate](#evaluate). See
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

Scale-Balanced-Grasp is RealSense-only because no Kinect checkpoint is
published upstream.

The real-robot outcomes are a display-only showcase in
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
  `lambda = 1.0` in the reported settings.

## Release Scope

This repository contains GraRe source code, configurations, tests, and
documentation. It does not redistribute datasets, detector repositories or
weights, GraRe checkpoints, or generated predictions. See
[docs/SCOPE.md](docs/SCOPE.md) for the release boundary
and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution.

## Reference Guides

- [Configurations](configs/README.md)
- [Offline reproduction](docs/REPRODUCTION.md)
- [Reported results](docs/RESULTS.md)
- [Real-robot results](docs/REAL_ROBOT_RESULTS.md)
- [Release scope](docs/SCOPE.md)
- [Data contract](DATA_FORMAT.md)
