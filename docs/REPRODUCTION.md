# Offline Reproduction

This guide reproduces the paper's offline GraspNet-1Billion pipeline. It does
not cover the physical robot setup; see [REAL_ROBOT_RESULTS.md](REAL_ROBOT_RESULTS.md)
for the display-only real-robot evidence.

## 1. Install dependencies

Install a PyTorch build appropriate for the target CUDA version, then install
GraRe and its feature-construction dependencies:

```bash
python -m pip install -e '.[test,prepare]'
```

The GraspNet API and MobileSAM are not on PyPI, so install both from their
repositories. `grare-prepare` and `grare-evaluate` require the API; feature
construction requires MobileSAM:

```bash
git clone https://github.com/graspnet/graspnetAPI ../graspnetAPI
python -m pip install -e ../graspnetAPI

git clone https://github.com/ChaoningZhang/MobileSAM.git ../MobileSAM
python -m pip install -e ../MobileSAM

python -m pip install grasp_nms
```

The official projects and licenses are listed in
[../DEPENDENCIES.md](../DEPENDENCIES.md).

## 2. Obtain inputs

### 2.1 Download GraspNet-1Billion first

Create the local asset workspace (this does not download data):

```bash
export GRARE_ASSET_WORKSPACE=/path/to/grare-assets
./scripts/prepare_data_assets.sh \
  --workspace "$GRARE_ASSET_WORKSPACE"
source "$GRARE_ASSET_WORKSPACE/grare_paths.env"
```

Download the original dataset from the
[official GraspNet download page](https://graspnet.net/datasets.html), accept
its terms, and extract it under `grare-assets/graspnet/`. Feature construction
needs `scenes/`, `models/`, and `dex_models/`; the last holds the Dex-Net
models the official API uses for the analytical force-closure labels:

```bash
./scripts/check_downloaded_assets.sh
```

### 2.2 Add feature-construction inputs later

When beginning Stage 3, obtain the backbone weights and the frozen-detector
checkpoints from their official sources, then generate a raw `(K, 17)`
candidate dump for every frame with `grare-dump`:

```bash
grare-dump --detector graspnet_baseline --camera realsense --split train
grare-dump --detector graspnet_baseline --camera realsense --split test
```

Pass `--deterministic` when you need bit-reproducible dumps on a fixed
machine; the upstream detectors otherwise vary the confidence column by about
`1e-4` between runs without changing grasp poses. Across different GPUs or CUDA
versions the detector outputs shift slightly, and Scale-Balanced-Grasp's score
threshold can admit or drop a few candidates, so AP reproduced from freshly
generated dumps can differ marginally from the reported values.

GraRe never changes candidate identities, poses, widths,
or candidate-set size. The exact dump contract is in
[../DATA_FORMAT.md](../DATA_FORMAT.md). The corresponding upstream projects
are listed in [../DEPENDENCIES.md](../DEPENDENCIES.md).

Download the MobileSAM and Point-MAE weights from their official sources as
listed in the root [README Downloads](../README.md#downloads) section, then
validate the complete feature-construction input set:

```bash
./scripts/check_downloaded_assets.sh --with-dumps
```

## 3. Construct features

Run the three preparation stages once for train dumps and once for test dumps.
Use the reported shell boundaries `(0, 5, 15, 25, 40)` mm, per-shell budgets
`(64, 128, 128, 192)`, and 512 object points. The compact canonical invocation
is in the root [Quick Start](../README.md#quick-start).

Store object clouds in a sidecar tree and precompute the frozen Point-MAE
features:

```bash
grare-precompute-object \
  --archive-root "$GRARE_DATA_ROOT/relabeled/graspnet_baseline/realsense/local_cloud/train" \
  --object-cloud-root "$GRARE_DATA_ROOT/relabeled/graspnet_baseline/realsense/object_cloud/train" \
  --output-root "$GRARE_DATA_ROOT/relabeled/graspnet_baseline/realsense/object_pooled/train" \
  --pmae-ckpt "$GRARE_POINT_MAE_CKPT" \
  --object-cloud-points 512 --device cuda
```

Repeat this command for the test split. For GN-Kinect and EG-Kinect, create
the mmap-packed training tree with `grare-pack` as documented in the root
README.

## 4. Train, re-rank, and evaluate

Set the roots referenced by the configurations:

```bash
export GRASPNET_ROOT=/path/to/graspnet
export GRARE_DATA_ROOT=/path/to/grare-data
export GRARE_OUTPUT_ROOT=/path/to/grare-output
export GRARE_POINT_MAE_CKPT=/path/to/point_mae_pretrain.pth
```

Run one setting at a time. Each takes hours, so stopping after training gives
a checkpoint gate before predictions are written:

```bash
grare-run --config configs/gn_realsense.yaml --stop-after train
```

`grare-rerank` consumes the relabeled `.npz` archives emitted by
`grare-prepare` and exports evaluator-compatible `.npy` files. It does not
take raw detector `.npy` dumps directly.

The official test evaluation is valid only when all 90 test scenes and all 256
frames per scene have been re-ranked. `grare-evaluate` checks this requirement
before calling the official GraspNet evaluator. A one-frame or one-scene run
is useful for debugging but must not be reported as an AP.

## Statistical comparison

For paired detector-versus-GraRe scene comparisons, use the raw tensors from
two complete official evaluations:

```bash
python scripts/paired_scene_bootstrap.py \
  --baseline /path/to/detector/per_scene_raw.npy \
  --treatment /path/to/grare/per_scene_raw.npy \
  --output paired_bootstrap.json
```

The script reports paired scene-bootstrap 95% intervals, mean AP changes, and
the number of scenes with positive changes for the overall, Seen, Similar, and
Novel splits.

## 5. Verification levels

| Level | Command | What it verifies |
| --- | --- | --- |
| Unit | `python -m pytest -q` | model, archive, training, and command contracts |
| Smoke | `grare-smoke` | synthetic train, checkpoint reload, and re-ranking |
| Configuration | `grare-run --config ... --dry-run` | a resolved command graph |
| Full | `grare-run --config ...` | data preparation outputs, training, reranking, and official AP |

Use the complete official test dump for any numerical comparison with the
paper values in [RESULTS.md](RESULTS.md).

## 6. Runtime tuning and recovery

The normal `grare-dump` and `grare-prepare` commands use the recommended
defaults. The environment file sets one thread per BLAS process unless you have
already chosen another value. Labels default to up to 20 CPU workers; the SAM
object stage selects a safe worker count from the number of visible GPUs and
their VRAM. Point-MAE precomputation uses a memory-safe 2,048-candidate GPU
batch. Worker choices can be overridden with `--num-workers` for a measured
host-specific experiment.

All dump and feature-construction stages skip completed files. Re-run the same
command after an interruption. Use `--limit 1` on `grare-prepare` or
`--max-batches 1` on `grare-dump` for a small debugging run.

GN and SBG can partition a dump across GPUs; each process receives a disjoint
shard. EconomicGrasp does not support dump sharding:

```bash
CUDA_VISIBLE_DEVICES=0 grare-dump --detector graspnet_baseline --camera realsense \
  --split train --index-shard-count 2 --index-shard-id 0 &
CUDA_VISIBLE_DEVICES=1 grare-dump --detector graspnet_baseline --camera realsense \
  --split train --index-shard-count 2 --index-shard-id 1 &
wait
```

Use `--deterministic` only when fixed-machine dump reproducibility is required;
it is slower. Fresh detector dumps on another GPU, CUDA version, or driver can
vary slightly in floating-point confidence values.

For the GN-Kinect and EG-Kinect configurations, build the packed training tree
after object-feature precomputation:

```bash
grare-pack \
  --input-root "$GRARE_DATA_ROOT/relabeled/graspnet_baseline/kinect/local_cloud/train" \
  --object-pooled-root "$GRARE_DATA_ROOT/relabeled/graspnet_baseline/kinect/object_pooled/train" \
  --output-root "$GRARE_DATA_ROOT/packed/graspnet_baseline/kinect/train" \
  --archive-manifest auto --require-archive-manifest --require-object-pooled
```

Run `grare-run --config ... --dry-run` to inspect a resolved command graph.
For a detector-only AP baseline, run the rerank/evaluation stages with
`--set rerank.lambda=0.0`; the artifacts use separate
`*_detector_baseline` directories. Official AP requires all 90 test scenes and
all 256 frames per scene.
