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

Install the public GraspNet API and its `grasp_nms` extension separately. The
official projects and licenses are listed in [../DEPENDENCIES.md](../DEPENDENCIES.md).

## 2. Obtain inputs

Create a local asset workspace and download the two public backbone
checkpoints:

```bash
./scripts/prepare_data_assets.sh \
  --workspace "$PWD/grare-assets" \
  --download-backbones
```

Download GraspNet-1Billion from its original public source, subject to its
terms, and place it under `grare-assets/graspnet/` (or retain it elsewhere).
For a full reconstruction from raw data, also obtain the five detector-setting
checkpoints (`gn_realsense`, `gn_kinect`, `sbg_realsense`, `eg_realsense`, and
`eg_kinect`) from the three original detector projects. Use those checkpoints
to run each frozen detector and retain its raw `(K, 17)` candidate dump for
every frame under `grare-assets/detector_dumps/` (or retain the dumps
elsewhere). GraRe never changes candidate identities, poses, widths, or
candidate-set size; once the dumps exist, detector checkpoints are no longer
needed by GraRe. The exact dump contract is in
[../DATA_FORMAT.md](../DATA_FORMAT.md). The corresponding upstream projects
are listed in [../DEPENDENCIES.md](../DEPENDENCIES.md).

The original GraspNet-1Billion archive does not contain the tolerance labels
used to train Scale-Balanced-Grasp. They are not read by GraRe or by
pretrained-detector candidate inference. Only if retraining SBG from scratch,
generate them from the upstream repository:

```bash
cd /path/to/Scale-Balanced-Grasp/dataset
python generate_tolerance_label.py \
  --dataset_root "$GRASPNET_ROOT" \
  --num_workers <N>
```

The generator writes `dataset/tolerance/` in that upstream repository; the
published tolerance archive is an equivalent upstream option.

When both assets are available, validate the workspace and load the generated
environment variables:

```bash
./scripts/prepare_data_assets.sh \
  --workspace "$PWD/grare-assets" \
  --graspnet-root /path/to/graspnet \
  --detector-dumps /path/to/frozen-detector-dumps \
  --setting gn_realsense \
  --check
source "$PWD/grare-assets/grare_paths.env"
```

Use `--all-paper-settings --require-detector-checkpoints --check` to audit
all five pretrained-detector candidate-generation paths before generating the
paper dumps. This check intentionally excludes SBG tolerance labels because
they are needed only for upstream SBG training.

## 3. Construct features

Run `grare-prepare` once for train dumps and once for test dumps. Use the
paper shell boundaries `(0, 5, 15, 25, 40)` mm, per-shell budgets
`(64, 128, 128, 192)`, and 512 object points. The command in the root
[README](../README.md#input-contract) is the canonical invocation.

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

Set the roots referenced by the paper configurations:

```bash
export GRASPNET_ROOT=/path/to/graspnet
export GRARE_DATA_ROOT=/path/to/grare-data
export GRARE_OUTPUT_ROOT=/path/to/grare-output
export GRARE_POINT_MAE_CKPT=/path/to/point_mae_pretrain.pth
```

Run a single setting, or invoke all five sequentially:

```bash
grare-run --config configs/gn_realsense.yaml
./scripts/run_paper_configs.sh --stop-after train
```

`grare-rerank` consumes the relabeled `.npz` archives emitted by
`grare-prepare` and exports evaluator-compatible `.npy` files. It does not
take raw detector `.npy` dumps directly.

The official test evaluation is valid only when all 90 test scenes and all 256
frames per scene have been re-ranked. `grare-evaluate` checks this requirement
before calling the official GraspNet evaluator. A one-frame or one-scene run
is useful for debugging but must not be reported as a paper AP.

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
| Configuration | `./scripts/run_paper_configs.sh --dry-run` | all five paper command graphs |
| Full | `grare-run --config ...` | data preparation outputs, training, reranking, and official AP |

Use the complete official test dump for any numerical comparison with the
paper values in [RESULTS.md](RESULTS.md).
