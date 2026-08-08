# GraRe: Grasp Candidate Re-Ranking for Frozen 6-DoF Grasp Detectors

<p align="center">
  Jibao Yuan · Yuhui Zhao · Yinzhen Lv · Chao Xu · Shun Li · Chenxi Deng · Shaofei Chen*
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2608.00946">Paper: arXiv:2608.00946</a>
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

Clone GraRe, create conda environment, and install dependencies:
```bash
git clone https://github.com/Minakanmi-Yuki/grare.git
cd grare
conda create -n grare python=3.12 -y
conda activate grare
python -m pip install --upgrade pip
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.12.1+cu130 torchvision==0.27.1+cu130
python -m pip install -e '.[test,detectors]'
```

Verify the installation:

```bash
grare-smoke
python -m pytest -q
```

Feature preparation and official AP evaluation additionally require the
official [GraspNet API](https://github.com/graspnet/graspnetAPI) and
[MobileSAM](https://github.com/ChaoningZhang/MobileSAM): 

```bash
git clone https://github.com/graspnet/graspnetAPI ../graspnetAPI
python -m pip install -e ../graspnetAPI

git clone https://github.com/ChaoningZhang/MobileSAM.git ../MobileSAM
python -m pip install -e ../MobileSAM

python -m pip install grasp_nms
python -m pip install -e '.[prepare]'
```

To generate frozen-detector grasp candidates, install the shared detector
runtime and build dependencies. This requires a CUDA toolkit with `nvcc` that
is compatible with the PyTorch build above:

```bash
conda install -y -c anaconda openblas-devel
conda install -y -c conda-forge ninja
python -m pip install tensorboard open3d Pillow
```

Clone the following detector repositories into `external/`:
```bash
mkdir -p external
git clone https://github.com/graspnet/graspnet-baseline external/graspnet-baseline
git clone https://github.com/mahaoxiang822/Scale-Balanced-Grasp external/Scale-Balanced-Grasp
git clone https://github.com/iSEE-Laboratory/EconomicGrasp external/EconomicGrasp
git clone https://github.com/THU-VCLab/HGGD external/HGGD
git clone https://github.com/THU-VCLab/RNGNet external/RNGNet
```

Build the CUDA extensions and verify them:

```bash
./scripts/build_detector_extensions.sh
./scripts/verify_detector_extensions.sh
```

## Downloads

GraRe requires the GraspNet-1Billion dataset, detector checkpoints,
and the MobileSAM and Point-MAE backbone weights. Set the asset directory,
then initialize its paths:

```bash
export GRARE_ASSET_WORKSPACE=/path/to/grare-assets
./scripts/prepare_data_assets.sh --workspace "$GRARE_ASSET_WORKSPACE"
source "$GRARE_ASSET_WORKSPACE/grare_paths.env"
```

Download assets from official source: 

| Asset | Official source | Place at |
| --- | --- | --- |
| GraspNet-1Billion | [graspnet.net/datasets.html](https://graspnet.net/datasets.html) | `$GRASPNET_ROOT` |
| MobileSAM weight | [MobileSAM repository](https://github.com/ChaoningZhang/MobileSAM), `weights/mobile_sam.pt` | `$GRARE_SAM_CKPT` |
| Point-MAE weight | [Point-MAE release](https://github.com/Pang-Yatian/Point-MAE/releases/tag/main), `pretrain.pth` | `$GRARE_POINT_MAE_CKPT` |
| GN checkpoints | [GraspNet-Baseline](https://github.com/graspnet/graspnet-baseline#training-and-testing), `checkpoint-rs.tar` and `checkpoint-kn.tar` | `$GRARE_DETECTOR_CKPT_ROOT/graspnet_baseline/` |
| SBG checkpoint | [Scale-Balanced-Grasp](https://github.com/mahaoxiang822/Scale-Balanced-Grasp#test), the `log_full_model/checkpoint.tar` in its Drive folder | `$GRARE_DETECTOR_CKPT_ROOT/scale_balanced_grasp/log_full_model/` |
| EG checkpoints | [EconomicGrasp v1 release](https://github.com/iSEE-Laboratory/EconomicGrasp/releases/tag/v1), `economicgrasp_realsense.tar` and `economicgrasp_kinect.tar` | `$GRARE_DETECTOR_CKPT_ROOT/economicgrasp/` |
| HGGD checkpoints | [HGGD Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/e3edfc2c8b114513b7eb/), `HGGD_realsense_checkpoint` and `HGGD_kinect_checkpoint` | `$GRARE_DETECTOR_CKPT_ROOT/hggd/` |
| RNGNet checkpoints | bundled as `realsense.pth` and `kinect.pth` in the official [RNGNet repository](https://github.com/THU-VCLab/RNGNet) | `$GRARE_DETECTOR_CKPT_ROOT/rngnet/` |

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
│   ├── economicgrasp/
│   │   ├── economicgrasp_realsense.tar             EG RealSense
│   │   └── economicgrasp_kinect.tar                EG Kinect
│   ├── hggd/
│   │   ├── HGGD_realsense_checkpoint               HGGD RealSense
│   │   └── HGGD_kinect_checkpoint                  HGGD Kinect
│   ├── rngnet/
│   │   ├── realsense.pth                           RNGNet RealSense
│   │   └── kinect.pth                              RNGNet Kinect
└── backbones/                                     frozen feature backbones
    ├── mobile_sam.pt
    └── point_mae_pretrain.pth
```

Verify the assets:

```bash
./scripts/check_downloaded_assets.sh --detector graspnet_baseline --camera realsense
```

## Prepare
To generate the grasp candidates and verify them:
```bash
source "$GRARE_ASSET_WORKSPACE/grare_paths.env"
DETECTOR=graspnet_baseline
CAMERA=realsense

grare-dump --detector "$DETECTOR" --camera "$CAMERA"
./scripts/check_downloaded_assets.sh --detector "$DETECTOR" --camera "$CAMERA" --with-dumps
```

Then prepare the training features and verify them:
```bash
grare-feature --detector "$DETECTOR" --camera "$CAMERA"
./scripts/check_downloaded_assets.sh --detector "$DETECTOR" --camera "$CAMERA" --with-features
```
The resulting `local_cloud`, `object_cloud`, and `object_pooled` directories are the inputs for training.

Published dumps and prepared features can instead be downloaded directly into
the same locations. Install the optional Hub client once, then use the same
detector and camera arguments:

```bash
python -m pip install -e '.[hub]'
grare-fetch dumps --detector "$DETECTOR" --camera "$CAMERA"
grare-fetch features --detector "$DETECTOR" --camera "$CAMERA"
```

## Train and Evaluate
Select the configuration matching the prepared detector and camera:

| Config | Frozen detector | Camera |
| --- | --- | --- |
| `configs/gn_realsense.yaml` | GraspNet-Baseline | RealSense |
| `configs/gn_kinect.yaml` | GraspNet-Baseline | Kinect |
| `configs/sbg_realsense.yaml` | Scale-Balanced-Grasp | RealSense |
| `configs/eg_realsense.yaml` | EconomicGrasp | RealSense |
| `configs/eg_kinect.yaml` | EconomicGrasp | Kinect |
| `configs/hggd_realsense.yaml` | HGGD | RealSense |
| `configs/hggd_kinect.yaml` | HGGD | Kinect |
| `configs/rngnet_realsense.yaml` | RNGNet | RealSense |
| `configs/rngnet_kinect.yaml` | RNGNet | Kinect |

For example:
```bash
grare-run --config configs/gn_realsense.yaml
```
This trains, re-ranks, and runs the official evaluation in order. Outputs are
written to `$GRARE_OUTPUT_ROOT`.

To use the matching published GraRe checkpoint without retraining:

```bash
grare-fetch checkpoint --detector graspnet_baseline --camera realsense
```

## Demo

The demo follows the single-frame RGB-D workflow of the
[GraspNet-Baseline demo](https://github.com/graspnet/graspnet-baseline/blob/main/demo.py).

<table align="center">
  <tbody>
    <tr>
      <td align="center"><img src="assets/demo/demo_detector.png" alt="GraspNet-Baseline detector preview" width="280"></td>
      <td align="center"><img src="assets/demo/demo_grare.png" alt="GraspNet-Baseline GraRe preview" width="280"></td>
    </tr>
    <tr>
      <th align="center">Detector</th>
      <th align="center">GraRe</th>
    </tr>
  </tbody>
</table>

After completing the corresponding workflow, run one GraspNet-1Billion frame
(the detector can be any supported selection with a matching configuration):
```bash
source "$GRARE_ASSET_WORKSPACE/grare_paths.env"
grare-demo --detector graspnet_baseline --camera realsense --scene 0100 --frame 0000
```

## Results
The following offline GraspNet-1Billion results use the official evaluation protocol.
Each metric is reported as `Detector / GraRe`.

| Detector | Camera | λ | Seen | Similar | Novel | Average |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GraspNet-Baseline | RealSense | 1.0 | 47.83 / **64.16** | 42.79 / **58.47** | 16.94 / **25.31** | 35.85 / **49.32** |
| Scale-Balanced-Grasp | RealSense | 0.9 | 62.27 / **68.67** | 56.92 / **62.62** | 23.80 / **27.30** | 47.66 / **52.86** |
| EconomicGrasp | RealSense | 0.9 | 69.30 / **75.18** | 61.50 / **65.05** | 25.28 / **28.14** | 52.02 / **56.12** |
| HGGD | RealSense | 0.8 | 63.37 / **67.38** | 52.28 / **55.26** | 23.83 / **25.72** | 46.49 / **49.45** |
| RNGNet | RealSense | 0.5 | 75.32 / **77.73** | 66.95 / **69.52** | 32.03 / **32.50** | 58.10 / **59.92** |
| GraspNet-Baseline | Kinect | 0.9 | 41.97 / **53.92** | 37.56 / **46.32** | 12.24 / **15.95** | 30.59 / **38.73** |
| EconomicGrasp | Kinect | 0.9 | 63.75 / **69.78** | 52.43 / **57.23** | 19.61 / **22.41** | 45.26 / **49.81** |
| HGGD | Kinect | 0.8 | 60.43 / **64.95** | 46.74 / **50.03** | 19.23 / **19.85** | 42.13 / **44.94** |
| RNGNet | Kinect | 0.4 | 73.50 / **75.37** | 59.20 / **60.66** | 26.44 / **26.46** | 53.05 / **54.16** |

![Seen, Similar, and Novel AP across score-fusion lambda values](assets/results/lambda_sweep_2x3.png)

## Computational Cost

Training was performed on a system with one NVIDIA GeForce RTX 5090 GPU
(32 GiB VRAM) and an Intel Xeon Platinum 8470Q CPU.


| Frozen detector | Ranking | Camera | Time | VRAM |
| --- | --- | --- | ---: | ---: |
| GraspNet-Baseline | GraRe | RealSense | 1 h 12 min | 12.57 GiB |
| GraspNet-Baseline | GraRe | Kinect | 1 h 17 min | 5.81 GiB |
| Scale-Balanced-Grasp | GraRe | RealSense | 58 min | 12.74 GiB |
| EconomicGrasp | GraRe | RealSense | 4 h 34 min | 5.81 GiB |
| EconomicGrasp | GraRe | Kinect | 7 h 44 min | 5.81 GiB |
| HGGD | GraRe | RealSense | 33 min | 9.22 GiB |
| HGGD | GraRe | Kinect | 1 h 12 min | 9.22 GiB |
| RNGNet | GraRe | RealSense | 40 min | 19.45 GiB |
| RNGNet | GraRe | Kinect | 1 h 9 min | 5.81 GiB |

## Citation

```bibtex
@article{yuan2026grare,
  title   = {GraRe: Grasp Candidate Re-Ranking for Frozen 6-DoF Grasp Detectors},
  author  = {Yuan, Jibao and Zhao, Yuhui and Lv, Yinzhen and Xu, Chao and
             Li, Shun and Deng, Chenxi and Chen, Shaofei},
  journal = {arXiv preprint arXiv:2608.00946},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.00946}
}
```
