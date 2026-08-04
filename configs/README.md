# Configurations

Each YAML file fixes a batch size of 2048 and a score-fusion weight of
`lambda = 1.0`. The first five are the reported GraspNet-1Billion settings;
the additional upstream-detector configurations use the same GraRe workflow.

| Configuration | Frozen detector | Camera | Packed training features |
| --- | --- | --- | --- |
| `gn_realsense.yaml` | GraspNet-Baseline | RealSense | no |
| `gn_kinect.yaml` | GraspNet-Baseline | Kinect | yes |
| `sbg_realsense.yaml` | Scale-Balanced-Grasp | RealSense | no |
| `eg_realsense.yaml` | EconomicGrasp | RealSense | no |
| `eg_kinect.yaml` | EconomicGrasp | Kinect | yes |
| `hggd_realsense.yaml` | HGGD | RealSense | no |
| `hggd_kinect.yaml` | HGGD | Kinect | no |
| `rngnet_realsense.yaml` | RNGNet | RealSense | no |
| `rngnet_kinect.yaml` | RNGNet | Kinect | no |
| `generalizing_grasp_realsense.yaml` | Generalizing-Grasp | RealSense | no |

Run one configuration with:

```bash
grare-run --config configs/gn_realsense.yaml
```

To inspect the fully resolved commands without accessing data or GPUs, append
`--dry-run`. See [../docs/REPRODUCTION.md](../docs/REPRODUCTION.md) for the
required candidate-dump and feature layout.

Only the first five configurations have released GraRe results and assets.
The other settings are detector integrations and must be trained and evaluated
before reporting results.
