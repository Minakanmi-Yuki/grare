# Paper Configurations

The five YAML files in this directory are the configurations used for the
main GraspNet-1Billion results. Each configuration fixes a batch size of 2048
and a score-fusion weight of `lambda = 1.0`.

| Configuration | Frozen detector | Camera | Packed training features |
| --- | --- | --- | --- |
| `gn_realsense.yaml` | GraspNet-Baseline | RealSense | no |
| `gn_kinect.yaml` | GraspNet-Baseline | Kinect | yes |
| `sbg_realsense.yaml` | Scale-Balanced-Grasp | RealSense | no |
| `eg_realsense.yaml` | EconomicGrasp | RealSense | no |
| `eg_kinect.yaml` | EconomicGrasp | Kinect | yes |

Run one configuration with:

```bash
grare-run --config configs/gn_realsense.yaml
```

To inspect the fully resolved commands without accessing data or GPUs, append
`--dry-run`. See [../docs/REPRODUCTION.md](../docs/REPRODUCTION.md) for the
required candidate-dump and feature layout.

## Ablations

Only the five main-result configurations are treated as release entry points.
The paper's ablations are documented in [../docs/RESULTS.md](../docs/RESULTS.md).
Exploratory sweeps and non-paper configuration variants are intentionally kept
out of this repository.
