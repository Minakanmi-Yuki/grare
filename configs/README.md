# Configurations

The five YAML files in this directory are the reported GraspNet-1Billion
settings. Each fixes a batch size of 2048 and a score-fusion weight of
`lambda = 1.0`.

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

These five settings are the only release entry points. The reported ablations
are implemented by the same GraRe modules and their results are listed in
[../docs/RESULTS.md](../docs/RESULTS.md); the exploratory sweep configurations
behind them are not part of this repository.
