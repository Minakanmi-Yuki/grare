# Release Scope

GraRe is released as an offline re-ranking framework for frozen 6-DoF grasp
detectors. The supported reproducibility target is the GraspNet-1Billion
workflow described in [REPRODUCTION.md](REPRODUCTION.md).

## Included

- GraRe feature construction, training, checkpoint loading, re-ranking, and
  official GraspNet evaluation code.
- Five configurations for GraspNet-Baseline, Scale-Balanced-Grasp, and
  EconomicGrasp.
- Self-contained smoke tests and unit tests.
- Documentation of expected benchmark and real-robot results.

## Not Included

- GraspNet-1Billion data, detector outputs, detector repositories, public
  backbone weights, or trained GraRe checkpoints.
- Raw predictions, logs, experiment queues, hyperparameter-search records, or
  intermediate visualizations.
- Robot-control, calibration, collision-filtering, motion-planning, or
  hardware-integration code.

Obtain third-party datasets, detector implementations, and backbone weights
from their original public sources and comply with their respective licenses.
The repository's MIT license covers only GraRe code.
