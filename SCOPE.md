# Publication Scope

This repository contains the core GraRe method and the five benchmark settings
reported in the paper:

- GraspNet-Baseline: RealSense and Kinect
- Scale-Balanced-Grasp: RealSense
- EconomicGrasp: RealSense and Kinect

Included functionality covers feature construction, analytical supervision,
training, validation-based checkpoint selection, candidate re-ranking, and
official GraspNet AP evaluation. See [docs/REPRODUCTION.md](docs/REPRODUCTION.md)
for the supported offline workflow.

The repository intentionally excludes detector source code, detector weights,
GraRe checkpoints, datasets, robot-control software, raw robot recordings,
and exploratory experiment utilities. These omissions avoid redistributing
third-party artifacts and keep the public surface focused on the paper's
offline GraspNet claims. Real-robot outcomes are presented as a result
showcase, not as a deployable robot stack.
