# Real-Robot Result Showcase

The paper evaluates the ordering produced by GraRe on a UR3 robot with a
RealSense D435 camera and a Robotiq 2F-85 gripper across ten mixed-object
cluttered scenes. Within each detector comparison, the hardware, detector,
collision filtering, planning, and stopping condition are unchanged; only the
candidate order differs.

This repository presents these results as physical-evaluation evidence. It
does **not** provide robot-control, calibration, collision-filtering,
motion-planning, or hardware-integration code, and therefore does not claim a
turnkey real-robot reproduction.

| Detector | Grasp success rate, detector / GraRe | Completion rate, detector / GraRe | Maximum consecutive failures, detector / GraRe | Mean latency (s), detector / GraRe |
| --- | ---: | ---: | ---: | ---: |
| GraspNet-Baseline | 73.4 / 88.9 | 10 / 100 | 4 / 2 | 1.06 / 1.45 |
| Scale-Balanced-Grasp | 70.4 / 87.2 | 30 / 90 | 6 / 2 | 0.95 / 1.21 |
| EconomicGrasp | 68.2 / 88.5 | 70 / 100 | 4 / 2 | 0.76 / 0.97 |

Across the 30 detector-order versus GraRe-order evaluations, the detector
order cleared 11 scenes and the GraRe order cleared 29. These values are a
result showcase, separate from the supported offline GraspNet reproduction in
[REPRODUCTION.md](REPRODUCTION.md).
