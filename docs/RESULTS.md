# Reported Offline Results

This page records the main GraspNet-1Billion results reported by GraRe. Values
are Average AP (%); each entry is `RealSense / Kinect` when both cameras are
reported. A comparison is meaningful only when the complete official test
protocol is used (90 scenes, 256 frames per scene, and the frozen detector's
unchanged candidate sets).

| Frozen detector | Detector | GraRe | Gain |
| --- | ---: | ---: | ---: |
| GraspNet-Baseline | 35.85 / 30.59 | 49.45 / 38.79 | +13.60 / +8.20 |
| Scale-Balanced-Grasp | 47.66 / — | 52.97 / — | +5.31 / — |
| EconomicGrasp | 52.02 / 45.26 | 55.95 / 49.98 | +3.93 / +4.72 |

## Key ablations on the RealSense test set

The full GraRe setting reaches 49.45, 52.97, and 55.95 Average AP for GN,
SBG, and EG, respectively. The paper reports the following reductions from
that full setting:

| Variant | GN | SBG | EG |
| --- | ---: | ---: | ---: |
| Concatenation fusion | -0.82 | -0.26 | -0.12 |
| Without candidate features | -1.79 | -2.16 | -5.46 |
| Without local features | -4.42 | -2.43 | -1.73 |
| Without object context | -0.74 | -0.43 | -0.32 |
| Without shell-wise FPS | -1.19 | -0.43 | -0.54 |
| Without auxiliary losses | -0.43 | -0.78 | -0.69 |
| Binary quality target | -0.51 | -0.64 | -0.87 |
| Randomized targets | -12.41 | -12.12 | -16.19 |

The reported confidence intervals use paired scene bootstrap resampling. The
exploratory sweep machinery is not part of this release; its purpose is to
support the paper evidence rather than define an additional reproduction API.
