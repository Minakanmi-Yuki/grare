# External Dependencies

GraRe consumes candidate dumps from frozen detectors. It does not vendor or
modify the detector implementations. The following public projects are the
source dependencies used by the paper:

| Component | Public project | Role |
| --- | --- | --- |
| GraspNet-1Billion and evaluator | https://github.com/graspnet/graspnetAPI | Dataset layout, analytical labels, official AP |
| GraspNet-Baseline | https://github.com/graspnet/graspnet-baseline | Frozen GN candidate generator |
| Scale-Balanced-Grasp | https://github.com/mahaoxiang822/Scale-Balanced-Grasp | Frozen SBG candidate generator |
| SBG tolerance labels | https://github.com/mahaoxiang822/Scale-Balanced-Grasp | Upstream SBG training labels; generate from GraspNet or obtain through the upstream release |
| EconomicGrasp | https://github.com/iSEE-Laboratory/EconomicGrasp | Frozen EG candidate generator |
| MobileSAM | https://github.com/ChaoningZhang/MobileSAM | Prompted visible-object masks |
| Point-MAE | https://github.com/Pang-Yatian/Point-MAE | Frozen object-context encoder |

Review each upstream project and dataset license before redistribution. The
MIT license in this package applies only to GraRe source code.
