# Data Contract

## Detector Dumps

GraRe starts from frozen-detector predictions in the standard GraspNet
`GraspGroup` array format. Each frame is a float32 `.npy` array with shape
`(K, 17)` and the following columns:

| Column | Meaning |
| --- | --- |
| 0 | detector confidence |
| 1 | gripper width |
| 2 | gripper height |
| 3 | grasp depth |
| 4:13 | flattened 3 x 3 rotation matrix |
| 13:16 | 3D translation |
| 16 | object id, when supplied by the detector |

The expected directory layout is:

```text
<dump-root>/
  scene_0000/
    realsense/
      0000.npy
```

The code also accepts `scene_0000/0000.npy`. The paper protocol retains every
candidate emitted by the detector. Candidate identities, widths, poses, and
set sizes are unchanged by GraRe; only the order and score column are updated.

## Relabeled Archives

`grare-prepare` writes one `.npz` per frame. Required arrays are:

| Key | Shape | Description |
| --- | --- | --- |
| `grasp_group_array` | `(K, 17)` | unchanged detector candidates |
| `base_scores` | `(K,)` | detector confidence |
| `grasp_widths` | `(K,)` | gripper widths |
| `grasp_poses` | `(K, 4, 4)` | homogeneous poses |
| `local_cloud` | `(K, 512, 3)` | shell-wise sampled local points |
| `cloud_mask` | `(K, 512)` | valid local-point mask |
| `mu_min` | `(K,)` | analytical minimum friction |
| `is_collision` | `(K,)` | collision targets |
| `is_empty` | `(K,)` | empty-grasp targets |
| `object_assignments` | `(K,)` | object-class targets or `-1` |

Object clouds and pooled Point-MAE features may be stored in sidecar trees
with the same relative frame paths. This is the recommended layout because it
keeps raw object clouds out of the training read path.
