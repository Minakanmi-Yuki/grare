from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grare.candidates import (
    DetectorPrediction,
    PredictionKey,
    iter_prediction_files,
)
from grare.detectors import (
    EconomicGraspConfig,
    EconomicGraspWrapper,
    GraspNetBaselineConfig,
    GraspNetBaselineWrapper,
    HGGDConfig,
    HGGDWrapper,
    RNGNetConfig,
    RNGNetWrapper,
    ScaleBalancedGraspConfig,
    ScaleBalancedGraspWrapper,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

_CASES = [
    (
        "graspnet_baseline",
        GraspNetBaselineWrapper,
        GraspNetBaselineConfig,
        "scripts/detectors/run_graspnet_baseline_split.py",
    ),
    (
        "scale_balanced_grasp",
        ScaleBalancedGraspWrapper,
        ScaleBalancedGraspConfig,
        "scripts/detectors/run_scale_balanced_grasp_split.py",
    ),
    (
        "economicgrasp",
        EconomicGraspWrapper,
        EconomicGraspConfig,
        "scripts/detectors/run_economicgrasp_split.py",
    ),
    (
        "hggd",
        HGGDWrapper,
        HGGDConfig,
        "scripts/detectors/run_hggd_split.py",
    ),
    (
        "rngnet",
        RNGNetWrapper,
        RNGNetConfig,
        "scripts/detectors/run_rngnet_split.py",
    ),
]


@pytest.mark.parametrize("name,wrapper_cls,config_cls,adapter", _CASES)
def test_wrapper_targets_an_adapter_that_exists(
    name: str, wrapper_cls, config_cls, adapter: str, tmp_path: Path
) -> None:
    config = config_cls(
        dataset_root="/data/graspnet",
        checkpoint_path="/ckpt/model.tar",
        camera="realsense",
        split="test",
    )
    wrapper = wrapper_cls(config, repo_root=REPO_ROOT, python_bin="python")
    commands = wrapper.build_inference_commands(tmp_path)

    assert len(commands) == 1
    command = [str(part) for part in commands[0]]
    assert command[1] == adapter
    # The adapter path is resolved relative to repo_root at run time.
    assert (REPO_ROOT / adapter).is_file(), f"missing adapter: {adapter}"
    assert "--dataset_root" in command
    assert "--checkpoint_path" in command
    assert str(tmp_path / name / "test") in command
    if name in {"hggd", "rngnet"}:
        assert command[command.index("--data_workers") + 1] == "8"
        assert command[command.index("--prefetch-factor") + 1] == "4"


@pytest.mark.parametrize("name,wrapper_cls,config_cls,adapter", _CASES)
def test_split_and_camera_reach_the_adapter(
    name: str, wrapper_cls, config_cls, adapter: str, tmp_path: Path
) -> None:
    config = config_cls(
        dataset_root="/data/graspnet",
        checkpoint_path="/ckpt/model.tar",
        camera="kinect",
        split="train",
    )
    wrapper = wrapper_cls(config, repo_root=REPO_ROOT, python_bin="python")
    command = [str(part) for part in wrapper.build_inference_commands(tmp_path)[0]]
    assert command[command.index("--camera") + 1] == "kinect"
    assert command[command.index("--split") + 1] == "train"
    assert str(tmp_path / name / "train") in command


def test_dump_configs_carry_the_recommended_throughput_defaults() -> None:
    for config_cls, worker_field in (
        (GraspNetBaselineConfig, "num_workers"),
        (ScaleBalancedGraspConfig, "data_workers"),
        (EconomicGraspConfig, "data_workers"),
    ):
        config = config_cls(dataset_root="/data", checkpoint_path="/ckpt")
        assert config.batch_size == 24
        assert getattr(config, worker_field) == 8
        assert config.prefetch_factor == 4
        assert config.pin_memory is True
        assert config.skip_existing is True
        assert config.postprocess_workers == 32


def test_adapters_resolve_project_root_to_repository_root() -> None:
    """The adapters live one level below scripts/, so parents[2] is the root."""
    for _, _, _, adapter in _CASES:
        path = REPO_ROOT / adapter
        assert path.resolve().parents[2] == REPO_ROOT
        source = path.read_text(encoding="utf-8")
        assert "PROJECT_ROOT = Path(__file__).resolve().parents[2]" in source


def test_save_eval_npy_uses_the_graspnet_dump_layout(tmp_path: Path) -> None:
    array = np.zeros((3, 17), dtype=np.float32)
    array[:, 0] = [0.3, 0.9, 0.6]
    prediction = DetectorPrediction(
        key=PredictionKey("graspnet_baseline", "graspnet", "test", 100, 7, "realsense"),
        grasp_group_array=array,
    )
    saved = prediction.save_eval_npy(tmp_path)
    assert saved == tmp_path / "scene_0100" / "realsense" / "0007.npy"
    assert np.array_equal(np.load(saved), array)


def test_iter_prediction_files_accepts_camera_and_flat_layouts(tmp_path: Path) -> None:
    camera_dir = tmp_path / "scene_0100" / "realsense"
    camera_dir.mkdir(parents=True)
    np.save(camera_dir / "0001.npy", np.zeros((1, 17), dtype=np.float32))
    np.save(camera_dir / "0000.npy", np.zeros((1, 17), dtype=np.float32))

    flat_dir = tmp_path / "scene_0101"
    flat_dir.mkdir(parents=True)
    np.save(flat_dir / "0000.npy", np.zeros((1, 17), dtype=np.float32))

    found = iter_prediction_files(tmp_path, "realsense")
    assert [path.name for path in found] == ["0000.npy", "0001.npy", "0000.npy"]
    assert found[0].parent == camera_dir
    assert found[2].parent == flat_dir
