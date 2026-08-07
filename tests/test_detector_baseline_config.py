from __future__ import annotations

import pytest

from grare.cli.run import _eval_command, _rerank_command
from grare.config import validate_config


def _config(rerank_lambda: float) -> dict:
    return {
        "name": "gn_realsense",
        "detector": "graspnet_baseline",
        "camera": "realsense",
        "paths": {
            "graspnet_root": "/data/graspnet",
            "local_cloud_test": "/data/local/test",
            "object_pooled_test": "/data/object/test",
            "checkpoint_dir": "/output/checkpoints/gn_realsense",
            "rerank_dir": "/output/predictions/gn_realsense",
            "eval_dir": "/output/evaluation/gn_realsense",
        },
        "train": {"batch_size": 2048},
        "rerank": {"lambda": rerank_lambda, "device": "cuda", "num_workers": 1},
        "eval": {"proc": 16},
    }


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_grare_ranking_lambda_is_accepted() -> None:
    validate_config(_config(1.0))


def test_detector_baseline_lambda_is_accepted() -> None:
    validate_config(_config(0.0))


@pytest.mark.parametrize("lambda_value", [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
def test_intermediate_lambda_is_accepted(lambda_value: float) -> None:
    validate_config(_config(lambda_value))


@pytest.mark.parametrize("bad_lambda", [-1.0, 1.5, float("nan"), float("inf")])
def test_out_of_range_lambda_is_rejected(bad_lambda: float) -> None:
    with pytest.raises(ValueError, match="rerank.lambda"):
        validate_config(_config(bad_lambda))


def test_grare_run_writes_to_default_directories() -> None:
    config = _config(1.0)
    rerank = _rerank_command(config)
    evaluate = _eval_command(config)
    assert _value_after(rerank, "--output-root") == "/output/predictions/gn_realsense"
    assert _value_after(evaluate, "--dump-folder") == "/output/predictions/gn_realsense"
    assert _value_after(evaluate, "--save-raw") == (
        "/output/evaluation/gn_realsense/per_scene_raw.npy"
    )
    assert _value_after(evaluate, "--tag") == "gn_realsense"
    assert _value_after(rerank, "--rescoring-score-weight") == "1.0"


def test_detector_baseline_does_not_overwrite_grare_artifacts() -> None:
    baseline = _config(0.0)
    rerank = _rerank_command(baseline)
    evaluate = _eval_command(baseline)

    assert _value_after(rerank, "--output-root") == (
        "/output/predictions/gn_realsense_detector_baseline"
    )
    assert _value_after(evaluate, "--dump-folder") == (
        "/output/predictions/gn_realsense_detector_baseline"
    )
    assert _value_after(evaluate, "--save-raw") == (
        "/output/evaluation/gn_realsense_detector_baseline/per_scene_raw.npy"
    )
    assert _value_after(evaluate, "--tag") == "gn_realsense_detector_baseline"
    assert _value_after(rerank, "--rescoring-score-weight") == "0.0"

    reported = _config(1.0)
    assert _value_after(rerank, "--output-root") != _value_after(
        _rerank_command(reported), "--output-root"
    )
    assert _value_after(evaluate, "--save-raw") != _value_after(
        _eval_command(reported), "--save-raw"
    )


def test_intermediate_lambda_does_not_overwrite_reported_artifacts() -> None:
    ablation = _config(0.2)
    rerank = _rerank_command(ablation)
    evaluate = _eval_command(ablation)

    assert _value_after(rerank, "--output-root") == "/output/predictions/gn_realsense_lambda_0p2"
    assert _value_after(evaluate, "--dump-folder") == "/output/predictions/gn_realsense_lambda_0p2"
    assert _value_after(evaluate, "--save-raw") == (
        "/output/evaluation/gn_realsense_lambda_0p2/per_scene_raw.npy"
    )
    assert _value_after(evaluate, "--tag") == "gn_realsense_lambda_0p2"
    assert _value_after(rerank, "--rescoring-score-weight") == "0.2"
