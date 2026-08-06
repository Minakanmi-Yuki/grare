from __future__ import annotations

from grare.cli.run import _eval_command, _pack_command, _pack_execution, _rerank_command, _train_command


def _config(*, resume_from: str | None) -> dict:
    return {
        "camera": "realsense",
        "paths": {
            "local_cloud_train": "/data/local",
            "checkpoint_dir": "/output/checkpoints",
            "object_pooled_train": "/data/object",
            "point_mae_checkpoint": "/checkpoints/point_mae.pth",
        },
        "train": {
            "val_ratio": 0.1,
            "val_split_seed": 7,
            "batch_size": 2048,
            "min_epochs": 20,
            "max_epochs": 40,
            "min_optimizer_steps": 0,
            "min_checkpoint_steps": 0,
            "early_stop_patience": 10,
            "lr": 0.0002,
            "weight_decay": 0.0001,
            "grad_clip_norm": 1.0,
            "scheduler_patience": 4,
            "scheduler_factor": 0.7,
            "scheduler_min_lr": 0.00001,
            "scheduler_warmup_steps": 0,
            "checkpoint_interval_steps": 0,
            "resume_from": resume_from,
            "device": "cuda",
            "num_workers": 8,
            "amp": "auto",
            "seed": 7,
        },
    }


def test_train_command_omits_empty_resume_path() -> None:
    command = _train_command(_config(resume_from=None))
    assert "--resume-from" not in command


def test_train_command_includes_resume_path() -> None:
    checkpoint = "/checkpoints/step_00005000.pt"
    command = _train_command(_config(resume_from=checkpoint))
    index = command.index("--resume-from")
    assert command[index + 1] == checkpoint


def test_kinect_packed_command_uses_packed_loader_only() -> None:
    config = _config(resume_from=None)
    config["paths"]["packed_train"] = "/data/packed"
    command = _train_command(config)
    assert "--packed-dataset-root" in command
    assert "--archive-batch-sampling" not in command
    assert "--object-pooled-root" not in command


def test_kinect_pack_command_uses_lazy_feature_inputs() -> None:
    config = _config(resume_from=None)
    config["paths"]["packed_train"] = "/data/packed"
    command = _pack_command(config)
    assert command is not None
    assert "--input-root" in command
    assert "--output-root" in command
    assert "--object-pooled-root" in command
    assert "--require-object-pooled" in command
    assert "--require-archive-manifest" in command
    assert "--include-object-cloud" not in command


def test_realsense_pack_command_is_not_configured() -> None:
    assert _pack_command(_config(resume_from=None)) is None


def test_pack_execution_reuses_completed_output_and_replaces_partial_output(tmp_path) -> None:
    config = _config(resume_from=None)
    output_root = tmp_path / "packed"
    config["paths"]["packed_train"] = str(output_root)

    command, skipped = _pack_execution(config, repack=False)
    assert command is not None
    assert "--overwrite" not in command
    assert skipped is None

    output_root.mkdir()
    command, skipped = _pack_execution(config, repack=False)
    assert command is not None
    assert "--overwrite" in command
    assert skipped is None

    (output_root / "index.json").write_text("{}", encoding="utf-8")
    command, skipped = _pack_execution(config, repack=False)
    assert command is None
    assert skipped is not None

    command, skipped = _pack_execution(config, repack=True)
    assert command is not None
    assert "--overwrite" in command
    assert skipped is None


def test_rerank_command_uses_fixed_protocol_arguments() -> None:
    config = {
        "camera": "realsense",
        "paths": {
            "local_cloud_test": "/data/local_test",
            "checkpoint_dir": "/output/checkpoints",
            "object_pooled_test": "/data/object_test",
            "rerank_dir": "/output/predictions",
        },
        "rerank": {"device": "cuda", "num_workers": 1, "lambda": 1.0},
    }
    command = _rerank_command(config)
    assert "--rescoring-score-weight" in command
    assert "--pose-dim" not in command
    assert "--score-normalization" not in command


def test_eval_command_force_discards_prior_checkpoint_shards() -> None:
    config = {
        "name": "example",
        "camera": "kinect",
        "paths": {
            "graspnet_root": "/data/graspnet",
            "rerank_dir": "/output/predictions",
            "eval_dir": "/output/evaluation",
        },
        "rerank": {"lambda": 1.0},
        "eval": {"proc": 16},
    }
    assert "--force" not in _eval_command(config)
    assert _eval_command(config, force=True)[-1] == "--force"
