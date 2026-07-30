from __future__ import annotations

from grare.cli.run import _rerank_command, _train_command


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
