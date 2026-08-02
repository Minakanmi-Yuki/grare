from __future__ import annotations


def test_point_mae_precompute_batch_default_is_memory_safe() -> None:
    from grare.cli.precompute_object import DEFAULT_BATCH_SIZE

    assert DEFAULT_BATCH_SIZE == 2048
