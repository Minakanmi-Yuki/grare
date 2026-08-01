from __future__ import annotations

import os

from grare.utils.cpu import cgroup_cpu_quota, effective_cpu_count


def test_effective_cpu_count_is_at_least_one() -> None:
    assert effective_cpu_count() >= 1


def test_effective_cpu_count_respects_a_container_quota() -> None:
    """os.cpu_count() reports host cores and would oversubscribe a small cgroup."""
    quota = cgroup_cpu_quota()
    if quota is None:
        return
    assert effective_cpu_count() <= max(1, int(quota)) or effective_cpu_count() == 1


def test_effective_cpu_count_never_exceeds_the_affinity_mask() -> None:
    try:
        allowed = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return
    assert effective_cpu_count() <= allowed


def test_evaluate_default_proc_is_bounded_by_usable_cores() -> None:
    from grare.cli.evaluate import _default_proc

    assert 1 <= _default_proc() <= max(1, effective_cpu_count())


def test_dump_default_workers_is_bounded_by_usable_cores() -> None:
    from grare.cli.dump import _default_workers

    assert 1 <= _default_workers() <= max(1, min(6, effective_cpu_count()))
