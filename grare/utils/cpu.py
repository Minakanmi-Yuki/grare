"""Effective CPU count, honouring container limits.

``os.cpu_count()`` reports the host's cores, not the share a container may use.
On a machine with 128 cores and a 2.5-core cgroup quota it returns 128, so
defaults derived from it oversubscribe the container badly: the run queue grows
into the dozens, the scheduler throttles the cgroup, and the GPU waits on CPU.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["effective_cpu_count", "cgroup_cpu_quota"]


def cgroup_cpu_quota() -> float | None:
    """CPU cores this cgroup may use, or None when unlimited or unavailable."""
    # cgroup v2: "<quota> <period>", or "max <period>" when unlimited.
    v2 = Path("/sys/fs/cgroup/cpu.max")
    try:
        quota_text, period_text = v2.read_text().split()[:2]
        if quota_text != "max":
            period = float(period_text)
            if period > 0:
                return float(quota_text) / period
        return None
    except (OSError, ValueError):
        pass

    # cgroup v1: separate quota and period files; -1 means unlimited.
    try:
        quota = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip())
        period = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
        if quota > 0 and period > 0:
            return quota / period
    except (OSError, ValueError):
        pass

    return None


def effective_cpu_count() -> int:
    """Usable cores: the cgroup quota, the CPU affinity mask, or the host count.

    Always at least 1, so callers can divide by it safely.
    """
    counts: list[float] = []

    quota = cgroup_cpu_quota()
    if quota is not None and quota > 0:
        counts.append(quota)

    try:
        counts.append(float(len(os.sched_getaffinity(0))))
    except (AttributeError, OSError):
        host = os.cpu_count()
        if host:
            counts.append(float(host))

    if not counts:
        return 1
    return max(1, int(min(counts)))
