"""Small process-wide runtime defaults shared by command-line entry points."""
from __future__ import annotations

import os


def configure_thread_pools() -> None:
    """Avoid BLAS oversubscription while preserving explicit user choices."""
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")
