from __future__ import annotations

import os

from grare.utils.runtime import configure_thread_pools


def test_thread_pool_defaults_preserve_explicit_values(monkeypatch) -> None:
    variables = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    for variable in variables:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "3")

    configure_thread_pools()

    assert os.environ["MKL_NUM_THREADS"] == "3"
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert os.environ[variable] == "1"
