"""GraspNet1B evaluation adapter and result utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["EvaluationSummary", "GraspNetEvalAdapter"]

_SYMBOL_TO_MODULE = {
    "EvaluationSummary": "common",
    "GraspNetEvalAdapter": "graspnet_eval_adapter",
}


def __getattr__(name: str) -> Any:
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = __import__(f"{__name__}.{module_name}", fromlist=[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from .common import EvaluationSummary
    from .graspnet_eval_adapter import GraspNetEvalAdapter
