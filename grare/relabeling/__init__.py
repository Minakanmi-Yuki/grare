"""Analytic relabeling pipeline modules (GraspNet1B)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "AnalyticLabelConfig",
    "AnalyticLabeler",
    "BatchAnalyticRelabeler",
    "OnlineFeatureExtractor",
    "ArchiveRelabeledCandidateDataset",
    "PackedRelabeledCandidateDataset",
    "RelabeledCandidateDataset",
    "SceneLabelingConfig",
    "build_manifest",
    "collate_archive_batches",
    "collate_rescoring_batch",
    "MANIFEST_FILENAME",
    "MANIFEST_SUMMARY_FILENAME",
    "build_archive_manifest",
    "default_manifest_path",
    "default_manifest_summary_path",
    "load_manifest_records",
    "manifest_records_by_archive_path",
    "summarize_manifest_records",
    "validate_manifest_coverage",
]

_SYMBOL_TO_MODULE = {
    "AnalyticLabelConfig": "analytic_labeler",
    "AnalyticLabeler": "analytic_labeler",
    "BatchAnalyticRelabeler": "scene_labeling",
    "OnlineFeatureExtractor": "scene_labeling",
    "ArchiveRelabeledCandidateDataset": "dataset_builder",
    "PackedRelabeledCandidateDataset": "dataset_builder",
    "RelabeledCandidateDataset": "dataset_builder",
    "SceneLabelingConfig": "scene_labeling",
    "build_manifest": "dataset_builder",
    "collate_archive_batches": "dataset_builder",
    "collate_rescoring_batch": "dataset_builder",
    "MANIFEST_FILENAME": "manifest",
    "MANIFEST_SUMMARY_FILENAME": "manifest",
    "build_archive_manifest": "manifest",
    "default_manifest_path": "manifest",
    "default_manifest_summary_path": "manifest",
    "load_manifest_records": "manifest",
    "manifest_records_by_archive_path": "manifest",
    "summarize_manifest_records": "manifest",
    "validate_manifest_coverage": "manifest",
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
    from .analytic_labeler import AnalyticLabelConfig, AnalyticLabeler
    from .dataset_builder import (
        ArchiveRelabeledCandidateDataset,
        PackedRelabeledCandidateDataset,
        RelabeledCandidateDataset,
        build_manifest,
        collate_archive_batches,
        collate_rescoring_batch,
    )
    from .manifest import (
        MANIFEST_FILENAME,
        MANIFEST_SUMMARY_FILENAME,
        build_archive_manifest,
        default_manifest_path,
        default_manifest_summary_path,
        load_manifest_records,
        manifest_records_by_archive_path,
        summarize_manifest_records,
        validate_manifest_coverage,
    )
    from .scene_labeling import BatchAnalyticRelabeler, OnlineFeatureExtractor, SceneLabelingConfig
