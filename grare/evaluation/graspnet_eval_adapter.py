from __future__ import annotations

from pathlib import Path
from typing import Callable
import numpy as np

from graspnetAPI import GraspNetEval

from .common import EvaluationSummary, save_summary, summarize_topk_accuracy


class GraspNetEvalAdapter:
    def __init__(
        self,
        dataset_root: str,
        camera: str = "kinect",
        split: str = "test",
        scene_ids: list[int] | None = None,
    ) -> None:
        self.dataset_root = dataset_root
        self.camera = camera
        self.split = split
        self.scene_ids = None if scene_ids is None else [int(scene_id) for scene_id in scene_ids]
        if self.scene_ids is not None:
            self.evaluator = GraspNetEval(
                root=dataset_root,
                camera=camera,
                split=self._resolve_custom_eval_split(),
            )
        elif split == "train":
            self.evaluator = GraspNetEval(root=dataset_root, camera=camera, split="train")
        else:
            self.evaluator = GraspNetEval(root=dataset_root, camera=camera, split="test")

    def evaluate(
        self,
        dump_folder: str | Path,
        *,
        proc: int = 4,
        save_path: str | Path | None = None,
    ) -> EvaluationSummary:
        res, ap_values = self.evaluate_raw(dump_folder, proc=proc)
        summary = summarize_topk_accuracy(
            res,
            benchmark="graspnet",
            camera=self.camera,
            split=self.split,
            dump_folder=dump_folder,
            ap_values=ap_values,
        )
        if self.scene_ids is not None:
            summary.extra.update(
                {
                    "custom_scene_ids": list(self.scene_ids),
                    "num_custom_scenes": len(self.scene_ids),
                }
            )
        if save_path is not None:
            save_summary(summary, save_path)
        return summary

    def evaluate_raw(self, dump_folder: str | Path, *, proc: int = 4) -> tuple[np.ndarray, list[float] | float]:
        if self.scene_ids is not None:
            return self._evaluate_custom_scene_ids(str(dump_folder), proc=proc)
        eval_fn = self._resolve_eval_fn()
        return eval_fn(str(dump_folder), proc=proc)

    def summarize(
        self,
        res: np.ndarray,
        ap_values: list[float] | tuple[float, ...] | np.ndarray | float,
        *,
        dump_folder: str | Path,
    ) -> EvaluationSummary:
        summary = summarize_topk_accuracy(
            res,
            benchmark="graspnet",
            camera=self.camera,
            split=self.split,
            dump_folder=dump_folder,
            ap_values=ap_values,
        )
        if self.scene_ids is not None:
            summary.extra.update(
                {
                    "custom_scene_ids": list(self.scene_ids),
                    "num_custom_scenes": len(self.scene_ids),
                }
            )
        return summary

    def _resolve_eval_fn(self) -> Callable[..., tuple]:
        if self.split == "train":
            return self._eval_train_split
        if self.split == "test":
            return self.evaluator.eval_all
        if self.split == "test_seen":
            return self.evaluator.eval_seen
        if self.split == "test_similar":
            return self.evaluator.eval_similar
        if self.split == "test_novel":
            return self.evaluator.eval_novel
        raise ValueError(f"Unsupported GraspNet split: {self.split}")

    def _resolve_custom_eval_split(self) -> str:
        assert self.scene_ids is not None
        if all(scene_id < 100 for scene_id in self.scene_ids):
            return "train"
        if all(scene_id >= 100 for scene_id in self.scene_ids):
            return "test"
        return "all"

    def _eval_train_split(self, dump_folder: str, proc: int = 4) -> tuple[np.ndarray, list[float]]:
        scene_ids = list(range(100))
        return self._evaluate_scene_ids(scene_ids, dump_folder, proc)

    def _evaluate_custom_scene_ids(self, dump_folder: str, proc: int = 4) -> tuple[np.ndarray, list[float]]:
        assert self.scene_ids is not None
        return self._evaluate_scene_ids(self.scene_ids, dump_folder, proc)

    def _evaluate_scene_ids(self, scene_ids: list[int], dump_folder: str, proc: int) -> tuple[np.ndarray, list[float]]:
        res = np.array(self.evaluator.parallel_eval_scenes(scene_ids=scene_ids, dump_folder=dump_folder, proc=proc))
        ap = float(np.mean(res))
        return res, [ap]
