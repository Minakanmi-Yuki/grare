from __future__ import annotations

from argparse import Namespace
import hashlib
from pathlib import Path
import tarfile

from grare.cli import fetch
from scripts import publish_hf_assets as publish


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_dump_tar_preserves_the_dump_root_layout(tmp_path: Path) -> None:
    dump_root = tmp_path / "dumps"
    frame = dump_root / "graspnet_baseline" / "train" / "scene_0000" / "realsense" / "0000.npy"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"frame")

    entry = publish._dump_entry_index(
        {"dump_sets": {}},
        detector="graspnet_baseline",
        camera="realsense",
        dump_root=dump_root,
    )
    assert entry["frame_counts"] == {"train": 1, "test": 0}

    shard = tmp_path / "dumps.tar"
    publish._create_tar(
        source_paths=publish._dump_scene_paths([frame.parents[1]], "realsense"),
        data_root=dump_root,
        destination=shard,
    )
    with tarfile.open(shard) as archive:
        assert archive.getnames() == ["graspnet_baseline/train/scene_0000/realsense/0000.npy"]


def test_fetch_dumps_extracts_and_verifies_a_complete_split(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "source"
    frame = source_root / "graspnet_baseline" / "train" / "scene_0000" / "realsense" / "0000.npy"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"frame")
    shard = tmp_path / "scene.tar"
    with tarfile.open(shard, mode="w") as archive:
        archive.add(frame, arcname=str(frame.relative_to(source_root)))

    remote_path = "shards/v1/graspnet_baseline/realsense/dumps/train/scenes-0000-0000.tar"
    index = {
        "dump_sets": {
            "graspnet_baseline/realsense": {
                "complete": True,
                "frame_counts": {"train": 1, "test": 0},
                "shards": [
                    {
                        "stage": "dumps",
                        "split": "train",
                        "path": remote_path,
                        "size": shard.stat().st_size,
                        "sha256": _sha256(shard),
                    }
                ],
            }
        }
    }
    monkeypatch.setattr(fetch, "_load_index", lambda **_: index)
    monkeypatch.setattr(fetch, "_hf_hub_download", lambda **_: shard)
    output_root = tmp_path / "output"
    args = Namespace(
        output_root=str(output_root),
        split="train",
        detector="graspnet_baseline",
        camera="realsense",
        repo_id="example/dataset",
        dry_run=False,
        overwrite=False,
        no_verify=False,
    )

    assert fetch._fetch_dumps(args) == 0
    assert (
        output_root / "graspnet_baseline" / "train" / "scene_0000" / "realsense" / "0000.npy"
    ).read_bytes() == b"frame"


def test_checkpoint_selection_resolves_the_matching_config(monkeypatch) -> None:
    loaded: list[Path] = []

    def fake_load(path: str | Path, _overrides: list[str]) -> dict:
        loaded.append(Path(path))
        return {"name": "gn_realsense"}

    monkeypatch.setattr(fetch, "load_config", fake_load)
    config, path = fetch._checkpoint_config(
        Namespace(config=None, detector="graspnet_baseline", camera="realsense")
    )

    assert config["name"] == "gn_realsense"
    assert path.name == "gn_realsense.yaml"
    assert loaded == [path]
