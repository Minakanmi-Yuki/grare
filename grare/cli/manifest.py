"""Build or refresh an archive manifest without relabeling any data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from grare.relabeling.manifest import build_archive_manifest
from grare.utils.cpu import effective_cpu_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fresh manifest.jsonl for an existing relabeled archive tree."
    )
    parser.add_argument("--input-root", required=True, help="Root containing relabeled .npz archives.")
    parser.add_argument("--pattern", default="**/*.npz")
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--success-mu-thresh", type=float, default=0.4)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(16, max(1, effective_cpu_count())),
        help="Manifest scan workers; default is up to 16 usable CPUs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_archive_manifest(
        Path(args.input_root),
        manifest_path=args.manifest_path,
        summary_path=args.summary_path,
        pattern=args.pattern,
        success_mu_thresh=float(args.success_mu_thresh),
        num_workers=max(1, int(args.num_workers)),
        show_progress=True,
    )
    print(
        json.dumps(
            {
                "stage": "manifest",
                "input_root": str(Path(args.input_root).resolve()),
                "manifest_path": str(result.manifest_path),
                "summary_path": str(result.summary_path),
                "summary": result.summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
