from __future__ import annotations

import subprocess
import sys


def test_manifest_command_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "grare.cli.manifest", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Build a fresh manifest.jsonl" in result.stdout
    assert "--input-root" in result.stdout
