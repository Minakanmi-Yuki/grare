"""Tests for the re-ranking worker-device resolver."""
from __future__ import annotations

import argparse
import unittest

from grare.cli import rerank as _module


def _ns(**overrides) -> argparse.Namespace:
    base = {"device": "cuda", "devices": None}
    base.update(overrides)
    return argparse.Namespace(**base)


class ResolveWorkerDevicesTests(unittest.TestCase):
    def test_no_devices_broadcasts_single_device(self) -> None:
        result = _module._resolve_worker_devices(_ns(), workers=4)
        self.assertEqual(result, ["cuda", "cuda", "cuda", "cuda"])

    def test_devices_list_used_round_robin(self) -> None:
        result = _module._resolve_worker_devices(
            _ns(devices="cuda:0,cuda:1"), workers=4
        )
        self.assertEqual(result, ["cuda:0", "cuda:1"])

    def test_devices_strips_whitespace_and_blanks(self) -> None:
        result = _module._resolve_worker_devices(
            _ns(devices=" cuda:0 , , cuda:1 "), workers=2
        )
        self.assertEqual(result, ["cuda:0", "cuda:1"])

    def test_empty_string_falls_back(self) -> None:
        result = _module._resolve_worker_devices(
            _ns(devices=""), workers=2
        )
        self.assertEqual(result, ["cuda", "cuda"])

    def test_only_blanks_falls_back(self) -> None:
        result = _module._resolve_worker_devices(
            _ns(devices=", , "), workers=3
        )
        self.assertEqual(result, ["cuda", "cuda", "cuda"])

    def test_workers_less_than_devices(self) -> None:
        # 2 workers, 4 devices: round-robin index 0,1 -> first two devices used.
        result = _module._resolve_worker_devices(
            _ns(devices="cuda:0,cuda:1,cuda:2,cuda:3"), workers=2
        )
        # Resolver returns full device list; round-robin happens at submit.
        self.assertEqual(result, ["cuda:0", "cuda:1", "cuda:2", "cuda:3"])

    def test_zero_workers_safe(self) -> None:
        result = _module._resolve_worker_devices(_ns(), workers=0)
        self.assertEqual(result, ["cuda"])


if __name__ == "__main__":
    unittest.main()
