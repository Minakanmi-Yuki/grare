"""FiLM zero-init: a freshly built modulator acts as the identity."""
from __future__ import annotations

import unittest

import torch

from grare.rescoring.model import FiLMModulator


class FiLMZeroInitTests(unittest.TestCase):
    def test_zero_init_is_identity(self):
        torch.manual_seed(0)
        film = FiLMModulator(target_dim=32, cond_dim=16)
        x = torch.randn(4, 32)
        cond = torch.randn(4, 16)
        out = film(x, cond)
        self.assertTrue(
            torch.allclose(out, x, atol=0.0, rtol=0.0),
            f"FiLM modulator at init must equal x; max diff {(out - x).abs().max().item():.3e}",
        )

    def test_after_training_step_breaks_identity(self):
        # Sanity: once γ, β receive non-zero gradients, FiLM diverges from
        # identity. Confirms zero-init is the only thing keeping it equal.
        torch.manual_seed(0)
        film = FiLMModulator(target_dim=8, cond_dim=4)
        x = torch.randn(2, 8)
        cond = torch.randn(2, 4)
        target = torch.randn(2, 8)
        opt = torch.optim.SGD(film.parameters(), lr=0.5)
        for _ in range(3):
            opt.zero_grad()
            loss = (film(x, cond) - target).pow(2).mean()
            loss.backward()
            opt.step()
        out = film(x, cond)
        self.assertFalse(torch.allclose(out, x, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
