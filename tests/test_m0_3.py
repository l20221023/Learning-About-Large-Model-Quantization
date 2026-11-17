"""SYNTHETIC checks for read-only greedy routing traces."""

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError as error:
    if error.name == "torch":
        raise unittest.SkipTest("M0-3 tests require PyTorch") from error
    raise

from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p2.trace.router import checked_routes, trace_prefill


class GreedyGate(nn.Module):
    n_routed_experts = 64
    top_k = 6
    scoring_func = "softmax"
    topk_method = "greedy"
    norm_topk_prob = False
    routed_scaling_factor = 1.0

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(64, 4) / 4)
        self.corrupt = False

    def forward(self, hidden):
        logits = hidden.reshape(-1, 4).float() @ self.weight.float().T
        weights, ids = torch.topk(logits.softmax(dim=-1), k=self.top_k, sorted=False)
        if self.corrupt:
            ids[:, 0] = 64
        return ids, weights, None


class TinyMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = GreedyGate()
        self.expert_values = nn.Parameter(torch.randn(64, 4) / 4)
        self.shared = nn.Linear(4, 4, bias=False)

    def forward(self, hidden):
        ids, weights, _ = self.gate(hidden)
        selected = self.expert_values[ids.clamp(max=63)]
        contribution = (selected * weights.unsqueeze(-1)).sum(dim=-2)
        return self.shared(hidden) + contribution.reshape_as(hidden)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(32, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 4)] + [TinyMoE() for _ in range(26)])
        self.head = nn.Linear(4, 32)

    def forward(self, input_ids, attention_mask, use_cache):
        if use_cache:
            raise ValueError("test model expects prefill without KV cache")
        hidden = self.embeddings(input_ids) * attention_mask.unsqueeze(-1)
        for layer in self.layers:
            hidden = hidden + layer(hidden) / 4
        return SimpleNamespace(logits=self.head(hidden))


class RouterTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)

    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261117)
            self.model = TinyModel().eval()
        self.gates = {index: layer.gate for index, layer in enumerate(self.model.layers[1:], 1)}
        self.inputs = {
            "input_ids": torch.tensor([[0, 1, 5, 6, 2], [1, 8, 2, 19, 0]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
        }

    def trace(self):
        return trace_prefill(self.model, self.gates, self.inputs, excluded_token_ids=[0, 1, 2])

    def test_full_layer_coverage_token_alignment_and_output_identity(self):
        original = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}
        rows, stats = self.trace()
        self.assertEqual(stats["layer_ids"], list(range(1, 27)))
        self.assertEqual((stats["valid_tokens"], stats["padding_tokens"], stats["special_tokens"]),
                         (3, 3, 4))
        self.assertEqual(stats["records"], 78)
        self.assertTrue(stats["logits_exact_equal"])
        self.assertEqual(stats["logits_max_abs_difference"], 0.0)
        for index, row in enumerate(rows):
            self.assertEqual(row["layer_id"], index // 3 + 1)
            coordinates = [(0, 2, 5), (0, 3, 6), (1, 1, 8)]
            self.assertEqual((row["batch_id"], row["sequence_index"], row["token_id"]),
                             coordinates[index % 3])
            logits = row["router_logits"]
            probabilities = row["router_probabilities"]
            experts, weights = row["topk_expert_ids"], row["topk_weights"]
            self.assertEqual((len(logits), len(probabilities), len(set(experts))), (64, 64, 6))
            self.assertTrue(set(experts) <= set(range(64)))
            exponentials = [math.exp(value - max(logits)) for value in logits]
            normalizer = sum(exponentials)
            for value, exponential in zip(probabilities, exponentials, strict=True):
                self.assertAlmostEqual(value, exponential / normalizer, places=7)
            for expert, weight in zip(experts, weights, strict=True):
                self.assertAlmostEqual(weight, probabilities[expert], places=7)
            self.assertLess(sum(weights), 1)
        self.assertEqual(self.trace(), (rows, stats))
        self.assertTrue(all(not gate._forward_hooks for gate in self.gates.values()))
        for name, parameter in self.model.named_parameters():
            self.assertTrue(torch.equal(parameter, original[name]), name)

    def test_equal_scores_allow_any_tied_expert_order(self):
        ids = torch.tensor([[63, 11, 20, 22, 1, 0]])
        weights = torch.full((1, 6), 1 / 64)
        probs = checked_routes(torch.zeros((1, 64)), ids, weights)
        self.assertTrue(torch.equal(probs, torch.full((1, 64), 1 / 64)))

    def test_invalid_topk_nan_and_renormalization_are_rejected(self):
        logits = torch.arange(64, dtype=torch.float32).view(1, 64)
        scores = logits.softmax(dim=-1)
        good = torch.tensor([[63, 62, 61, 60, 59, 58]])
        weights = scores.gather(1, good)
        for ids, selected, message in (
            (torch.tensor([[64, 62, 61, 60, 59, 58]]), weights, "outside 0..63"),
            (torch.tensor([[63, 63, 61, 60, 59, 58]]), weights, "duplicate"),
            (good, weights / weights.sum(), "unnormalized"),
            (good, weights.masked_fill(weights > 0.5, torch.nan), "unnormalized"),
            (good - 1, scores.gather(1, good - 1), "outside the greedy Top-6"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                checked_routes(logits, ids, selected)
        with self.assertRaisesRegex(ValueError, "non-finite router logits"):
            checked_routes(torch.full_like(logits, torch.nan), good, weights)

    def test_removes_hooks_when_native_output_is_invalid(self):
        self.gates[2].corrupt = True
        with self.assertRaisesRegex(ValueError, "outside 0..63"):
            self.trace()
        self.assertTrue(all(not gate._forward_hooks for gate in self.gates.values()))

    def test_rejects_empty_token_selection_and_training_mode(self):
        with self.assertRaisesRegex(ValueError, "no valid tokens"):
            trace_prefill(self.model, self.gates, self.inputs, excluded_token_ids=range(32))
        self.model.train()
        with self.assertRaisesRegex(ValueError, "eval-mode"):
            self.trace()

    def test_detects_missing_layer_without_leaving_hooks(self):
        unattached = GreedyGate()
        self.gates[27] = unattached
        with self.assertRaisesRegex(ValueError, "missing gate calls for layers"):
            self.trace()
        self.assertFalse(unattached._forward_hooks)


if __name__ == "__main__":
    unittest.main()
