"""Synthetic checks for expert output and fixed-router-weight comparison."""

import sys
import json
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p2.experts.similarity import (fixed_weight_replacement_relative_l2,
                                   output_similarity, reconstruct_routed_output)
from scripts.aggregate_m0_5 import load_pairs, load_rows


class ExpertSimilarityTests(unittest.TestCase):
    def test_cosine_relative_l2_and_norm_metrics(self):
        metrics = output_similarity(torch.tensor([3.0, 4.0]), torch.tensor([0.0, 2.0]))
        self.assertAlmostEqual(metrics["cosine_similarity"], 0.8)
        self.assertAlmostEqual(metrics["relative_l2"], (13**0.5) / 5.0)
        self.assertAlmostEqual(metrics["left_norm"], 5.0)
        self.assertAlmostEqual(metrics["right_norm"], 2.0)

    def test_router_weighted_reconstruction(self):
        outputs = {3: torch.tensor([1.0, 2.0]), 8: torch.tensor([-1.0, 1.0])}
        result = reconstruct_routed_output(outputs, [3, 8], [0.25, 0.5])
        torch.testing.assert_close(result, torch.tensor([-0.25, 1.0]))

    def test_fixed_router_weight_counterfactual_delta(self):
        baseline = torch.tensor([4.0, 0.0])
        source = torch.tensor([2.0, 0.0])
        replacement = torch.tensor([0.0, 2.0])
        observed = fixed_weight_replacement_relative_l2(baseline, source, replacement, 0.5)
        self.assertAlmostEqual(observed, (2**0.5) / 4.0)

    def test_rejects_misaligned_or_nonfinite_outputs(self):
        with self.assertRaisesRegex(ValueError, "same-shape"):
            output_similarity(torch.ones(2), torch.ones(3))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            output_similarity(torch.tensor([float("nan")]), torch.ones(1))
        with self.assertRaisesRegex(ValueError, "unique"):
            reconstruct_routed_output({1: torch.ones(2)}, [1, 1], [0.5, 0.5])

    def test_aggregator_rejects_incomplete_expert_pair_coverage(self):
        base = {"evidence": "MEASURED_COMMUNITY_AWQ_INT4", "layer_id": 13,
                "sample_id": "s1", "token_position": 4, "expert_id": 6,
                "frequency_class": "high", "workload_group": "general",
                "cosine_similarity": 1.0, "relative_l2": 0.0, "left_norm": 1.0,
                "right_norm": 1.0, "norm_ratio_right_over_left": 1.0,
                "absolute_l2": 0.0, "native_router_weight": 0.2,
                "fixed_weight_replacement_relative_l2": 0.0, "native_selected": True}
        rows = [dict(base, expert_id=expert) for expert in (6, 63)]
        pair = {"evidence": "MEASURED_COMMUNITY_AWQ_INT4", "layer_id": 13,
                "sample_id": "s1", "token_position": 4, "expert_a": 6, "expert_b": 63,
                "cosine_similarity": 1.0, "relative_l2": 0.0, "left_norm": 1.0,
                "right_norm": 1.0, "norm_ratio_right_over_left": 1.0, "absolute_l2": 0.0}
        with tempfile.TemporaryDirectory() as directory:
            rows_path, pairs_path = Path(directory) / "rows.json", Path(directory) / "pairs.json"
            rows_path.write_text(json.dumps(rows), encoding="utf-8")
            pairs_path.write_text(json.dumps([pair]), encoding="utf-8")
            self.assertEqual(len(load_pairs(pairs_path, load_rows(rows_path))), 1)
            pairs_path.write_text(json.dumps([]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty"):
                load_pairs(pairs_path, rows)


if __name__ == "__main__":
    unittest.main()
