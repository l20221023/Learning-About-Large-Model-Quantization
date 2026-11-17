"""SYNTHETIC test of the AWQ export's layer and token coordinates."""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import torch
except ModuleNotFoundError as error:
    if error.name == "torch":
        raise unittest.SkipTest("AWQ export test requires PyTorch") from error
    raise

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.trace_awq_3090 import validate_and_export
from p2.trace.router import checked_routes
from verify_router_trace import softmax_matches


class AwqExportTest(unittest.TestCase):
    def test_fp16_probability_midpoint_uses_stable_reference_rounding(self):
        logits = torch.randn((8, 64), generator=torch.Generator().manual_seed(186)).half()
        native_like = logits.double().softmax(-1).half()
        weights, ids = native_like.topk(6, dim=-1)
        cpu_half = logits.softmax(-1)
        self.assertGreater((weights - cpu_half.gather(1, ids)).abs().max().item(), 0)
        checked = checked_routes(logits, ids, weights.float(), softmax_dtype=torch.float16)
        self.assertTrue(torch.equal(checked, native_like.float()))

    def test_half_rounding_around_a_softmax_midpoint(self):
        observed = 0.01153564453125
        expected = 0.011539459732553286
        self.assertTrue(softmax_matches(observed, expected, "torch.float16"))
        self.assertFalse(softmax_matches(observed, expected, "torch.float32"))
        self.assertFalse(softmax_matches(observed * 2, expected, "torch.float16"))

    def test_native_route_coordinates_and_padding_invariant(self):
        logits = (torch.arange(128, dtype=torch.float32).reshape(2, 64) / 64).half()
        weights, ids = logits.softmax(-1).topk(6)
        capture = ("torch.float16", logits.tolist(), ids.tolist(), weights.tolist())
        routes = {layer: capture for layer in range(1, 27)}
        with TemporaryDirectory() as temp:
            path = Path(temp) / "trace.jsonl"
            count = validate_and_export(routes, [100000, 21], {100000}, path, evidence="SYNTHETIC")
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(count, len(rows))
            self.assertEqual([row["layer_id"] for row in rows], list(range(1, 27)))
            self.assertTrue(all(row["sequence_index"] == 1 and row["token_id"] == 21 for row in rows))
            self.assertEqual(rows[0]["topk_weights"], weights.tolist()[1])
            self.assertEqual(rows[0]["router_softmax_dtype"], "torch.float16")
            self.assertEqual(rows[0]["evidence"], "SYNTHETIC")
            with self.assertRaisesRegex(ValueError, "router token count differs"):
                validate_and_export(routes, [21], set(), path)


if __name__ == "__main__":
    unittest.main()
