"""Synthetic invariants for M0-4 aggregation; never treated as model evidence."""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.aggregate_m0_4 import aggregate, load_rows


class M04AggregateTest(unittest.TestCase):
    def test_balanced_synthetic_trace_and_outputs(self):
        rows = []
        ids = [0, 1, 2, 3, 4, 5]
        probs = [1 / 64] * 64
        for group in ("general", "chinese", "code"):
            for layer in range(1, 27):
                rows.append({
                    "evidence": "MEASURED_COMMUNITY_AWQ_INT4", "workload_group": group,
                    "sample_id": f"{group}-01", "layer_id": layer, "sequence_index": 0,
                    "topk_expert_ids": ids, "topk_weights": [1 / 64] * 6,
                    "router_probabilities": probs,
                })
        with TemporaryDirectory() as temp:
            base = Path(temp)
            source = base / "trace.jsonl"
            rows.append(rows[0].copy())
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate trace coordinate"):
                load_rows(source)
            rows.pop()
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            result = aggregate(load_rows(source), base / "aggregate")
            self.assertEqual(result["rows"], 78)
            self.assertEqual(result["valid_tokens_per_group"], {"general": 1, "chinese": 1, "code": 1})
            self.assertEqual(len(list((base / "aggregate" / "layer_charts").glob("*.svg"))), 26)
            count_rows = (base / "aggregate" / "routing_counts.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(count_rows), 3 * 26 * 64 + 1)

    def test_selection_sum_mismatch_rejected(self):
        row = {"evidence": "MEASURED_COMMUNITY_AWQ_INT4", "workload_group": "general",
               "sample_id": "g-01", "layer_id": 1, "sequence_index": 0,
               "topk_expert_ids": list(range(6)), "topk_weights": [1/64]*6,
               "router_probabilities": [1/64]*64}
        with TemporaryDirectory() as temp:
            source = Path(temp) / "trace.jsonl"
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing one or more MoE layers"):
                aggregate(load_rows(source), Path(temp) / "out")


if __name__ == "__main__":
    unittest.main()
