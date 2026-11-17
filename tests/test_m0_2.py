"""SYNTHETIC fixtures. The tests require neither network nor model weights."""

import csv
import io
import json
import os
import struct
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p2.weights.audit import audit, compare
from p2.weights.checkpoint import REVISION, SHARDS, TOTAL_BYTES, expected_weights, parse_index, routed_weights
from p2.weights.extract import expert_plan, extract_expert
from p2.weights.safetensors_io import (
    MAX_HEADER_BYTES, parse_header, read_local_header, read_remote_header, request_bytes,
)


def shard_bytes(tensors):
    """Build a small fixture independently of the production serializer."""
    entries, payload = {}, bytearray()
    for key, (dtype, shape, data) in tensors.items():
        entries[key] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [len(payload), len(payload) + len(data)]}
        payload.extend(data)
    raw = json.dumps(entries).encode()
    return len(raw).to_bytes(8, "little") + raw + payload


class HeaderTests(unittest.TestCase):
    def test_header_never_reads_tensor_payload(self):
        data = shard_bytes({"a": ("BF16", [2], b"abcd")})
        stream = MagicMock(wraps=io.BytesIO(data))
        header, raw = read_local_header(stream)
        self.assertEqual([call.args for call in stream.read.call_args_list], [(8,), (len(raw),)])
        self.assertEqual(header.tensors["a"].offsets, (0, 4))

    def test_rejects_corruption(self):
        cases = (
            (struct.pack("<Q", MAX_HEADER_BYTES + 1), "outside supported bounds"),
            (shard_bytes({"a": ("BF16", [2], b"ab")}), "invalid offsets or byte count"),
            (shard_bytes({"a": ("BF16", [2], b"abcd")})[:-1], "cover the file payload"),
            (shard_bytes({"a": ("F4", [2], b"a")}), "unsupported dtype"),
            (b"1234567", "truncated read"),
        )
        for data, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                read_local_header(io.BytesIO(data))

    def test_rejects_gap_overlap_and_duplicate(self):
        for offsets in ((1, 3), (3, 5)):
            entries = {"a": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
                       "b": {"dtype": "BF16", "shape": [1], "data_offsets": offsets}}
            raw = json.dumps(entries).encode()
            with self.subTest(offsets=offsets), self.assertRaisesRegex(ValueError, "gap or overlap"):
                parse_header(raw, 8 + len(raw) + 6)
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            parse_header(b'{"a": {}, "a": {}}', 64)


class RangeTests(unittest.TestCase):
    @patch("p2.weights.safetensors_io.build_opener")
    def test_invalid_range_rejected_before_any_body_read(self, opener):
        response = opener.return_value.open.return_value.__enter__.return_value
        cases = (
            (200, {"Content-Length": "8594887408"}, "Range refused"),
            (206, {"Content-Range": "bytes 1-8/100"}, "unexpected Content-Range"),
            (206, {"Content-Range": "bytes 0-7/100", "Content-Length": "9"}, "response length"),
            (206, {"Content-Range": "bytes 0-7/*"}, "invalid Content-Range"),
            (206, {"Content-Range": "bytes 0-7/100", "X-Repo-Commit": "wrong"}, "repository commit"),
        )
        for status, headers, message in cases:
            response.status = status
            response.headers = headers
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                request_bytes("https://huggingface.co/test", bounds=(0, 7))
        response.read.assert_not_called()

    @patch("p2.weights.safetensors_io.build_opener")
    def test_valid_range_is_bounded_and_short_read_fails(self, opener):
        response = opener.return_value.open.return_value.__enter__.return_value
        response.status = 206
        response.headers = {"Content-Range": "bytes 0-7/100", "ETag": '"tag"'}
        response.geturl.return_value = "https://huggingface.co/test"
        response.read.return_value = b"12345678"
        data, record = request_bytes("https://huggingface.co/test", bounds=(0, 7))
        self.assertEqual((data, record["object_bytes"]), (b"12345678", 100))
        response.read.assert_called_once_with(9)
        response.read.return_value = b"1234567"
        with self.assertRaisesRegex(ValueError, "truncated Range"):
            request_bytes("https://huggingface.co/test", bounds=(0, 7))

    @patch("p2.weights.safetensors_io.build_opener")
    def test_redirect_preserves_range_without_reading_redirect_body(self, opener):
        redirect, final = MagicMock(), MagicMock()
        opener.return_value.open.side_effect = (redirect, final)
        first, second = (item.__enter__.return_value for item in (redirect, final))
        first.status = 302
        first.headers = {"Location": "https://example.com/shard", "X-Repo-Commit": REVISION}
        second.status = 206
        second.headers = {"Content-Range": "bytes 0-7/100"}
        second.read.return_value = b"12345678"
        second.geturl.return_value = "https://example.com/shard"
        _, record = request_bytes("https://huggingface.co/test", bounds=(0, 7))
        self.assertEqual(record["repo_commit"], REVISION)
        first.read.assert_not_called()
        for call in opener.return_value.open.call_args_list:
            self.assertEqual(call.args[0].get_header("Range"), "bytes=0-7")

    @patch("p2.weights.safetensors_io.request_bytes")
    def test_refuses_mixed_object_header(self, request):
        data = shard_bytes({"a": ("BF16", [2], b"abcd")})
        first = {"object_bytes": len(data), "etag": "first", "linked_etag": None}
        request.side_effect = [(data[:8], first), (data[8:-4], dict(first, etag="second"))]
        records = []
        with self.assertRaisesRegex(ValueError, "shard identity"):
            read_remote_header("https://example.com/shard", records)
        self.assertEqual(len(records), 2)


class CheckpointTests(unittest.TestCase):
    def test_expected_inventory_and_boundaries(self):
        expected = expected_weights()
        self.assertEqual(len(expected), 5291)
        self.assertEqual(sum(t.role == "routed" for t in expected.values()), 4992)
        self.assertEqual(len(routed_weights(1, 0)), 3)
        self.assertEqual(len(routed_weights(26, 63)), 3)
        for layer, expert in ((0, 0), (27, 0), (1, -1), (1, 64)):
            with self.subTest(layer=layer, expert=expert), self.assertRaises(ValueError):
                routed_weights(layer, expert)

    def test_rejects_index_path_traversal(self):
        for name in ("../a.safetensors", "..\\a.safetensors", "C:a.safetensors",
                     "a:stream.safetensors", "https://host/a.safetensors"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "unsafe shard"):
                parse_index(json.dumps({"weight_map": {"w": name}}).encode())

    def test_comparison_explains_unverified_and_wrong_shard(self):
        expected = routed_weights(10, 7)
        gate, up, down = expected
        header, _ = read_local_header(io.BytesIO(shard_bytes({gate: ("F32", [1], b"abcd")})))
        rows = {row["tensor_key"]: row for row in compare(
            expected, {gate: SHARDS[0], up: SHARDS[0]}, {SHARDS[1]: header})}
        self.assertEqual(rows[gate]["issue"], "shard_mismatch|shape_mismatch|dtype_mismatch")
        self.assertEqual(rows[up]["status"], "PREDICTED")
        self.assertEqual(rows[up]["dtype"], None)
        self.assertEqual(rows[down]["issue"], "missing_from_index")
        self.assertEqual(next(compare(expected, {}, {}))["status"], "UNKNOWN")

    def test_audit_retains_missing_shard_evidence_and_refuses_overwrite(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = {key: SHARDS[i % len(SHARDS)] for i, key in enumerate(expected_weights())}
            key = next(iter(routed_weights(1, 0)))
            index = root / "index.json"
            index.write_text(json.dumps({"weight_map": weights, "metadata": {"total_size": TOTAL_BYTES}}),
                             encoding="utf-8")
            out = root / "audit"
            result = audit(out, index_path=index, shard_dir=root, remote=False)
            self.assertEqual(result["status"], "PARTIAL")
            self.assertTrue(all("error" in r for r in result["shards"].values()))
            with (out / "expert_weight_mapping.csv").open(newline="", encoding="utf-8") as stream:
                row = next(row for row in csv.DictReader(stream) if row["tensor_key"] == key)
            self.assertEqual((row["status"], row["actual_shape"]), ("PREDICTED", ""))
            with self.assertRaises(FileExistsError):
                audit(out, index_path=index, shard_dir=root, remote=False)


class ExtractTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.shards = self.root / "shards"
        self.shards.mkdir()
        self.expected = routed_weights(10, 7)
        self.gate, self.up, self.down = self.expected
        self.weights = {self.gate: SHARDS[0], self.up: SHARDS[1], self.down: SHARDS[0]}
        self.index = self.root / "index.json"
        self.index.write_text(json.dumps({"weight_map": self.weights}), encoding="utf-8")
        self.source0 = self.shards / SHARDS[0]
        self.source1 = self.shards / SHARDS[1]
        self.out = self.root / "result.safetensors"

    def command(self, *args):
        return subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts" / "extract_expert.py"),
             "--index", str(self.index), "--shard-dir", str(self.shards),
             "--layer", "10", "--expert", "7", *args],
            capture_output=True, text=True, check=True,
        )

    def test_cli_dry_run_requires_only_index(self):
        result = self.command("--dry-run")
        rows = json.loads(result.stdout)["tensors"]
        self.assertEqual({r["status"] for r in rows}, {"PREDICTED"})
        self.assertEqual({r["tensor_key"] for r in rows}, self.expected.keys())
        self.assertEqual(list(self.shards.iterdir()), [])

    def test_cli_extracts_only_expert_bytes_across_two_shards(self):
        size = 1408 * 2048
        payloads = {self.gate: b"\x80\x3f" * size, self.up: b"\x00\x40" * size,
                    self.down: b"\x40\x40" * size}
        gate, up, down = [(spec.dtype, spec.shape, payloads[key]) for key, spec in self.expected.items()]
        original0 = shard_bytes({"unrelated": ("BF16", [1], b"\x80\xbf"),
                                 self.down: down, self.gate: gate})
        original1 = shard_bytes({self.up: up})
        self.source0.write_bytes(original0)
        self.source1.write_bytes(original1)
        with patch("p2.weights.extract.copy_tensor") as copy:
            rows = extract_expert(expert_plan(self.expected, self.weights), self.shards)
            copy.assert_not_called()
        self.assertEqual({r["status"] for r in rows}, {"VERIFIED"})
        self.command("--metadata-only")
        self.assertFalse(self.out.exists())
        self.command("--output", str(self.out))
        file = self.out.read_bytes()
        end = 8 + int.from_bytes(file[:8], "little")
        entries = json.loads(file[8:end])
        self.assertEqual(set(entries) - {"__metadata__"}, self.expected.keys())
        for key, spec in self.expected.items():
            entry = entries[key]
            start, stop = entry["data_offsets"]
            self.assertEqual(entry["dtype"], spec.dtype)
            self.assertEqual(entry["shape"], list(spec.shape))
            self.assertEqual(file[end + start:end + stop], payloads[key])
        for target in (self.out, self.source0):
            with self.assertRaises(FileExistsError):
                extract_expert(expert_plan(self.expected, self.weights), self.shards, target)
        self.assertEqual(self.source0.read_bytes(), original0)
        self.assertEqual(self.source1.read_bytes(), original1)
        self.assertEqual(self.out.read_bytes(), file)

    def test_missing_keys_and_shards(self):
        with self.assertRaisesRegex(ValueError, "missing from index"):
            expert_plan(self.expected, {})
        plan = expert_plan(self.expected, self.weights)
        with self.assertRaises(FileNotFoundError):
            extract_expert(plan, self.shards, self.out)
        self.source0.write_bytes(shard_bytes({}))
        self.source1.write_bytes(shard_bytes({}))
        with self.assertRaisesRegex(ValueError, "missing from"):
            extract_expert(plan, self.shards, self.out)
        self.assertFalse(self.out.exists())

    def test_invalid_shape_and_dtype_fail_before_output(self):
        self.source1.write_bytes(shard_bytes({}))
        plan = expert_plan(self.expected, self.weights)
        self.source0.write_bytes(shard_bytes({self.gate: ("BF16", [1], b"ab")}))
        with self.assertRaisesRegex(ValueError, "expected shape"):
            extract_expert(plan, self.shards, self.out)
        spec = self.expected[self.gate]
        self.source0.write_bytes(shard_bytes({self.gate: ("I16", spec.shape, b"ab" * (1408 * 2048))}))
        with self.assertRaisesRegex(ValueError, "expected dtype"):
            extract_expert(plan, self.shards, self.out)
        self.assertFalse(self.out.exists())

    def test_rejects_symlink_escape(self):
        outside = self.root / "external.safetensors"
        outside.write_bytes(shard_bytes({}))
        try:
            os.symlink(outside, self.source0)
        except OSError as error:
            if getattr(error, "winerror", None) != 1314:
                raise
            resolve = Path.resolve
            original0 = resolve(self.shards) / self.source0.name

            def simulated_link(path, strict=False):
                if path == original0:
                    return resolve(outside, strict=strict)
                return resolve(path, strict=strict)

            with patch.object(Path, "resolve", simulated_link):
                with self.assertRaisesRegex(ValueError, "inside checkpoint directory"):
                    extract_expert(expert_plan(self.expected, self.weights), self.shards)
            return
        with self.assertRaisesRegex(ValueError, "inside checkpoint directory"):
            extract_expert(expert_plan(self.expected, self.weights), self.shards)


if __name__ == "__main__":
    unittest.main()
