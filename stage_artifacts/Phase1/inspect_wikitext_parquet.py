#!/usr/bin/env python3
"""Inspect pinned WikiText Parquet and reconstruct its raw test stream."""
import hashlib
import sys
from pathlib import Path

import pyarrow.parquet as pq

source = Path(sys.argv[1])
target = Path(sys.argv[2])
table = pq.read_table(source)
assert table.column_names == ["text"], table.column_names
rows = table.column("text").to_pylist()
assert len(rows) == 4358, len(rows)
assert all(isinstance(row, str) for row in rows)
raw = "".join(row if row.endswith("\n") else row + "\n" for row in rows).encode("utf-8")
target.write_bytes(raw)
print({"rows": len(rows), "parquet_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
       "raw_bytes": len(raw), "raw_sha256": hashlib.sha256(raw).hexdigest(),
       "first_rows": [repr(row[:80]) for row in rows[:4]]})
