"""Bounded Safetensors metadata and read-only tensor payload operations."""

import hashlib
import json
import re
import struct
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from http.client import HTTPException
from math import prod
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPErrorProcessor, Request, build_opener

from p2.weights.checkpoint import REVISION, json_object

MAX_HEADER_BYTES = 4 * 1024 * 1024
MAX_METADATA_BYTES = 2 * 1024 * 1024
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4,
    "U64": 8, "I64": 8, "F64": 8,
}


def now():
    return datetime.now().astimezone().isoformat()


@dataclass(frozen=True)
class Tensor:
    dtype: str
    shape: tuple[int, ...]
    offsets: tuple[int, int]

    def serialize(self):
        return {"dtype": self.dtype, "shape": self.shape, "data_offsets": self.offsets}


@dataclass(frozen=True)
class Header:
    tensors: dict[str, Tensor]
    length: int
    file_size: int
    sha256: str


def exact_read(stream, length):
    data = stream.read(length)
    if len(data) != length:
        raise ValueError(f"truncated read: expected {length} bytes, received {len(data)}")
    return data


def header_length(prefix):
    if len(prefix) != 8:
        raise ValueError("Safetensors length prefix must be 8 bytes")
    length, = struct.unpack("<Q", prefix)
    if not 2 <= length <= MAX_HEADER_BYTES:
        raise ValueError(f"Safetensors header length {length} outside supported bounds")
    return length


def parse_header(raw, file_size):
    if not raw.startswith(b"{"):
        raise ValueError("Safetensors header must begin with a JSON object")
    entries = json_object(raw)
    metadata = entries.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
        raise ValueError("Safetensors metadata must map strings to strings")
    tensors = {}
    for key, entry in entries.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{key}: tensor metadata must be an object")
        dtype, shape, offsets = (entry.get(field) for field in ("dtype", "shape", "data_offsets"))
        if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
            raise ValueError(f"{key}: unsupported dtype {dtype!r}")
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f"{key}: invalid shape {shape!r}")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(n) is not int for n in offsets)
                or not 0 <= offsets[0] <= offsets[1]
                or offsets[1] - offsets[0] != prod(shape) * DTYPE_BYTES[dtype]):
            raise ValueError(f"{key}: invalid offsets or byte count {offsets!r}")
        tensors[key] = Tensor(dtype, tuple(shape), tuple(offsets))
    cursor = 0
    for key, tensor in sorted(tensors.items(), key=lambda item: item[1].offsets):
        start, end = tensor.offsets
        if start != cursor:
            raise ValueError(f"{key}: gap or overlap in data offsets at {cursor}")
        cursor = end
    if cursor != file_size - 8 - len(raw):
        raise ValueError("data offsets do not cover the file payload exactly")
    return Header(tensors, len(raw), file_size, hashlib.sha256(raw).hexdigest())


def read_local_header(stream):
    length = header_length(exact_read(stream, 8))
    raw = exact_read(stream, length)
    size = stream.seek(0, 2)
    return parse_header(raw, size), raw


def copy_tensor(source, destination, header, tensor):
    start, end = tensor.offsets
    source.seek(8 + header.length + start)
    remaining = end - start
    while remaining:
        block = exact_read(source, min(remaining, 1024 * 1024))
        destination.write(block)
        remaining -= len(block)


def encode_header(entries):
    raw = json.dumps(entries, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw


class RawHttpResponses(HTTPErrorProcessor):
    # urllib's default redirect handler drains the response body before following it.
    def http_response(self, request, response):
        return response

    https_response = http_response


@contextmanager
def open_checkpoint(url, headers):
    opener = build_opener(RawHttpResponses())
    hub_headers = {}
    for _ in range(5):
        target = urlsplit(url)
        if target.scheme != "https" or target.username is not None:
            raise ValueError("checkpoint requests and redirects must use HTTPS without credentials")
        with opener.open(Request(url, headers=headers), timeout=20) as response:
            if target.hostname == "huggingface.co":
                for key in ("X-Repo-Commit", "X-Linked-ETag"):
                    if response.headers.get(key):
                        hub_headers[key] = response.headers[key]
                commit = hub_headers.get("X-Repo-Commit")
                if commit and commit != REVISION:
                    raise ValueError(f"unexpected repository commit {commit}")
            if response.status not in (301, 302, 303, 307, 308):
                yield response, hub_headers
                return
            location = response.headers.get("Location")
            if not location:
                raise ValueError("checkpoint redirect has no Location")
            url = urljoin(url, location)
    raise ValueError("too many checkpoint redirects")


def request_bytes(url, *, bounds=None, max_metadata_bytes=MAX_METADATA_BYTES):
    headers = {"Accept-Encoding": "identity"}
    if bounds is not None:
        start, end = bounds
        headers["Range"] = f"bytes={start}-{end}"
    try:
        with open_checkpoint(url, headers) as (response, hub_headers):
            status = response.status
            if status >= 400:
                raise ValueError(f"HTTP {status}; response body not read")
            content_range = response.headers.get("Content-Range")
            commit = hub_headers.get("X-Repo-Commit") or response.headers.get("X-Repo-Commit")
            if commit and commit != REVISION:
                raise ValueError(f"unexpected repository commit {commit}")
            content_length = response.headers.get("Content-Length")
            if content_length is not None and not re.fullmatch(r"\d+", content_length):
                raise ValueError(f"invalid Content-Length {content_length!r}")
            length = int(content_length) if content_length is not None else None
            if bounds is not None:
                if status != 206:
                    raise ValueError(f"Range refused: HTTP {status}; response body not read")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range or "")
                if not match:
                    raise ValueError(f"invalid Content-Range {content_range!r}")
                first, last, size = map(int, match.groups())
                if (first, last) != bounds or last >= size:
                    raise ValueError(f"unexpected Content-Range {content_range!r}")
                limit = end - start + 1
                if limit > MAX_HEADER_BYTES or (length is not None and length != limit):
                    raise ValueError("Range response length does not match the request or bound")
            else:
                if status != 200 or (length is not None and length > max_metadata_bytes):
                    raise ValueError(f"unexpected metadata response: HTTP {status}, length {length}")
                limit, size = max_metadata_bytes, None
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise ValueError("encoded checkpoint response rejected")
            data = response.read(limit + 1)
            if len(data) > limit or (length is not None and len(data) != length):
                raise ValueError("response length exceeded limit or differed from Content-Length")
            if bounds is not None and len(data) != limit:
                raise ValueError("truncated Range response")
            return data, {
                "url": url, "response_host": urlsplit(response.geturl()).hostname,
                "http_status": status, "content_range": content_range,
                "response_bytes": len(data), "object_bytes": size,
                "etag": response.headers.get("ETag"),
                "linked_etag": hub_headers.get("X-Linked-ETag"),
                "repo_commit": commit, "retrieved_at": now(),
            }
    except HTTPError as error:
        error.close()
        raise ValueError(f"HTTP {error.code}; response body not read") from None
    except URLError as error:
        # Redirect URLs may carry temporary signatures.
        raise ValueError(f"connection failed: {type(error.reason).__name__}") from None
    except HTTPException as error:
        raise ValueError(f"invalid HTTP response: {type(error).__name__}") from None


def read_remote_header(url, requests):
    prefix, first = request_bytes(url, bounds=(0, 7))
    requests.append(first)
    length = header_length(prefix)
    raw, second = request_bytes(url, bounds=(8, 7 + length))
    requests.append(second)
    if first["object_bytes"] != second["object_bytes"]:
        raise ValueError("shard size changed between Range requests")
    identities = ("etag", "linked_etag")
    if (not any(first[key] for key in identities)
            or any(first[key] != second[key] for key in identities)):
        raise ValueError("missing or inconsistent shard identity across Range requests")
    return parse_header(raw, first["object_bytes"]), raw
