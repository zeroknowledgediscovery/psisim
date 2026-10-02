#!/usr/bin/env python3
"""Unit tests for the persistent model cache (no native code or model needed)."""
from __future__ import annotations

import gzip
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "applications" / "psisimulation"))

from common import cached_json  # noqa: E402


def test_hit_miss_invalidation_and_corruption():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        model = td / "models" / "gss" / "gss_test"
        cache = td / "cache"
        calls = []

        def compute():
            calls.append(1)
            return [["a", 0.1], ["b", 0.9000000000000001]]

        v1, hit1 = cached_json(model, "psi0", compute, fingerprint="aaa", cache_dir=cache)
        v2, hit2 = cached_json(model, "psi0", compute, fingerprint="aaa", cache_dir=cache)
        assert (hit1, hit2) == (False, True) and len(calls) == 1
        assert v1 == v2 and v2[1][1] == 0.9000000000000001     # exact float round trip

        # A different fingerprint (changed model or runtime) recomputes and
        # removes the stale entry for that model.
        _, hit3 = cached_json(model, "psi0", compute, fingerprint="bbb", cache_dir=cache)
        files = sorted(p.name for p in (cache / "psi0" / "gss").iterdir())
        assert not hit3 and len(calls) == 2 and files == ["gss_test-bbb.json.gz"]

        # A corrupt or invalid entry is recomputed, never trusted.
        (cache / "psi0" / "gss" / "gss_test-bbb.json.gz").write_bytes(b"not gzip")
        _, hit4 = cached_json(model, "psi0", compute, fingerprint="bbb", cache_dir=cache)
        assert not hit4 and len(calls) == 3
        _, hit5 = cached_json(model, "psi0", compute, fingerprint="bbb", cache_dir=cache,
                              valid=lambda v: len(v) == 3)
        assert not hit5 and len(calls) == 4
        with gzip.open(cache / "psi0" / "gss" / "gss_test-bbb.json.gz", "rt") as fh:
            assert fh.read().startswith("[[")


if __name__ == "__main__":
    test_hit_miss_invalidation_and_corruption()
    print("PASS test_hit_miss_invalidation_and_corruption")
