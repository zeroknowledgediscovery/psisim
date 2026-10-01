#!/usr/bin/env python3
"""Fetch a public DTAG/LSM model from the versioned GCS model release.

This is intentionally self-contained: psisimulation does not require a DTAG
checkout. It follows DTAG's public model manifest, verifies SHA256, performs
safe .tar.zst extraction, and installs into ~/.cache/dtag/models by default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

import zstandard as zstd

DEFAULT_RELEASE = os.environ.get("DTAG_MODEL_RELEASE", "v0.2.0")
DEFAULT_BUCKET = os.environ.get(
    "DTAG_PUBLIC_BUCKET", "git-zeroknowledgediscovery-dtag"
)
MODEL_KEY_RE = re.compile(
    r"^(gss|afrobarometer|wvs|eurobarometer)/[A-Za-z0-9._-]+$"
)


def default_root() -> Path:
    return Path(
        os.environ.get("DTAG_MODEL_ROOT", "~/.cache/dtag/models")
    ).expanduser().resolve()


def manifest_url(release: str = DEFAULT_RELEASE) -> str:
    override = os.environ.get("DTAG_MODEL_MANIFEST_URL", "").strip()
    if override:
        return override
    return (
        f"https://storage.googleapis.com/{DEFAULT_BUCKET}/"
        f"models/{release}/manifest.json"
    )


def load_manifest(release: str = DEFAULT_RELEASE, timeout: float = 60) -> dict:
    with urllib.request.urlopen(manifest_url(release), timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def validate_key(key: str) -> str:
    key = str(key).strip()
    if not MODEL_KEY_RE.fullmatch(key) or ".." in key:
        raise ValueError(f"invalid model key: {key!r}")
    return key


def installed(root: Path, key: str) -> bool:
    model = root / key
    return (
        (model / "source_maps").is_dir()
        and (model / "trees" / "binary").is_dir()
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _check_member(member: tarfile.TarInfo) -> None:
    name = PurePosixPath(member.name)
    if name.is_absolute() or ".." in name.parts:
        raise RuntimeError(f"unsafe archive path: {member.name!r}")
    if member.issym() or member.islnk() or member.isdev():
        raise RuntimeError(f"unsupported archive member: {member.name!r}")


def safe_extract_tar_zst(archive: Path, dest: Path) -> None:
    with archive.open("rb") as raw:
        with zstd.ZstdDecompressor().stream_reader(raw) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tf:
                for member in tf:
                    _check_member(member)
                    try:
                        tf.extract(member, dest, filter="data")
                    except TypeError:
                        tf.extract(member, dest)


def fetch_model(
    key: str,
    root: Path | None = None,
    release: str = DEFAULT_RELEASE,
    force: bool = False,
) -> Path:
    key = validate_key(key)
    root = default_root() if root is None else Path(root).expanduser().resolve()

    if installed(root, key) and not force:
        model = root / key
        print(f"OK installed: {key} -> {model}")
        return model

    manifest = load_manifest(release)
    models = manifest.get("models", {})
    if key not in models:
        raise KeyError(
            f"{key!r} is not present in public release {release}. "
            "Use --list --family gss to inspect available keys."
        )

    entry = models[key]
    rel = PurePosixPath(str(entry["archive"]))
    if rel.is_absolute() or ".." in rel.parts:
        raise RuntimeError(f"unsafe archive path in manifest: {rel}")

    url = manifest["base_url"].rstrip("/") + "/" + str(rel)
    dest = root / key
    dest.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".psisim-model-", dir=dest.parent) as td:
        tmp = Path(td)
        archive = tmp / rel.name

        print(f"GET {key}")
        print(f"    {url}")

        with urllib.request.urlopen(url, timeout=120) as response:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with archive.open("wb") as out:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    out.write(block)
                    done += len(block)
                    if total:
                        print(f"    {done / total:6.1%}", end="\r", flush=True)
        print()

        expected = str(entry["sha256"])
        got = sha256_file(archive)
        if got != expected:
            raise RuntimeError(
                f"SHA256 mismatch for {key}: expected {expected}, got {got}"
            )

        extract = tmp / "extract"
        extract.mkdir()
        safe_extract_tar_zst(archive, extract)

        top = extract / Path(key).name
        if not (top / "source_maps").is_dir():
            raise RuntimeError("archive missing source_maps/")
        if not (top / "trees" / "binary").is_dir():
            raise RuntimeError("archive missing trees/binary/")

        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(top), str(dest))

    print(f"INSTALLED {key} -> {dest}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        nargs="?",
        default="gss/gss_2018",
        help="public model key, e.g. gss/gss_2018",
    )
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--root", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--family", default="")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else default_root()

    if args.list:
        manifest = load_manifest(args.release)
        keys = sorted(manifest.get("models", {}))
        if args.family:
            keys = [k for k in keys if k.startswith(args.family + "/")]
        for key in keys:
            flag = "*" if installed(root, key) else " "
            print(f"{flag} {key}")
        return

    fetch_model(
        args.model,
        root=root,
        release=args.release,
        force=args.force,
    )


if __name__ == "__main__":
    main()
