#!/usr/bin/env python3
"""Write a path-neutral, self-excluding SHA-256 manifest for a local candidate."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: write_candidate_manifest.py CANDIDATE_ROOT")

    root = Path(sys.argv[1]).resolve()
    if not root.is_dir():
        raise SystemExit("candidate root must be an existing directory")

    manifest = root / "candidate-manifest.json"
    entries = []
    total_size = 0
    for path in sorted(root.rglob("*")):
        if path == manifest or not path.is_file():
            continue
        if path.is_symlink():
            raise SystemExit(f"refusing symlink: {path.relative_to(root).as_posix()}")
        size = path.stat().st_size
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": size,
                "sha256": sha256(path),
            }
        )
        total_size += size

    payload = {
        "candidate": root.name,
        "manifest_scope": "all regular candidate files except this manifest",
        "file_count": len(entries),
        "total_size_bytes": total_size,
        "files": entries,
    }
    temporary = root / ".candidate-manifest.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest)
    print(json.dumps({"manifest": manifest.name, "file_count": len(entries), "total_size_bytes": total_size}))


if __name__ == "__main__":
    main()
