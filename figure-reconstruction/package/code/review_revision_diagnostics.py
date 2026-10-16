"""Path-neutral helpers required by the saved-aggregate figure renderer.

This module is a release adapter, not the scientific diagnostics implementation.
It deliberately omits CUDA, model fitting, row readers, and protected input roots.
"""
import hashlib
import json
import os
from pathlib import Path

STORE = Path(os.environ.get("VALORANT_ARTIFACT_ROOT", "."))
TASKS = ("map_winner", "series_winner")


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
