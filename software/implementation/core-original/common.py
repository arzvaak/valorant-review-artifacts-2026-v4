"""Durable research identities and CUDA-only numerics."""

import hashlib, json, os, time
from pathlib import Path
import numpy as np

PHASES = ("train", "stopping", "selection", "calibration", "evaluation")
TASKS = ("map_winner", "series_winner")
REPS = ("B", "BC", "BN", "BNC", "BNL", "BNLC")
FAMILIES = ("linear", "catboost", "xgboost", "mlp", "tabm")
SEMANTIC = (
    "opening",
    "stars",
    "economy",
    "pistols",
    "map_side",
    "impact",
    "experience",
    "schedule_stability",
    "matchups",
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + ".pending")
    with q.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    q.replace(p)
    fd = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(fd)
    os.close(fd)


def save_npz(path, **arrays):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(".pending")
    with q.open("wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    q.replace(p)


def code_identity():
    return {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA mandatory; CPU fitting forbidden")
    return torch.device("cuda:0")


def weights(keys):
    from collections import Counter

    c = Counter(k["event_id"] for k in keys)
    return np.asarray([1 / len(c) / c[k["event_id"]] for k in keys], dtype=np.float64)


def loss(p, y, keys):
    good = np.isfinite(y)
    p = np.asarray(p)[good]
    y = np.asarray(y)[good]
    keys = [k for k, g in zip(keys, good) if g]
    if not len(y):
        return {"brier": None, "log_loss": None}
    w = weights(keys)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "brier": float(w @ ((p - y) ** 2)),
        "log_loss": float(w @ (-y * np.log(p) - (1 - y) * np.log1p(-p))),
    }


def check(path, h):
    if digest(path) != h:
        raise RuntimeError("Artifact identity changed: " + str(path))


def verify_files(base, files):
    for n, h in files.items():
        p = Path(n)
        if p.is_absolute() or ".." in p.parts:
            raise RuntimeError("Invalid artifact path")
        check(Path(base) / p, h)


def files(base):
    return {
        str(p.relative_to(base)): digest(p)
        for p in sorted(Path(base).rglob("*"))
        if p.is_file() and p.name != "prepared.json"
    }


def attempt(directory, identity):
    p = Path(directory)
    p.mkdir(parents=True, exist_ok=True)
    i = 1
    while (p / f"attempt-{i}.json").exists():
        i += 1
    r = {"identity": identity, "started": time.time(), "status": "running"}
    save_json(p / f"attempt-{i}.json", r)
    return i, r
