"""One-shot source-bound snapshots; no evaluation labels."""

import json, math, shutil
from pathlib import Path
import numpy as np
from common import *


def prepare(root, parent):
    root = Path(root)
    parent = Path(parent)
    check(root / "config.json", read(root / "release-binding.json")["config_sha256"])
    out = root / "inputs"
    if (out / "prepared.json").exists():
        r = read(out / "prepared.json")
        verify_files(out, r["files"])
        return r
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Interrupted preparation preserved; version repair required")
    out.mkdir(parents=True, exist_ok=True)
    r = read(parent / "inputs/prepared.json")
    verify_files(parent / "inputs", r["files"])
    # Decide stopping/selection membership using outcome-free metadata, before labels.
    allkeys = read(parent / "inputs/map_winner/validation-keys.json")
    start = {}
    end = {}
    for k in allkeys:
        e = k["event_id"]
        start[e] = min(start.get(e, float("inf")), k["prediction_cutoff"])
        end[e] = max(end.get(e, 0), k["event_completion"] + 7200)
    events = sorted(start, key=lambda e: (start[e], e))
    cut = max(1, int(0.6 * len(events))) if events else 0
    later = set(events[cut:])
    earlier = set(events[:cut])
    boundary = min((start[e] for e in later), default=float("inf"))
    excluded = {
        e: "stopping_event_extent_overlaps_selection_cutoff"
        for e in earlier
        if end[e] >= boundary
    }
    earlier -= set(excluded)
    save_json(
        root / "partition.json",
        {
            "parent_sha256": digest(parent / "partition.json"),
            "stopping": sorted(earlier),
            "selection": sorted(later),
            "excluded": excluded,
            "outcome_free": True,
        },
    )
    meta = {}
    counts = {}
    for task in TASKS:
        old = r["tasks"][task]
        names = old["feature_names"]
        b = set(old["context_columns"])
        groups = {}
        fam = {}
        kept = []
        for j, n in enumerate(names):
            matches = [f for f, v in old["family_columns"].items() if j in v]
            if j in b:
                if matches:
                    raise RuntimeError("Context field also assigned to numeric family")
                semantic = "context"
            else:
                if len(matches) != 1 or matches[0] not in SEMANTIC:
                    raise RuntimeError(
                        "Unknown or multiply assigned source field: " + n
                    )
                semantic = matches[0]
            if "numeric_coverage" in n or "all_missing" in n:
                continue
            group = "B" if j in b else "L" if "matched_" in n else "N"
            groups[j] = group
            fam[j] = semantic
            kept.append(j)
        newnames = [names[j] for j in kept]
        newgroups = [groups[j] for j in kept]
        newfam = [fam[j] for j in kept]
        remap = {j: i for i, j in enumerate(kept)}
        swap = [remap[old["swap_numeric_indices"][j]] for j in kept]
        sign = [old["swap_numeric_signs"][j] for j in kept]
        cover = []
        for f in SEMANTIC:
            for group in ("N", "L"):
                cols = [j for j in kept if groups[j] == group and fam[j] == f]
                if not cols:
                    continue
                ix = len(newnames)
                newnames.extend(
                    [f + "_" + group + "_coverage_a", f + "_" + group + "_coverage_b"]
                )
                newgroups.extend([group, group])
                newfam.extend([f, f])
                swap.extend([ix + 1, ix])
                sign.extend([1, 1])
                cover.append(
                    (
                        [j for j in cols if names[j].endswith("_a")],
                        [j for j in cols if names[j].endswith("_b")],
                    )
                )
        for phase in PHASES:
            orig = "validation" if phase in ("stopping", "selection") else phase
            folder = parent / "inputs" / task
            keys = read(folder / f"{orig}-keys.json")
            ix = [
                i
                for i, k in enumerate(keys)
                if phase not in ("stopping", "selection")
                or k["event_id"] in (earlier if phase == "stopping" else later)
            ]
            keys = [dict(keys[i], phase=phase) for i in ix]
            for k in keys:
                if (
                    k["history_end_timestamp"] is not None
                    and k["history_end_timestamp"] >= k["prediction_cutoff"]
                ):
                    raise RuntimeError("Historical cutoff violation")
                if phase != "evaluation" and k["region"] == "global":
                    raise RuntimeError("Global row in fitting phase")
            with np.load(folder / f"{orig}.npz", allow_pickle=False) as z:
                numeric = z["numeric"][ix]
                x = numeric[:, kept]
                cc = []
                for ca, cb in cover:
                    cc.extend(
                        [
                            (
                                np.isfinite(numeric[:, ca]).mean(1)
                                if ca
                                else np.zeros(len(ix))
                            ),
                            (
                                np.isfinite(numeric[:, cb]).mean(1)
                                if cb
                                else np.zeros(len(ix))
                            ),
                        ]
                    )
                if cc:
                    x = np.column_stack([x, *cc])
                comp = z["composition"][ix]
            dest = out / task
            dest.mkdir(exist_ok=True)
            save_npz(
                dest / f"{phase}.npz", numeric=x.astype("float32"), composition=comp
            )
            save_json(dest / f"{phase}-keys.json", keys)
            if phase != "evaluation":
                np.save(
                    dest / f"{phase}-labels.npy",
                    np.load(folder / f"{orig}-labels.npy", allow_pickle=False)[ix],
                )
            counts[task, phase] = (len(keys), len({k["event_id"] for k in keys}))
        meta[task] = {
            "feature_names": newnames,
            "groups": newgroups,
            "semantic": newfam,
            "swap_numeric_indices": swap,
            "swap_numeric_signs": sign,
            "composition_width": old["composition_width"],
            "swap_composition_indices": old["swap_composition_indices"],
            "swap_composition_signs": old["swap_composition_signs"],
            "old_names": names,
            "field_dispositions": [
                {
                    "index": j,
                    "name": n,
                    "disposition": (
                        "omit_redundant_coverage"
                        if "numeric_coverage" in n or "all_missing" in n
                        else groups[j]
                    ),
                    "family": fam.get(j),
                }
                for j, n in enumerate(names)
            ],
        }
    # Player snapshots provide genuine historical-agent-conditioned series L;
    # old series "matched" channels were aliases of overall profiles.
    from player_snapshot import prepare_players

    prepare_players(root, parent, r, meta)
    rnew = {
        "tasks": meta,
        "files": files(out),
        "parent_prepared_sha256": digest(parent / "inputs/prepared.json"),
        "parent_root": str(parent.resolve()),
        "chronology": r["chronology_audit"],
        "sources": r["sources"],
        "counts": {t: {p: counts[t, p] for p in PHASES} for t in TASKS},
        "evaluation_labels_read": False,
        "format": "omitted; no timestamped pre-play availability contract",
        "family_coverage": "recomputed separately for N and L; no selected-agent missingness in N",
    }
    save_json(out / "prepared.json", rnew)
    return rnew


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--block", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    a = p.parse_args()
    prepare(a.block, a.parent)
