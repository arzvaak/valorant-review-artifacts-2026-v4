"""Selection-event-only finite probability ensembles; separate calibration."""

import itertools
from pathlib import Path
import numpy as np
from common import *
from fit import baselines, calibrated, scale_probs, stochastic_eligible


def select_and_freeze(block, task):
    block = Path(block)
    check(block / "config.json", read(block / "release-binding.json")["config_sha256"])
    out = block / "selector" / task
    out.mkdir(parents=True, exist_ok=True)
    receipt = {
        "core_freeze": digest(block / "core-frozen.json"),
        "prepared": digest(block / "inputs/prepared.json"),
        "code": code_identity(),
    }
    if (out / "frozen.json").exists():
        r = read(out / "frozen.json")
        if r["identity"] != receipt:
            raise RuntimeError("Selector identity changed")
        verify_files(out, r["files"])
        return r
    keys = {p: read(block / "inputs" / task / f"{p}-keys.json") for p in PHASES}
    ys = np.load(block / "inputs" / task / "selection-labels.npy", allow_pickle=False)
    records = [
        read(block / "packages/core" / task / f"{rep}-{f}/frozen.json")
        for rep in REPS
        for f in FAMILIES
    ]
    eligible = (
        [r for r in records if r["trained"]] if stochastic_eligible(block, task) else []
    )
    ranked = sorted(
        eligible,
        key=lambda r: (
            r["selection"]["brier"],
            r["selection"]["log_loss"],
            r["identity"]["representation"],
            r["identity"]["family"],
        ),
    )[:3]
    forecasts = {"elo_k20": baselines(block, task)}
    for r in ranked:
        name = r["identity"]["representation"] + "-" + r["identity"]["family"]
        with np.load(Path(r["directory"]) / "predictions.npz", allow_pickle=False) as z:
            forecasts[name] = {p: z[p + "_raw"] for p in PHASES}
    candidates = []
    for n, pred in forecasts.items():
        candidates.append(
            {
                "components": {n: 1.0},
                "id": n,
                "loss": loss(pred["selection"], ys, keys["selection"]),
            }
        )
    for a, b in itertools.combinations(forecasts, 2):
        for w in (0.25, 0.5, 0.75):
            candidates.append(
                {
                    "components": {a: w, b: 1 - w},
                    "id": a + "+" + b + "-" + str(w),
                    "loss": loss(
                        w * forecasts[a]["selection"]
                        + (1 - w) * forecasts[b]["selection"],
                        ys,
                        keys["selection"],
                    ),
                }
            )
    chosen = (
        min(
            candidates,
            key=lambda r: (r["loss"]["brier"], r["loss"]["log_loss"], r["id"]),
        )
        if len(ys) and ranked
        else candidates[0]
    )
    raw = {
        p: sum(w * forecasts[n][p] for n, w in chosen["components"].items())
        for p in PHASES
    }
    yc = np.load(block / "inputs" / task / "calibration-labels.npy", allow_pickle=False)
    scale, cal = calibrated(raw["calibration"], yc, keys["calibration"])
    elo = forecasts["elo_k20"]
    es, ec = calibrated(elo["calibration"], yc, keys["calibration"])
    ytrain = np.load(block / "inputs" / task / "train-labels.npy", allow_pickle=False)
    prior = float(weights(keys["train"]) @ ytrain) if len(ytrain) else 0.5
    extra = {}
    if task == "map_winner":
        from fit import map_baseline

        mp = map_baseline(block)
        ms, mc = calibrated(mp["calibration"], yc, keys["calibration"])
        extra = {
            "evaluation_same_map_raw": mp["evaluation"],
            "evaluation_same_map_calibrated": scale_probs(mp["evaluation"], ms),
        }
    save_npz(
        out / "predictions.npz",
        **extra,
        **{p + "_raw": v for p, v in raw.items()},
        **{p + "_calibrated": scale_probs(v, scale) for p, v in raw.items()},
        **{p + "_elo_raw": v for p, v in elo.items()},
        **{p + "_elo_calibrated": scale_probs(v, es) for p, v in elo.items()},
        chance=np.full(len(keys["evaluation"]), 0.5),
        training_prior=np.full(len(keys["evaluation"]), prior),
    )
    save_json(
        out / "selection.json",
        {
            "chosen": chosen,
            "candidates": candidates,
            "ranked": [
                {k: r[k] for k in ("identity", "selection", "directory")}
                for r in ranked
            ],
            "calibrator": cal,
            "elo_calibrator": ec,
            "sparse_fixed_fallback": not bool(ranked),
        },
    )
    r = {
        "identity": receipt,
        "files": {
            "predictions.npz": digest(out / "predictions.npz"),
            "selection.json": digest(out / "selection.json"),
        },
        "directory": str(out),
        "evaluation_labels_read": False,
    }
    save_json(out / "frozen.json", r)
    return r


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--block", type=Path, required=True)
    p.add_argument("--task", choices=TASKS, required=True)
    a = p.parse_args()
    select_and_freeze(a.block, a.task)
