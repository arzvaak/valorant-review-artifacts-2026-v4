"""Finite search, three-seed packages, calibration and separate research stages."""

import argparse, gc, itertools, time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from common import *
from preprocess import arrays
from models import candidate


def presets(family):
    if family == "linear":
        return [
            {"penalty": p}
            for p in (1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
        ]
    if family == "catboost":
        return [{"depth": d, "l2": r} for d in (4, 6) for r in (3, 10)]
    if family == "xgboost":
        return [{"depth": d, "min_child": r} for d in (3, 6) for r in (10, 50)]
    if family == "mlp":
        return [{"width": w, "dropout": d} for w in (128, 256) for d in (0.1, 0.2)]
    if family == "tabm":
        return [
            {"blocks": b, "width": w, "dropout": d}
            for b, w in ((2, 128), (3, 256))
            for d in (0.1, 0.2)
        ]
    if family == "ft_transformer":
        return [
            {"blocks": b, "width": w, "dropout": d}
            for b, w in ((2, 128), (3, 192))
            for d in (0.1, 0.2)
        ]
    if family == "deep_sets":
        return [{"width": w, "dropout": d} for w in (128, 256) for d in (0.1, 0.2)]
    return [{"width": w, "dropout": d} for w in (64, 128) for d in (0.1, 0.2)]


def calibrated(pred, y, keys):
    if len({k["event_id"] for k in keys}) < 5 or len(np.unique(y)) < 2:
        return 1.0, {
            "method": "identity",
            "reason": "insufficient_calibration_events_or_classes",
        }
    z = torch.logit(
        torch.tensor(pred, device=cuda(), dtype=torch.float64).clamp(1e-7, 1 - 1e-7)
    )
    target = torch.tensor(y, device=cuda(), dtype=torch.float64)
    w = torch.tensor(weights(keys), device=cuda(), dtype=torch.float64)
    theta = torch.tensor(
        -2.1972245773362196, device=cuda(), dtype=torch.float64, requires_grad=True
    )
    op = torch.optim.LBFGS(
        [theta],
        max_iter=1000,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure():
        op.zero_grad()
        v = (
            F.binary_cross_entropy_with_logits(
                10 * theta.sigmoid() * z, target, reduction="none"
            )
            * w
        ).sum()
        if not torch.isfinite(v):
            raise RuntimeError("Nonfinite calibration objective")
        v.backward()
        return v

    op.step(closure)
    rows = []
    with torch.no_grad():
        for scale in (0.0, 1.0, float(10 * theta.sigmoid())):
            rows.append(
                {
                    "scale": scale,
                    "log_loss": float(
                        (
                            F.binary_cross_entropy_with_logits(
                                scale * z, target, reduction="none"
                            )
                            * w
                        ).sum()
                    ),
                }
            )
    chosen = min(rows, key=lambda x: (x["log_loss"], abs(x["scale"] - 1)))
    return chosen["scale"], {
        "method": "separate-event CUDA scalar logit calibration",
        "candidates": rows,
        "selected": chosen,
    }


def scale_probs(p, scale):
    z = torch.logit(
        torch.tensor(p, device=cuda(), dtype=torch.float64).clamp(1e-7, 1 - 1e-7)
    )
    return (z * scale).sigmoid().float().cpu().numpy()


def baselines(block, task):
    pred = {}
    for p in PHASES:
        with np.load(block / "inputs" / task / f"{p}.npz", allow_pickle=False) as z:
            names = read(block / "inputs/prepared.json")["tasks"][task]["feature_names"]
            l = z["numeric"][:, names.index("elo_log_odds_difference")]
            l = np.nan_to_num(l)
            pred[p] = 1 / (1 + np.exp(-np.clip(l, -30, 30)))
    return pred


def map_baseline(block):
    names = read(block / "inputs/prepared.json")["tasks"]["map_winner"]["feature_names"]
    result = {}
    for phase in PHASES:
        with np.load(
            block / "inputs/map_winner" / f"{phase}.npz", allow_pickle=False
        ) as z:
            logits = np.nan_to_num(
                z["numeric"][:, names.index("map_history_log_odds_difference")]
            )
            result[phase] = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
    return result


def eligible(data, keys, family=None):
    if family == "linear":
        return (
            bool(data)
            and len(data["train"]["x"]) > 0
            and len(np.unique(data["train"]["y"].cpu().numpy())) == 2
            and len(keys["selection"]) > 0
            and len(np.unique(data["selection"]["y"].cpu().numpy())) == 2
        )
    return (
        bool(data)
        and len(data["train"]["x"]) > 0
        and len(np.unique(data["train"]["y"].cpu().numpy())) == 2
        and all(
            len({k["event_id"] for k in keys[p]}) >= 5
            and len(np.unique(data[p]["y"].cpu().numpy())) == 2
            for p in ("stopping", "selection")
        )
    )


def stochastic_eligible(block, task):
    return all(
        len({k["event_id"] for k in read(block / "inputs" / task / f"{p}-keys.json")})
        >= 5
        and len(
            np.unique(
                np.load(block / "inputs" / task / f"{p}-labels.npy", allow_pickle=False)
            )
        )
        == 2
        for p in ("stopping", "selection")
    )


def aggregate_players(data, proc):
    width = len(proc["swap_indices"])
    pw = data["train"]["players"].shape[-1]
    extra = pw * 4
    for phase, v in data.items():
        p = v.pop("players")
        summ = (
            torch.cat(
                (
                    p.mean(2),
                    p.std(2, unbiased=False),
                    p.max(2).values,
                    p.topk(2, dim=2).values.mean(2),
                ),
                2,
            ).reshape(len(p), 2 * extra)
            if len(p)
            else p.new_empty((0, 2 * extra))
        )
        v["x"] = torch.cat((v["x"], summ), 1)
    proc["swap_indices"] += list(range(width + extra, width + 2 * extra)) + list(
        range(width, width + extra)
    )
    proc["swap_signs"] += [1.0] * (2 * extra)
    proc["aggregate_control"] = (
        "mean/std/max/top-two of same imputed player tensors and masks"
    )
    return data, proc


def package(
    block,
    task,
    rep,
    family,
    cfg,
    stage="core",
    drop=None,
    coverage=False,
    params_fixed=None,
):
    block = Path(block)
    suffix = rep + ("-drop_" + drop if drop else "") + ("-coverage" if coverage else "")
    directory = block / "packages" / stage / task / f"{suffix}-{family}"
    directory.mkdir(parents=True, exist_ok=True)
    identity = {
        "release": read(block / "release-binding.json"),
        "prepared": digest(block / "inputs/prepared.json"),
        "task": task,
        "representation": rep,
        "drop": drop,
        "coverage_only": coverage,
        "family": family,
        "stage": stage,
        "code": code_identity(),
        "fixed_parameters": params_fixed,
    }
    marker = directory / "frozen.json"
    if marker.exists():
        r = read(marker)
        if r["identity"] != identity:
            raise RuntimeError("Package identity changed")
        verify_files(directory, r["files"])
        return r
    mode = ("full" if rep == "BNLC" else "general") if stage == "players" else None
    from recovery import reuse

    reused = reuse(directory, identity, cfg, "package")
    if reused is not None:
        return reused
    data, keys, proc = arrays(block, task, rep, drop, coverage, mode)
    if stage == "players" and family in ("mlp", "tabm") and data:
        data, proc = aggregate_players(data, proc)
    base = baselines(block, task)
    records = []
    fallback = not eligible(data, keys, family)
    raw = base
    selected = None
    if not fallback:
        seeds = cfg["seeds"]
        grid = [params_fixed] if params_fixed is not None else presets(family)
        for i, parameters in enumerate(grid):
            path = directory / "candidates" / f"config-{i}-{seeds[0]}"
            rec = candidate(
                data, keys, proc, path, family, parameters, seeds[0], cfg, identity
            )
            records.append({"path": str(path), "record": rec, "config_id": i})
        fitted = [r for r in records if r["record"]["status"] == "fitted"]
        if fitted:
            selected = min(
                fitted,
                key=lambda r: (
                    r["record"]["selection"]["brier"],
                    r["record"]["selection"]["log_loss"],
                    r["config_id"],
                ),
            )
            params = selected["record"]["identity"]["parameters"]
            chosen = [selected]
            if family != "linear":
                for seed in seeds[1:]:
                    path = (
                        directory
                        / "candidates"
                        / f"selected-{selected['config_id']}-{seed}"
                    )
                    rec = candidate(
                        data, keys, proc, path, family, params, seed, cfg, identity
                    )
                    item = {
                        "path": str(path),
                        "record": rec,
                        "config_id": selected["config_id"],
                    }
                    records.append(item)
                    chosen.append(item)
            if all(r["record"]["status"] == "fitted" for r in chosen):
                predictions = []
                for r in chosen:
                    with np.load(
                        Path(r["path"]) / "predictions.npz", allow_pickle=False
                    ) as z:
                        predictions.append({p: z[p].copy() for p in PHASES})
                raw = {p: np.mean([z[p] for z in predictions], 0) for p in PHASES}
            else:
                fallback = True
        else:
            fallback = True
    scale, cal = calibrated(
        raw["calibration"],
        np.load(block / "inputs" / task / "calibration-labels.npy", allow_pickle=False),
        keys["calibration"],
    )
    save_npz(
        directory / "predictions.npz",
        **{p + "_raw": v for p, v in raw.items()},
        **{p + "_calibrated": scale_probs(v, scale) for p, v in raw.items()},
    )
    save_json(directory / "processor.json", proc)
    save_json(
        directory / "selection.json",
        {
            "selected": selected,
            "candidates": records,
            "fallback": fallback,
            "reason": (
                "insufficient_history"
                if not eligible(data, keys, family)
                else "candidate_or_selected_seed_failure" if fallback else None
            ),
            "calibrator": cal,
        },
    )
    r = {
        "identity": identity,
        "files": {
            "predictions.npz": digest(directory / "predictions.npz"),
            "processor.json": digest(directory / "processor.json"),
            "selection.json": digest(directory / "selection.json"),
        },
        "selection": loss(
            raw["selection"],
            np.load(
                block / "inputs" / task / "selection-labels.npy", allow_pickle=False
            ),
            keys["selection"],
        ),
        "trained": not fallback,
        "model_records": records,
        "directory": str(directory),
        "evaluation_labels_read": False,
    }
    save_json(marker, r)
    del data
    gc.collect()
    torch.cuda.empty_cache()
    return r


def main(block, stage):
    block = Path(block)
    check(block / "config.json", read(block / "release-binding.json")["config_sha256"])
    cfg = read(block / "config.json")
    receipt = read(block / "inputs/prepared.json")
    verify_files(block / "inputs", receipt["files"])
    cuda()
    torch.set_num_threads(4)
    completed = []
    if stage == "core":
        units = [(t, r, f, None, None) for t in TASKS for r in REPS for f in FAMILIES]
    elif stage == "players":
        units = [
            (t, r, f, None, None)
            for t in TASKS
            for r in ("BN", "BNLC")
            for f in ("deep_sets", "set_transformer", "mlp", "tabm")
        ]
    else:
        units = []
        for t in TASKS:
            candidates = [
                read(block / "packages/core" / t / f"BNLC-{f}/frozen.json")
                for f in FAMILIES
            ]
            available = [r for r in candidates if r["trained"]]
            if not stochastic_eligible(block, t):
                available = []
            if not available:
                save_json(
                    block / f"{stage}-{t}-unavailable.json",
                    {"reason": "no_fitted_full_reference"},
                )
                continue
            best = min(
                available,
                key=lambda r: (
                    r["selection"]["brier"],
                    r["selection"]["log_loss"],
                    r["identity"]["family"],
                ),
            )
            sel = read(Path(best["directory"]) / "selection.json")["selected"]
            params = sel["record"]["identity"]["parameters"]
            family = best["identity"]["family"]
            save_json(
                block / f"{stage}-{t}-parent.json",
                {
                    "directory": best["directory"],
                    "freeze_sha256": digest(Path(best["directory"]) / "frozen.json"),
                    "family": family,
                    "params": params,
                },
            )
            units += [(t, "BNLC", family, f, params) for f in SEMANTIC] + [
                (t, "BNLC", family, "coverage", params)
            ]
    for i, (task, rep, fam, drop, params) in enumerate(units):
        save_json(
            block / "progress.json",
            {
                "stage": stage,
                "task": task,
                "representation": rep,
                "family": fam,
                "completed": i,
                "total": len(units),
            },
        )
        print("PACKAGE", block.name, stage, task, rep, fam, drop, flush=True)
        rec = package(
            block,
            task,
            rep,
            fam,
            cfg,
            stage,
            drop if drop != "coverage" else None,
            drop == "coverage",
            params,
        )
        path = Path(rec["directory"]) / "frozen.json"
        completed.append({"path": str(path), "sha256": digest(path)})
    save_json(
        block / f"{stage}-frozen.json",
        {
            "packages": completed,
            "evaluation_labels_read": False,
            "prepared_sha256": digest(block / "inputs/prepared.json"),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--block", type=Path, required=True)
    p.add_argument("--stage", choices=("core", "players", "removals"), required=True)
    a = p.parse_args()
    main(a.block, a.stage)
