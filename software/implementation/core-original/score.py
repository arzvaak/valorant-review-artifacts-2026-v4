"""Identity-bound post-freeze scoring and complete-event CUDA uncertainty."""

import argparse, collections, datetime, itertools
from pathlib import Path
import numpy as np, torch
from scipy.stats import beta
from sklearn.metrics import roc_auc_score
from common import *


def verify_stage(root, stage):
    identity = read(root / "release-identity.json")
    verify_files(root / "code", identity["code"])
    check(root / "config.json", identity["config_sha256"])
    freeze = read(root / f"{stage}-all-predictions-frozen.json")
    if freeze["release_sha256"] != digest(root / "release-identity.json"):
        raise RuntimeError("Whole-stage release mismatch")
    cfg = read(root / "config.json")
    if [r["block"] for r in freeze["blocks"]] != [r["name"] for r in cfg["folds"]]:
        raise RuntimeError("Incomplete whole-stage freeze")
    for row in freeze["blocks"]:
        block = root / "blocks" / row["block"]
        check(block / "config.json", identity["config_sha256"])
        file = block / f"{stage}-frozen.json"
        check(file, row["core_sha256"] if stage == "core" else row["sha256"])
        r = read(file)
        check(block / "inputs/prepared.json", r["prepared_sha256"])
        verify_files(block / "inputs", read(block / "inputs/prepared.json")["files"])
        for p in r["packages"]:
            check(p["path"], p["sha256"])
            rec = read(p["path"])
            verify_files(Path(rec["directory"]), rec["files"])
            for candidate in rec["model_records"]:
                c = candidate["record"]
                if c["status"] == "fitted":
                    check(c["model"], c["model_sha256"])
                    check(
                        Path(candidate["path"]) / "predictions.npz",
                        c["prediction_sha256"],
                    )
                    check(
                        Path(candidate["path"]) / "processor.json",
                        c["processor_sha256"],
                    )
        if stage == "core":
            for t, h in row["selectors"].items():
                s = block / "selector" / t
                check(s / "frozen.json", h)
                verify_files(s, read(s / "frozen.json")["files"])
    import legacy_prepare

    legacy_prepare.verify_registry(root)
    return cfg


def metric(p, y, keys):
    w = weights(keys)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    pred = p > 0.5
    ll = -y * np.log(p) - (1 - y) * np.log1p(-p)
    result = {
        "rows": len(y),
        "events": len({k["event_id"] for k in keys}),
        "matches": len({k["match_id"] for k in keys}),
        "brier": float(w @ ((p - y) ** 2)),
        "log_loss": float(w @ ll),
        "row_brier": float(np.mean((p - y) ** 2)),
        "row_log_loss": float(ll.mean()),
        "accuracy": float(w @ (pred == y)),
        "row_accuracy": float(np.mean(pred == y)),
        "balanced_accuracy": (
            float(np.mean([np.mean(pred[y == v] == v) for v in (0, 1)]))
            if len(np.unique(y)) == 2
            else None
        ),
        "roc_auc": (
            float(roc_auc_score(y, p, sample_weight=w))
            if len(np.unique(y)) == 2
            else None
        ),
        "orientation_team_a_fraction": float(w @ y),
    }
    for bins in (5, 10, 20):
        ids = np.minimum((p * bins).astype(int), bins - 1)
        result[f"ece_{bins}"] = float(
            sum(abs(w[ids == i] @ (p[ids == i] - y[ids == i])) for i in range(bins))
        )
    if len(np.unique(y)) == 2:
        z = torch.logit(torch.tensor(p, device=cuda(), dtype=torch.float64))
        yy = torch.tensor(y, device=cuda(), dtype=torch.float64)
        ww = torch.tensor(w, device=cuda(), dtype=torch.float64)
        a = torch.tensor(
            [0.0, 1.0], device=cuda(), dtype=torch.float64, requires_grad=True
        )
        op = torch.optim.LBFGS(
            [a], max_iter=500, line_search_fn="strong_wolfe", tolerance_grad=1e-9
        )

        def closure():
            op.zero_grad()
            l = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    a[0] + a[1] * z, yy, reduction="none"
                )
                * ww
            ).sum()
            l.backward()
            return l

        op.step(closure)
        aa = a.detach().cpu().tolist()
        result.update(calibration_intercept=aa[0], calibration_slope=aa[1])
    return result


def adjustment(rows):
    for family in sorted({r["correction_family"] for r in rows}):
        group = [r for r in rows if r["correction_family"] == family]
        n = len(group)
        for field, output in (
            ("p_approx", "holm_p"),
            ("mc_upper_p", "holm_mc_upper_p"),
        ):
            order = sorted(group, key=lambda r: (r.get(field, 1.0), r["id"]))
            previous = 0.0
            for i, r in enumerate(order):
                previous = max(previous, min(1.0, (n - i) * r.get(field, 1.0)))
                r[output] = previous
        for r in group:
            r["conservative_rejection"] = r["holm_mc_upper_p"] <= 0.05


def contrasts_for(stage, task):
    out = []

    def add(family, a, b, name):
        for lossname in ("brier", "log_loss"):
            out.append(
                {
                    "task": task,
                    "correction_family": family,
                    "a": a,
                    "b": b,
                    "metric": lossname,
                    "id": task + "/" + name + "/" + lossname,
                }
            )

    if stage == "core":
        for f in FAMILIES:
            for a, b in (
                ("BN", "B"),
                ("BC", "B"),
                ("BNC", "BN"),
                ("BNC", "BC"),
                ("BN", "BC"),
            ):
                add("primary120", a + "-" + f, b + "-" + f, f + "/" + a + "-" + b)
            for a, b in (("BNL", "BN"), ("BNLC", "BNC")):
                add("linked48", a + "-" + f, b + "-" + f, f + "/" + a + "-" + b)
        for parent in ("elo_k20", "B-linear", "B-tabm"):
            add("operational12", "selector", parent, "selector-" + parent)
    elif stage == "players":
        for r in ("BN", "BNLC"):
            for f in ("deep_sets", "set_transformer"):
                for control in ("mlp", "tabm"):
                    add(
                        "players48",
                        r + "-" + f,
                        r + "-" + control,
                        r + "/" + f + "-" + control,
                    )
        for f in ("deep_sets", "set_transformer"):
            add("players48", "BNLC-" + f, "selector", f + "-selector")
            add("players48", "BNLC-" + f, "BN-" + f, f + "/full-general")
    else:
        for f in SEMANTIC:
            add("removals40", "drop_" + f, "full_reference", f + "-full")
        add("removals40", "coverage", "full_reference", "coverage-full")
    return out


def event_losses(p, y, keys, metricname):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    v = (
        (p - y) ** 2
        if metricname == "brier"
        else -y * np.log(p) - (1 - y) * np.log1p(-p)
    )
    groups = collections.defaultdict(list)
    for i, k in enumerate(keys):
        groups[k["event_id"]].append(i)
    return {e: float(v[ix].mean()) for e, ix in groups.items()}


def paired(row, pred, y, keys, draws, family_size, seed):
    a = event_losses(pred[row["a"]], y, keys, row["metric"])
    b = event_losses(pred[row["b"]], y, keys, row["metric"])
    times = {}
    for k in keys:
        times[k["event_id"]] = min(
            times.get(k["event_id"], float("inf")), k["prediction_cutoff"]
        )
    events = sorted(a, key=lambda e: (times[e], e))
    delta = np.asarray([a[e] - b[e] for e in events])
    point = float(delta.mean())
    n = len(delta)
    d = torch.tensor(delta, device=cuda(), dtype=torch.float64)
    gen = torch.Generator(device=cuda()).manual_seed(seed)
    samples = []
    for start in range(0, draws, 256):
        ix = torch.randint(
            n, (min(256, draws - start), n), device=cuda(), generator=gen
        )
        samples.append(d[ix].mean(1))
    boot = torch.cat(samples)
    ci = (
        torch.quantile(
            boot, torch.tensor([0.025, 0.975], device=cuda(), dtype=torch.float64)
        )
        .cpu()
        .tolist()
    )
    extreme = int(((boot - point).abs() >= abs(point)).sum())
    upper = (
        float(beta.ppf(1 - 0.01 / family_size, extreme + 1, draws - extreme))
        if extreme < draws
        else 1.0
    )
    row.update(
        difference=point,
        ci95=ci,
        rows=len(y),
        events=n,
        draws=draws,
        p_approx=(extreme + 1) / (draws + 1),
        mc_upper_p=upper,
        mc_simultaneous_confidence=0.99,
        extreme_count=extreme,
        seed=seed,
        interpretation="negative difference favors a; intervals conditional on fitted forecasts",
        year_omissions={},
        moving_blocks={},
    )
    years = {
        e: datetime.datetime.fromtimestamp(times[e], datetime.timezone.utc).year
        for e in events
    }
    for yr in sorted(set(years.values())):
        remain = [i for i, e in enumerate(events) if years[e] != yr]
        row["year_omissions"][str(yr)] = {
            "events": len(remain),
            "difference": float(delta[remain].mean()) if remain else None,
        }
    for length in (10, 25, 50):
        if n < length:
            row["moving_blocks"][str(length)] = {"status": "insufficient_events"}
            continue
        boot = []
        blocks = (n + length - 1) // length
        offset = torch.arange(length, device=cuda())
        gen.manual_seed(seed + length)
        for start in range(0, 8192, 128):
            heads = torch.randint(n, (128, blocks), device=cuda(), generator=gen)
            ix = ((heads[..., None] + offset) % n).reshape(128, -1)[:, :n]
            boot.append(d[ix].mean(1))
        interval = (
            torch.quantile(
                torch.cat(boot),
                torch.tensor([0.025, 0.975], device=cuda(), dtype=torch.float64),
            )
            .cpu()
            .tolist()
        )
        row["moving_blocks"][str(length)] = {
            "draws": 8192,
            "ci95": interval,
            "method": "circular moving complete-event blocks; descriptive",
        }
    return row


def main(root, stage):
    root = Path(root)
    cfg = verify_stage(root, stage)
    out = root / "report" / stage
    out.mkdir(parents=True, exist_ok=True)
    allmetrics = []
    allcontrasts = []
    failures = []
    seedmetrics = []
    audit = []
    for task in TASKS:
        pooled = collections.defaultdict(list)
        raw = collections.defaultdict(list)
        allkeys = []
        ally = []
        seedpooled = collections.defaultdict(list)
        for fold in cfg["folds"]:
            block = root / "blocks" / fold["name"]
            keys = read(block / "inputs" / task / "evaluation-keys.json")
            cache = root / "scored-labels" / fold["name"] / f"{task}.npy"
            if cache.exists():
                receipt = read(cache.with_suffix(".json"))
                check(cache, receipt["sha256"])
                if receipt["keys_sha256"] != digest(
                    block / "inputs" / task / "evaluation-keys.json"
                ) or receipt["release_sha256"] != digest(
                    root / "release-identity.json"
                ):
                    raise RuntimeError("Label cache identity mismatch")
                y = np.load(cache, allow_pickle=False)
            else:
                if stage != "core":
                    raise RuntimeError(
                        "Secondary stage requires core target projection"
                    )
                import legacy_prepare

                y = legacy_prepare.project_labels(block, task, keys)
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, y)
                save_json(
                    cache.with_suffix(".json"),
                    {
                        "sha256": digest(cache),
                        "keys_sha256": digest(
                            block / "inputs" / task / "evaluation-keys.json"
                        ),
                        "release_sha256": digest(root / "release-identity.json"),
                        "whole_programme_freeze_sha256": digest(
                            root / "core-all-predictions-frozen.json"
                        ),
                    },
                )
            with np.load(
                block / "selector" / task / "predictions.npz", allow_pickle=False
            ) as z:
                local = {
                    "selector": z["evaluation_calibrated"],
                    "elo_k20": z["evaluation_elo_calibrated"],
                    "chance": z["chance"],
                    "training_prior": z["training_prior"],
                }
                localraw = {
                    "selector": z["evaluation_raw"],
                    "elo_k20": z["evaluation_elo_raw"],
                    "chance": z["chance"],
                    "training_prior": z["training_prior"],
                }
                if task == "map_winner":
                    local["same_map"] = z["evaluation_same_map_calibrated"]
                    localraw["same_map"] = z["evaluation_same_map_raw"]
            records = read(block / f"{stage}-frozen.json")["packages"]
            available = True
            if stage == "removals":
                parent = block / f"removals-{task}-parent.json"
                if not parent.exists():
                    available = False
                else:
                    rr = read(parent)
                    check(Path(rr["directory"]) / "frozen.json", rr["freeze_sha256"])
                    with np.load(
                        Path(rr["directory"]) / "predictions.npz", allow_pickle=False
                    ) as z:
                        local["full_reference"] = z["evaluation_calibrated"]
                        localraw["full_reference"] = z["evaluation_raw"]
            for record in records:
                rec = read(record["path"])
                identity = rec["identity"]
                if identity["task"] != task:
                    continue
                name = (
                    identity["representation"] + "-" + identity["family"]
                    if stage != "removals"
                    else (
                        "coverage"
                        if identity["coverage_only"]
                        else "drop_" + identity["drop"]
                    )
                )
                with np.load(
                    Path(rec["directory"]) / "predictions.npz", allow_pickle=False
                ) as z:
                    local[name] = z["evaluation_calibrated"]
                    localraw[name] = z["evaluation_raw"]
                failures.append(
                    {
                        "block": fold["name"],
                        "task": task,
                        "package": name,
                        "trained": rec["trained"],
                        "candidates": len(rec["model_records"]),
                        "failed_candidates": [
                            {"path": r["path"], "reason": r["record"]["reason"]}
                            for r in rec["model_records"]
                            if r["record"]["status"] == "failed"
                        ],
                    }
                )
                selected = read(Path(rec["directory"]) / "selection.json")["selected"]
                if rec["trained"] and selected:
                    selectedparams = selected["record"]["identity"]["parameters"]
                    for r in rec["model_records"]:
                        c = r["record"]
                        if (
                            c["status"] == "fitted"
                            and c["identity"]["parameters"] == selectedparams
                        ):
                            with np.load(
                                Path(r["path"]) / "predictions.npz", allow_pickle=False
                            ) as z:
                                seedpooled[
                                    name + "-seed-" + str(c["identity"]["seed"])
                                ].append((keys, y, z["evaluation"].copy()))
            if not available:
                audit.append(
                    {
                        "block": fold["name"],
                        "task": task,
                        "status": "diagnostic_unavailable",
                    }
                )
                continue
            if any(
                len(v) != len(keys) or not np.isfinite(v).all() for v in local.values()
            ):
                raise RuntimeError("Forecast membership or finiteness mismatch")
            if len(y) != len(keys):
                raise RuntimeError("Target membership mismatch")
            if any(int(k["team_a_id"]) >= int(k["team_b_id"]) for k in keys):
                raise RuntimeError("Target canonical orientation mismatch")
            for name, v in local.items():
                pooled[name].append(v)
                raw[name].append(localraw[name])
            allkeys.extend(keys)
            ally.append(y)
            audit.append(
                {
                    "block": fold["name"],
                    "task": task,
                    "frozen_rows": len(keys),
                    "valid_labels": int(np.isfinite(y).sum()),
                    "invalid_label_rows": int((~np.isfinite(y)).sum()),
                }
            )
        if not ally:
            continue
        y = np.concatenate(ally)
        pred = {n: np.concatenate(v) for n, v in pooled.items()}
        rawpred = {n: np.concatenate(v) for n, v in raw.items()}
        if len(
            {k["game_id"] if task == "map_winner" else k["match_id"] for k in allkeys}
        ) != len(allkeys):
            raise RuntimeError("Repeated held-forward target across outer blocks")
        scopes = {
            "broad_nonglobal": lambda k: k["region"] != "global",
            "reviewed_six_circuits": lambda k: k["region"]
            in ("americas", "emea", "pacific")
            and k["population"] in ("open", "game_changers"),
            "known_globals": lambda k: k["region"] == "global",
        }
        for group in sorted({k["group"] for k in allkeys}):
            scopes["group:" + group] = lambda k, g=group: k["group"] == g
        for scope, condition in scopes.items():
            mask = np.asarray([condition(k) for k in allkeys]) & np.isfinite(y)
            if not mask.any():
                continue
            kk = [k for k, g in zip(allkeys, mask) if g]
            yy = y[mask]
            for mode, forecasts in (("calibrated", pred), ("raw", rawpred)):
                for n, p in forecasts.items():
                    allmetrics.append(
                        {
                            "task": task,
                            "scope": scope,
                            "mode": mode,
                            "package": n,
                            **metric(p[mask], yy, kk),
                        }
                    )
            if scope == "broad_nonglobal":
                for row in contrasts_for(stage, task):
                    if row["a"] not in pred or row["b"] not in pred:
                        row.update(status="unavailable", difference=None)
                        allcontrasts.append(row)
                        continue
                    size = {
                        "primary120": 120,
                        "linked48": 48,
                        "operational12": 12,
                        "players48": 48,
                        "removals40": 40,
                    }[row["correction_family"]]
                    allcontrasts.append(
                        paired(
                            row,
                            {n: p[mask] for n, p in pred.items()},
                            yy,
                            kk,
                            cfg["bootstrap_draws"],
                            size,
                            20261009 + len(allcontrasts),
                        )
                    )
        for name, parts in seedpooled.items():
            kk = [k for keys, _, _ in parts for k in keys]
            yy = np.concatenate([y for _, y, _ in parts])
            pp = np.concatenate([p for _, _, p in parts])
            mask = np.isfinite(yy) & np.asarray([k["region"] != "global" for k in kk])
            kk = [k for k, g in zip(kk, mask) if g]
            if mask.any():
                seedmetrics.append(
                    {
                        "task": task,
                        "package_seed": name,
                        "scope": "broad_nonglobal",
                        "mode": "raw seed diagnostic",
                        **metric(pp[mask], yy[mask], kk),
                    }
                )
    adjustment([r for r in allcontrasts if r.get("status") != "unavailable"])
    result = {
        "stage": stage,
        "metrics": allmetrics,
        "contrasts": allcontrasts,
        "seed_metrics": seedmetrics,
        "failures_and_fallbacks": failures,
        "denominators": audit,
        "identity": {
            "whole_freeze_sha256": digest(
                root / f"{stage}-all-predictions-frozen.json"
            ),
            "release_sha256": digest(root / "release-identity.json"),
        },
        "uncertainty": "conditional complete-event bootstrap; no training refits; secondary scopes cannot rescue primary",
        "prospective_validation": False,
    }
    save_json(out / "results.json", result)
    print(
        "SCORED",
        stage,
        "metrics",
        len(allmetrics),
        "contrasts",
        len(allcontrasts),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--stage", choices=("core", "players", "removals"), required=True)
    a = p.parse_args()
    cuda()
    torch.set_num_threads(4)
    main(a.root, a.stage)
