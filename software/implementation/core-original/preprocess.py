"""Common training-only imputation, missingness, vocabularies and swap orbits."""

from pathlib import Path
import numpy as np, torch
from common import *


def transform(raw, ix, sg):
    device = cuda()
    ix = torch.tensor(ix, device=device)
    sg = torch.tensor(sg, device=device)
    x = raw["train"]
    pool = torch.cat((x, x[:, ix] * sg))
    pool = torch.where(torch.isfinite(pool), pool, torch.nan)
    median = torch.nanmedian(pool, 0).values.nan_to_num(0)
    median = (median + median[ix] * sg) / 2
    expanded = torch.cat(
        (
            torch.where(torch.isfinite(pool), pool, median),
            (~torch.isfinite(pool)).float(),
        ),
        1,
    )
    exix = torch.cat((ix, ix + len(ix)))
    exsg = torch.cat((sg, torch.ones_like(sg)))
    mean = expanded.mean(0)
    mean = (mean + mean[exix] * exsg) / 2
    scale = expanded.std(0, unbiased=False)
    scale = (scale + scale[exix]) / 2
    keep = (scale > 1e-7).nonzero().flatten().tolist()
    remap = {j: i for i, j in enumerate(keep)}
    if any(int(exix[j]) not in remap for j in keep):
        raise RuntimeError("Incomplete transformed swap orbit")
    proc = {
        "median": median.cpu().tolist(),
        "mean": mean.cpu().tolist(),
        "scale": scale.cpu().tolist(),
        "active": keep,
        "swap_indices": [remap[int(exix[j])] for j in keep],
        "swap_signs": [float(exsg[j]) for j in keep],
    }
    out = {}
    for p, x in raw.items():
        ex = torch.cat(
            (torch.where(torch.isfinite(x), x, median), (~torch.isfinite(x)).float()), 1
        )
        out[p] = ((ex - mean) / scale.clamp_min(1e-7))[:, keep].float()
        if not torch.isfinite(out[p]).all():
            raise RuntimeError("Nonfinite processed values")
    return out, proc


def arrays(block, task, rep, drop=None, coverage=False, player_mode=None):
    r = read(block / "inputs/prepared.json")
    m = r["tasks"][task]
    keys = {p: read(block / "inputs" / task / f"{p}-keys.json") for p in PHASES}
    device = cuda()
    cols = [
        j
        for j, (g, f) in enumerate(zip(m["groups"], m["semantic"]))
        if g in rep and (drop is None or f != drop or g == "B")
    ]
    remap = {j: i for i, j in enumerate(cols)}
    ix = [remap[m["swap_numeric_indices"][j]] for j in cols]
    sg = [m["swap_numeric_signs"][j] for j in cols]
    usecomp = "C" in rep
    if usecomp:
        ix += [len(cols) + j for j in m["swap_composition_indices"]]
        sg += m["swap_composition_signs"]
    raw = {}
    for p in PHASES:
        with np.load(block / "inputs" / task / f"{p}.npz", allow_pickle=False) as z:
            x = z["numeric"][:, cols].copy()
            if coverage:
                for j, orig in enumerate(cols):
                    if (
                        m["groups"][orig] != "B"
                        and "_coverage_" not in m["feature_names"][orig]
                    ):
                        x[:, j] = np.where(np.isfinite(x[:, j]), 0, np.nan)
            if usecomp:
                x = np.column_stack((x, z["composition"]))
        raw[p] = torch.tensor(x, device=device)
    # Empty train uses explicit fallback and never enters transform.
    if not len(keys["train"]):
        return {}, keys, {"empty": True}
    transformed, proc = transform(raw, ix, sg)
    proc.update(
        columns=cols,
        representation=rep,
        drop=drop,
        coverage_only=coverage,
        fit_population="mirrored training only",
    )
    maps = (
        sorted({k["map_entity_id"] for k in keys["train"]})
        if task == "map_winner"
        else []
    )
    vocab = {m: i + 1 for i, m in enumerate(maps)}
    if maps:
        n = len(proc["swap_indices"])
        proc["swap_indices"] += list(range(n, n + len(vocab) + 1))
        proc["swap_signs"] += [1.0] * (len(vocab) + 1)
        for p in PHASES:
            cat = torch.tensor(
                [vocab.get(k["map_entity_id"], 0) for k in keys[p]],
                device=device,
                dtype=torch.long,
            )
            transformed[p] = torch.cat(
                (
                    transformed[p],
                    torch.nn.functional.one_hot(cat, len(vocab) + 1).float(),
                ),
                1,
            )
    proc["map_vocabulary"] = vocab
    data = {
        p: {
            "x": transformed[p],
            "weight": torch.tensor(
                weights(keys[p]) * len(keys[p]), device=device, dtype=torch.float32
            ),
        }
        for p in PHASES
    }
    for p in PHASES[:-1]:
        data[p]["y"] = torch.tensor(
            np.load(block / "inputs" / task / f"{p}-labels.npy", allow_pickle=False),
            device=device,
            dtype=torch.float32,
        )
    if player_mode:
        general = list(range(len(m["player_fields"]) * 2)) + [m["player_width"] - 1]
        full = list(range(m["player_width"]))
        pc = full if player_mode == "full" else general
        pr = {}
        for p in PHASES:
            with np.load(
                block / "inputs" / task / f"{p}-players.npz", allow_pickle=False
            ) as z:
                pr[p] = torch.tensor(z["players"][..., pc], device=device)
        pool = pr["train"].reshape(-1, len(pc))
        med = torch.nanmedian(pool, 0).values.nan_to_num(0)
        filled = torch.where(torch.isfinite(pool), pool, med)
        mu = filled.mean(0)
        std = filled.std(0, unbiased=False).clamp_min(1e-7)
        proc["player_processor"] = {
            "columns": pc,
            "median": med.cpu().tolist(),
            "mean": mu.cpu().tolist(),
            "scale": std.cpu().tolist(),
            "agent_vocabulary": m["player_agent_vocabulary"],
            "mode": player_mode,
        }
        for p in PHASES:
            x = pr[p]
            v = torch.cat(
                (
                    (torch.where(torch.isfinite(x), x, med) - mu) / std,
                    (~torch.isfinite(x)).float(),
                ),
                -1,
            )
            if player_mode == "full":
                with np.load(
                    block / "inputs" / task / f"{p}-players.npz", allow_pickle=False
                ) as z:
                    agent = torch.tensor(z["agents"], device=device, dtype=torch.long)
                v = torch.cat(
                    (
                        v,
                        torch.nn.functional.one_hot(
                            agent, len(m["player_agent_vocabulary"]) + 1
                        ).float(),
                    ),
                    -1,
                )
            data[p]["players"] = v
    return data, keys, proc


def swap(x, proc):
    return x[:, proc["swap_indices"]] * torch.tensor(
        proc["swap_signs"], device=x.device
    )
