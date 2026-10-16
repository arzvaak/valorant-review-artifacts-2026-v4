"""Explicit v2 dependency repair: retain original model/code/seed identities."""

import ast, shutil
from pathlib import Path
from common import *


def admissible(root, old):
    new = read(root / "release-identity.json")
    prior = read(old / "release-identity.json")
    verify_files(old / "code", prior["code"])
    check(old / "config.json", prior["config_sha256"])
    check(old / "environment.lock", prior["environment_sha256"])
    check(old / "method.md", prior["method_sha256"])
    check(root / "config.json", new["config_sha256"])
    check(root / "environment.lock", new["environment_sha256"])
    check(root / "method.md", new["method_sha256"])
    repair = read(root / "config.json").get("empty_player_repair", False)
    restore = (
        read(root / "config.json").get("scope_change")
        == "owner-20261009-restore-ft-transformer"
    )
    defer = (
        not repair
        and read(root / "config.json").get("scope_change")
        == "owner-20261010-defer-ft-transformer"
    )
    scope = (
        defer
        or restore
        or read(root / "config.json").get("scope_change")
        == "owner-20261009-drop-ft-transformer"
    )
    if scope:
        expected = (
            old / "method.md"
        ).read_bytes() + b"\n\nOwner scope amendment 2026-10-09: drop FT-Transformer from the active core comparison and selection. Preserve all prior fits/checkpoints; retain five core families, both tasks, all predictors, player-set models and removals. No evaluation outcomes inspected for this decision.\n"
        if restore:
            appendix = b"\n\nOwner scope amendment 2026-10-09: restore FT-Transformer to the full six-family core comparison and selection after inspecting its preserved selection scores only. Reuse compatible completed units from v3 and FT units/checkpoints from v2. Preserve all prior releases. Both tasks, all predictor sets, player-set comparisons and removals remain in scope. No held-forward evaluation outcomes inspected for this decision. This adaptive restoration is exploratory.\n"
            expected = (
                (old / "method.md").read_bytes()
                if read(old / "config.json").get("scope_change")
                == "owner-20261009-drop-ft-transformer"
                else expected
            ) + appendix
        if defer:
            expected = (
                old / "method.md"
            ).read_bytes() + b"\n\nOwner scope amendment 2026-10-10: defer FT-Transformer to a preserved follow-up; continue five core families across all blocks and exclude FT from active selection, scoring and removal-parent eligibility. Preserve all FT models, forecasts, checkpoints and prior releases. Both tasks, all predictor sets, player comparisons and removals remain unchanged. The decision follows selection diagnostics and observed compute cost only; no held-forward evaluation outcomes inspected. Revisit FT only if additional study evidence is needed, with a new recorded authorization and frozen settings. This adaptive deferral is exploratory.\n"
        if (root / "method.md").read_bytes() != expected:
            raise RuntimeError("Unreviewed scope method change")
    elif repair:
        if (root / "method.md").read_bytes() != (
            old / "method.md"
        ).read_bytes() + b"\n\nRuntime repair 2026-10-10: handle empty player partitions with an explicit zero-row summary width. Nonempty mean/std/max/top-two summaries and all core fitting, prediction, calibration, selection and scoring methods remain unchanged. Preserve v5 core results and failed player attempt. No outcome-driven method changes.\n":
            raise RuntimeError("Unreviewed empty partition repair")
    elif new["method_sha256"] != prior["method_sha256"]:
        raise RuntimeError("Recovery method changed")
    a = read(root / "config.json")
    b = read(old / "config.json")
    for c in (a, b):
        c.pop("reuse_root", None)
        c.pop("runtime_repair", None)
        c.pop("ft_reuse_root", None)
    if scope:
        if restore:
            if a["families"] != [
                "linear",
                "catboost",
                "xgboost",
                "mlp",
                "tabm",
                "ft_transformer",
            ] or b["families"] not in (a["families"], a["families"][:-1]):
                raise RuntimeError("Unexpected restoration family scope")
            if b.get("scope_change") not in (
                None,
                "owner-20261009-drop-ft-transformer",
            ):
                raise RuntimeError("Unexpected restoration source scope")
            b.pop("scope_change", None)
        elif (
            b["families"]
            != ["linear", "catboost", "xgboost", "mlp", "tabm", "ft_transformer"]
            or a["families"] != b["families"][:-1]
        ):
            raise RuntimeError("Unexpected family scope change")
        if defer:
            if b.get("scope_change") != "owner-20261009-restore-ft-transformer":
                raise RuntimeError("Unexpected deferral source scope")
            b.pop("scope_change")
        a["families"] = b["families"]
        a.pop("scope_change", None)
        a["owner_authorization"] = b["owner_authorization"]
    if repair:
        if not a.pop("empty_player_repair", False) or b.get("empty_player_repair"):
            raise RuntimeError("Unexpected repair source")
    if a != b:
        raise RuntimeError("Recovery numerical/source config changed")
    a = read(root / "environment.lock")
    b = read(old / "environment.lock")
    for key in ("torch", "cuda", "gpu", "xgboost_build", "catboost_gpu_policy"):
        if a[key] != b[key]:
            raise RuntimeError("Recovery numerical runtime changed: " + key)
    for key, value in b["distributions"].items():
        if a["distributions"].get(key) != value:
            raise RuntimeError("Recovery installed distribution changed: " + key)
    for name, version in {"cupy-cuda13x": "14.2.0", "fastrlock": "0.8.3"}.items():
        if a["distributions"].get(name) != version:
            raise RuntimeError("Repair dependency version changed: " + name)
    if (
        set(a["distributions"])
        - set(b["distributions"])
        - {"cupy-cuda13x", "fastrlock"}
    ):
        raise RuntimeError("Unreviewed dependency additions")
    if scope:

        def common_body(path):
            tree = ast.parse(path.read_text())
            tree.body = [
                n
                for n in tree.body
                if not (
                    isinstance(n, ast.Assign)
                    and any(
                        isinstance(t, ast.Name) and t.id == "FAMILIES"
                        for t in n.targets
                    )
                )
            ]
            return ast.dump(tree, include_attributes=False)

        if common_body(root / "code/common.py") != common_body(old / "code/common.py"):
            raise RuntimeError("Non-scope common code changed")
    for name in (
        (
            "preprocess.py",
            "snapshot.py",
            "player_snapshot.py",
            "legacy_prepare.py",
            "gpu_runtime.py",
            "set_modules.py",
            "targets.py",
            "selector.py",
        )
        if scope
        else (
            "common.py",
            "preprocess.py",
            "snapshot.py",
            "player_snapshot.py",
            "legacy_prepare.py",
            "gpu_runtime.py",
            "set_modules.py",
            "targets.py",
            "selector.py",
        )
    ):
        if digest(root / "code" / name) != digest(old / "code" / name):
            raise RuntimeError("Recovery source/processor code changed: " + name)
    for name, nodes in {
        "models.py": (
            "Net",
            "Epoch",
            "loader",
            "neural",
            "linear",
            "tree",
            "MemoryMonitor",
        ),
        "fit.py": (
            "presets",
            "calibrated",
            "scale_probs",
            "baselines",
            "map_baseline",
            "eligible",
            "stochastic_eligible",
            "aggregate_players",
        ),
    }.items():

        def bodies(path):
            return {
                n.name: ast.dump(n, include_attributes=False)
                for n in ast.parse(path.read_text()).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            }

        aa = bodies(root / "code" / name)
        bb = bodies(old / "code" / name)
        if repair and name == "fit.py":
            expected = (
                (old / "code/fit.py")
                .read_text()
                .replace(
                    "  p=v.pop('players');summ=torch.cat((p.mean(2),p.std(2,unbiased=False),p.max(2).values,p.topk(2,dim=2).values.mean(2)),2).reshape(len(p),-1);v['x']=torch.cat((v['x'],summ),1)",
                    "  p=v.pop('players')\n  summ=torch.cat((p.mean(2),p.std(2,unbiased=False),p.max(2).values,p.topk(2,dim=2).values.mean(2)),2).reshape(len(p),2*extra) if len(p) else p.new_empty((0,2*extra))\n  v['x']=torch.cat((v['x'],summ),1)",
                )
            )
            if (
                expected == (old / "code/fit.py").read_text()
                or bodies(root / "code/fit.py")["aggregate_players"]
                != {
                    n.name: ast.dump(n, include_attributes=False)
                    for n in ast.parse(expected).body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                }["aggregate_players"]
            ):
                raise RuntimeError("Unexpected player aggregation repair")
            aa["aggregate_players"] = bb["aggregate_players"]
        if any(aa[n] != bb[n] for n in nodes):
            raise RuntimeError("Recovery numerical adapter changed: " + name)
    return {
        "prior_release_sha256": digest(old / "release-identity.json"),
        "new_release_sha256": digest(root / "release-identity.json"),
        "scope_change": scope,
        "dependency_additions": (
            {} if scope or repair else {"cupy-cuda13x": "14.2.0", "fastrlock": "0.8.3"}
        ),
        "empty_player_repair": repair,
        "policy": (
            "explicit empty player partition width; nonempty numerical method and core unchanged"
            if repair
            else (
                (
                    "owner restores FT after selection diagnostics only; original numerical adapters and execution identities retained"
                    if restore
                    else "owner removes FT from family eligibility only; unchanged retained numerical adapters, inputs, parameters and seeds; original execution identities retained"
                )
                if scope
                else "only CuPy/fastrlock addition; same numerical adapter AST, method, inputs, parameters, seed; original execution identity retained"
            )
        ),
    }


def reuse(path, identity, cfg, kind):
    if not cfg.get("reuse_root"):
        return None
    root = (
        path.parents[len(path.relative_to(Path(cfg["run_root"])).parts) - 1]
        if cfg.get("run_root")
        else next(p for p in path.parents if (p / "release-identity.json").exists())
    )
    old = Path(
        cfg["ft_reuse_root"]
        if cfg.get("ft_reuse_root") and identity.get("family") == "ft_transformer"
        else cfg["reuse_root"]
    )
    source = old / path.relative_to(root)
    marker = source / "frozen.json"
    if not marker.exists():
        attempts = sorted(source.glob("attempt-*.json")) if kind == "candidate" else []
        if attempts and (source / "processor.json").exists():
            original = read(attempts[-1])["identity"]
            aa = {k: v for k, v in identity.items() if k not in ("release", "code")}
            bb = {k: v for k, v in original.items() if k not in ("release", "code")}
            if aa == bb and digest(source / "processor.json") == digest(
                path / "processor.json"
            ):
                proof = admissible(root, old)
                copied = []
                for directory in source.glob("neural-attempt-*"):
                    dest = path / directory.name
                    if not dest.exists():
                        copy_checkpoint(directory, dest)
                        copied.append(directory.name)
                    elif (dest / "copy-origin.json").exists() and not list(
                        path.glob("attempt-*.json")
                    ):
                        verify_files(dest, read(dest / "copy-origin.json")["files"])
                if copied:
                    save_json(
                        path / "checkpoint-recovery.json",
                        {
                            "original_execution_identity": original,
                            "source": str(source),
                            "copied": copied,
                            "compatibility": proof,
                        },
                    )
        return None
    r = read(marker)
    original = r["identity"]
    aa = {k: v for k, v in identity.items() if k not in ("release", "code")}
    bb = {k: v for k, v in original.items() if k not in ("release", "code")}
    if aa != bb:
        return None
    proof = admissible(root, old)
    if kind == "package":
        if any(c["record"]["status"] == "failed" for c in r["model_records"]):
            return None
        verify_files(source, r["files"])
        for name in r["files"]:
            shutil.copyfile(source / name, path / name)
        for item in r["model_records"]:
            c = item["record"]
            if c["status"] == "fitted":
                check(c["model"], c["model_sha256"])
                check(Path(item["path"]) / "predictions.npz", c["prediction_sha256"])
                check(Path(item["path"]) / "processor.json", c["processor_sha256"])
        r["directory"] = str(path)
    else:
        check(r["model"], r["model_sha256"])
        check(source / "predictions.npz", r["prediction_sha256"])
        check(source / "processor.json", r["processor_sha256"])
        check(path / "processor.json", r["processor_sha256"])
        shutil.copyfile(source / "predictions.npz", path / "predictions.npz")
    r["reused_from"] = {
        "path": str(source),
        "freeze_sha256": digest(marker),
        "original_execution_identity": original,
        "compatibility": proof,
    }
    r["identity"] = identity
    save_json(path / "frozen.json", r)
    return r


def copy_checkpoint(source, destination):
    # Failed copy attempts remain separately named; only verified complete copies
    # become the final checkpoint directory. Source service has already stopped.
    i = 1
    while destination.with_name(destination.name + f".copy-attempt-{i}").exists():
        i += 1
    pending = destination.with_name(destination.name + f".copy-attempt-{i}")
    manifest = files(source)
    shutil.copytree(source, pending)
    verify_files(pending, manifest)
    last = pending / "checkpoints/last.ckpt"
    changes = {}
    if last.exists():
        import torch
        import gpu_runtime as rt

        checkpoint = torch.load(last, map_location="cpu", weights_only=False)

        def relink(value):
            if isinstance(value, str) and value.startswith(str(source)):
                new = str(destination) + value[len(str(source)) :]
                changes[value] = new
                return new
            if isinstance(value, dict):
                return {relink(k): relink(v) for k, v in value.items()}
            if isinstance(value, list):
                return [relink(v) for v in value]
            if isinstance(value, tuple):
                return tuple(relink(v) for v in value)
            return value

        checkpoint["callbacks"] = relink(checkpoint.get("callbacks", {}))
        rt.save_torch(last, checkpoint)
    save_json(
        pending / "copy-origin.json",
        {
            "source": str(source),
            "source_files": manifest,
            "files": files(pending),
            "checkpoint_callback_path_relocation": changes,
            "numerical_weights_optimizer_rng_unchanged": True,
        },
    )
    verify_files(pending, read(pending / "copy-origin.json")["files"])
    pending.rename(destination)
    fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(fd)
    os.close(fd)
