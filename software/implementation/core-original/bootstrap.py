"""Freeze approved sources, implementation, environment and owner authorization."""

import argparse, importlib.metadata, os, shutil, subprocess, sys
from pathlib import Path
from common import *


def main(
    root,
    repo,
    reuse_root=None,
    drop_ft=False,
    restore_ft=False,
    ft_reuse_root=None,
    defer_ft=False,
    empty_player_repair=False,
):
    root = Path(root)
    repo = Path(repo)
    parent = (
        Path(os.environ["VALORANT_ARCHIVE_ROOT"]) / "advanced-predictors-20261008-v3"
    )
    base = read(parent / "config.json")
    base.update(
        version="strong-predictors/1",
        repository=str(repo),
        owner_authorization="2026-10-09: approved dual-horizon fitting campaign",
        parent_root=str(parent),
        tasks=list(TASKS),
        representations=list(REPS),
        families=list(FAMILIES),
        seeds=[20261009, 20261010, 20261011],
        maximum_epochs=128,
        patience=15,
        format_policy="omitted pending pre-play availability contract",
        series_linked_policy="prior most-observed agent across historical maps",
        bootstrap_draws=131072,
    )
    if reuse_root:
        base.update(
            reuse_root=str(reuse_root),
            runtime_repair="v2: add CUDA13 CuPy14.2.0/fastrlock0.8.3; preserve v1 failures and reusable completed numerical units",
        )
    if drop_ft:
        if not reuse_root:
            raise RuntimeError("Scope amendment requires preserved prior release")
        base = read(Path(reuse_root) / "config.json")
        base.update(
            reuse_root=str(reuse_root),
            families=list(FAMILIES),
            scope_change="owner-20261009-drop-ft-transformer",
            owner_authorization="2026-10-09: FT-Transformer excluded from active candidate set",
            runtime_repair="v3 scope amendment; numerical adapter and dependencies unchanged",
        )
    if restore_ft:
        if not reuse_root or not ft_reuse_root:
            raise RuntimeError("Restore requires both preserved releases")
        base = read(Path(reuse_root) / "config.json")
        base.update(
            reuse_root=str(reuse_root),
            ft_reuse_root=str(ft_reuse_root),
            families=list(FAMILIES),
            scope_change="owner-20261009-restore-ft-transformer",
            owner_authorization="2026-10-09: FT-Transformer restored to candidate scope",
            runtime_repair="v4 six-family scope restored; numerical adapter and dependencies unchanged",
        )
    if defer_ft:
        if not reuse_root:
            raise RuntimeError("Deferral requires preserved prior release")
        base = read(Path(reuse_root) / "config.json")
        base.pop("ft_reuse_root", None)
        base.update(
            reuse_root=str(reuse_root),
            families=list(FAMILIES),
            scope_change="owner-20261010-defer-ft-transformer",
            owner_authorization="2026-10-10: FT-Transformer deferred from active scoring",
            runtime_repair="v5 FT-Transformer deferral; numerical adapter and dependencies unchanged",
        )
    if empty_player_repair:
        if not reuse_root:
            raise RuntimeError("Repair requires preserved prior release")
        base = read(Path(reuse_root) / "config.json")
        base.update(
            reuse_root=str(reuse_root),
            empty_player_repair=True,
            runtime_repair="v6 empty player partition shape repair; nonempty numerical method unchanged",
        )
    source = repo / "ops/strong-predictors"
    code = root / "code"
    if code.exists():
        raise RuntimeError("Existing frozen root; version repair rather than overwrite")
    shutil.copytree(source, code, ignore=shutil.ignore_patterns("__pycache__"))
    save_json(root / "config.json", base)
    save_json(
        repo
        / (
            "configs/strong-predictors/20261010-v6.json"
            if empty_player_repair
            else (
                "configs/strong-predictors/20261010-v5.json"
                if defer_ft
                else (
                    "configs/strong-predictors/20261009-v4.json"
                    if restore_ft
                    else (
                        "configs/strong-predictors/20261009-v3.json"
                        if drop_ft
                        else (
                            "configs/strong-predictors/20261009-v2.json"
                            if reuse_root
                            else "configs/strong-predictors/20261009-v1.json"
                        )
                    )
                )
            )
        ),
        base,
    )
    shutil.copyfile(
        repo / "docs/research-design/strong-predictor-study-plan-2026-10-09.md",
        root / "method.md",
    )
    if drop_ft:
        (root / "method.md").write_bytes(
            (Path(reuse_root) / "method.md").read_bytes()
            + b"\n\nOwner scope amendment 2026-10-09: drop FT-Transformer from the active core comparison and selection. Preserve all prior fits/checkpoints; retain five core families, both tasks, all predictors, player-set models and removals. No evaluation outcomes inspected for this decision.\n"
        )
    if restore_ft:
        (root / "method.md").write_bytes(
            (Path(reuse_root) / "method.md").read_bytes()
            + b"\n\nOwner scope amendment 2026-10-09: restore FT-Transformer to the full six-family core comparison and selection after inspecting its preserved selection scores only. Reuse compatible completed units from v3 and FT units/checkpoints from v2. Preserve all prior releases. Both tasks, all predictor sets, player-set comparisons and removals remain in scope. No held-forward evaluation outcomes inspected for this decision. This adaptive restoration is exploratory.\n"
        )
    if defer_ft:
        (root / "method.md").write_bytes(
            (Path(reuse_root) / "method.md").read_bytes()
            + b"\n\nOwner scope amendment 2026-10-10: defer FT-Transformer to a preserved follow-up; continue five core families across all blocks and exclude FT from active selection, scoring and removal-parent eligibility. Preserve all FT models, forecasts, checkpoints and prior releases. Both tasks, all predictor sets, player comparisons and removals remain unchanged. The decision follows selection diagnostics and observed compute cost only; no held-forward evaluation outcomes inspected. Revisit FT only if additional study evidence is needed, with a new recorded authorization and frozen settings. This adaptive deferral is exploratory.\n"
        )
    if empty_player_repair:
        (root / "method.md").write_bytes(
            (Path(reuse_root) / "method.md").read_bytes()
            + b"\n\nRuntime repair 2026-10-10: handle empty player partitions with an explicit zero-row summary width. Nonempty mean/std/max/top-two summaries and all core fitting, prediction, calibration, selection and scoring methods remain unchanged. Preserve v5 core results and failed player attempt. No outcome-driven method changes.\n"
        )
    env = {
        "python": sys.version,
        "executable": sys.executable,
        "pythonpath": os.environ.get("PYTHONPATH"),
        "distributions": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
        "set_transformer_commit": "73432c640ac78140496d6738416c54d32c686d65",
    }
    import torch, xgboost, catboost

    env["pip_freeze"] = (
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        if importlib.util.find_spec("pip")
        else "pip unavailable; distributions recorded"
    )
    env.update(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        xgboost_build=xgboost.build_info(),
        catboost_gpu_policy="task_type GPU mandatory; native host inference disclosed",
    )
    if not torch.cuda.is_available() or not xgboost.build_info()["USE_CUDA"]:
        raise RuntimeError("CUDA environment unavailable")
    save_json(root / "environment.lock", env)
    import legacy_prepare as legacy

    legacy.verify_registry(root)
    identity = {
        "code": {p.name: digest(p) for p in sorted(code.iterdir()) if p.is_file()},
        "config_sha256": digest(root / "config.json"),
        "method_sha256": digest(root / "method.md"),
        "environment_sha256": digest(root / "environment.lock"),
        "registry_sha256": digest(Path(base["registry_root"]) / "release.json"),
        "git_base": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "owner_authorization": base["owner_authorization"],
    }
    save_json(root / "release-identity.json", identity)
    if reuse_root:
        from recovery import admissible

        proof = admissible(root, Path(reuse_root))
        if restore_ft:
            save_json(
                root / "ft-recovery-compatibility.json",
                admissible(root, Path(ft_reuse_root)),
            )
        save_json(root / "recovery-compatibility.json", proof)
        for fold in base["folds"]:
            old = Path(reuse_root) / "blocks" / fold["name"]
            new = root / "blocks" / fold["name"]
            if (old / "inputs/prepared.json").exists():
                verify_files(
                    old / "inputs", read(old / "inputs/prepared.json")["files"]
                )
                new.mkdir(parents=True, exist_ok=True)
                shutil.copytree(old / "inputs", new / "inputs")
                shutil.copyfile(old / "partition.json", new / "partition.json")
                save_json(
                    new / "reused-inputs.json",
                    {
                        "directory": str(old / "inputs"),
                        "prepared_sha256": digest(old / "inputs/prepared.json"),
                        "compatibility": proof,
                    },
                )
    print("FROZEN", root, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repository", type=Path, required=True)
    p.add_argument("--reuse-root", type=Path)
    p.add_argument("--drop-ft", action="store_true")
    p.add_argument("--restore-ft", action="store_true")
    p.add_argument("--ft-reuse-root", type=Path)
    p.add_argument("--defer-ft", action="store_true")
    p.add_argument("--empty-player-repair", action="store_true")
    a = p.parse_args()
    main(
        a.root,
        a.repository,
        a.reuse_root,
        a.drop_ft,
        a.restore_ft,
        a.ft_reuse_root,
        a.defer_ft,
        a.empty_player_repair,
    )
