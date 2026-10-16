"""Saved-model inference only; no new labels, fitting, tuning or scoring."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import xgboost as xgb
from later_reference_state import read, sha, sigmoid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--method", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    assert not args.output.exists(), "Preserve existing prediction freezes"
    assert torch.cuda.is_available() and xgb.build_info().get(
        "USE_CUDA"
    ), "Retained inference requires the CUDA runtime"
    torch.set_num_threads(2)
    method, features, reference = (
        read(args.method),
        read(args.features / "frozen.json"),
        read(args.reference / "frozen.json"),
    )
    assert features["method_sha256"] == reference["method_sha256"] == sha(args.method)
    assert features["reference_sha256"] == sha(args.reference / "frozen.json")
    for root, receipt in ((args.features, features), (args.reference, reference)):
        for name, h in receipt["files"].items():
            assert sha(root / name) == h
    keys = read(args.features / "keys.json")
    assert keys == read(args.reference / "keys.json")
    assert {(p["arm"], p["task"]) for p in method["parent_model_packages"]} == {
        (a, t) for a in ("original_residual", "rich_residual") for t in method["tasks"]
    }
    assert len(method["parent_model_packages"]) == 4
    with np.load(args.features / "features.npz") as z:
        matrices = {
            "original_residual": z["original"].copy(),
            "rich_residual": z["rich"].copy(),
        }
    with np.load(args.reference / "features.npz") as z:
        dynamic, baselines = z["dynamic"].copy(), z["matched_baselines"].copy()
    probabilities = {
        "dynamic_strength": sigmoid(dynamic[:, 0]),
        "online_elo_k20": baselines[:, 0],
        "training_prior": baselines[:, 1],
        "chance": np.full(len(keys), 0.5),
    }
    decisions = {name: (p > 0.5).astype("int8") for name, p in probabilities.items()}
    raw_predictions = {name: np.empty(len(keys), dtype="float32") for name in matrices}
    for name in matrices:
        probabilities[name] = np.empty(len(keys), dtype="float32")
        decisions[name] = np.empty(len(keys), dtype="int8")
    models = []
    for package in method["parent_model_packages"]:
        arm, task = package["arm"], package["task"]
        assert sha(package["recipe_path"]) == package["recipe_sha256"]
        assert sha(package["release_path"]) == package["release_sha256"]
        indices = [i for i, (t, g) in enumerate(keys) if t == task]
        matrix = matrices[arm][indices]
        assert matrix.shape == (len(indices), len(package["feature_names"]))
        seed_predictions = []
        for model in package["models"]:
            assert (
                sha(model["path"]) == model["sha256"]
                and sha(model["receipt_path"]) == model["receipt_sha256"]
            )
            receipt = read(model["receipt_path"])
            assert (
                receipt["model"] == model["sha256"]
                and receipt["best_iteration"] == model["best_iteration"]
            )
            booster = xgb.Booster(model_file=model["path"])
            booster.set_param({"device": "cuda:0", "nthread": 2})
            assert booster.num_features() == len(package["feature_names"])
            if len(indices):
                data = xgb.DMatrix(matrix, base_margin=dynamic[indices, 0])
                seed_predictions.append(
                    booster.predict(
                        data, iteration_range=(0, model["best_iteration"] + 1)
                    )
                )
            models.append(
                {
                    "arm": arm,
                    "task": task,
                    "path": model["path"],
                    "sha256": model["sha256"],
                    "best_iteration": model["best_iteration"],
                }
            )
        assert len(package["models"]) == 3
        if not indices:
            continue
        raw = np.mean(seed_predictions, axis=0)
        z = torch.logit(
            torch.tensor(raw, device="cuda", dtype=torch.float64).clamp(1e-7, 1 - 1e-7)
        )
        p = (z * package["calibration_logit_scale"]).sigmoid().float().cpu().numpy()
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        raw_predictions[arm][indices] = raw
        probabilities[arm][indices] = p
        decisions[arm][indices] = (raw > package["decision_threshold"]).astype("int8")
    assert len(models) == 12
    assert set(probabilities) == set(method["arms"])
    args.output.mkdir(parents=True)
    np.savez_compressed(
        args.output / "predictions.npz",
        **{name + "_probability": p for name, p in probabilities.items()},
        **{name + "_pick": p for name, p in decisions.items()},
        **{name + "_raw": p for name, p in raw_predictions.items()}
    )
    (args.output / "keys.json").write_text(json.dumps(keys, indent=2) + "\n")
    complete = features.get("complete_cohort", False) and reference.get(
        "complete_cohort", False
    )
    receipt = {
        "status": (
            "predictions_frozen_before_new_targets"
            if complete
            else "partial_predictions_frozen_before_new_targets"
        ),
        "rows": len(keys),
        "method_sha256": sha(args.method),
        "features_sha256": sha(args.features / "frozen.json"),
        "reference_sha256": sha(args.reference / "frozen.json"),
        "code_sha256": sha(__file__),
        "new_targets_read": False,
        "new_feedback": False,
        "fits_tuning_calibration_or_scores": False,
        "complete_cohort": complete,
        "models": models,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "xgboost": xgb.__version__,
            "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "files": {p.name: sha(p) for p in args.output.iterdir()},
    }
    (args.output / "frozen.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "rows": len(keys),
                "models_loaded": len(models),
                "status": receipt["status"],
                "scores": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
