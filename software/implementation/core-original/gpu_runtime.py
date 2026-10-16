"""Published architectures + Lightning epochs/checkpoints; CUDA-only real fits."""

import argparse
import ast
import gc
import json
import math
import os
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric.plugins.io import TorchCheckpointIO
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from rtdl_num_embeddings import LinearReLUEmbeddings
from rtdl_revisiting_models import FTTransformer
from tabm import TabM
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
)

from common import code_identity, cuda, digest, save_json


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: torch.as_tensor(data[k].copy(), device=cuda()) for k in data.files}


def save_torch(path, value):
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(fd)
    os.close(fd)


def event_weights(keys):
    events = [r["event_id"] for r in keys]
    counts = Counter(events)
    return torch.tensor(
        [len(events) / len(counts) / counts[e] for e in events], device=cuda()
    )


def swap_spec(task, representation, metadata, width, cat_width):
    numeric = list(range(width))
    signs = [1.0] * width
    categorical = list(range(cat_width))
    c = metadata["dimensions"]["context_width"]
    m, a = metadata["maps"], metadata["agents"]

    def exchange(indices, first, second, length):
        indices[first : first + length], indices[second : second + length] = (
            list(range(second, second + length)),
            list(range(first, first + length)),
        )

    if task == "map":
        if representation == "aggregate":
            signs[:18] = [-1.0] * 18
            exchange(numeric, 18, 18 + a, a)
        else:
            signs[:2] = [-1.0, -1.0]
            p = metadata["dimensions"]["profile_width"] * 5
            exchange(numeric, c, c + p, p)
            exchange(categorical, 0, 5, 5)
    else:
        signs[: 1 + m] = [-1.0] * (1 + m)
        exchange(numeric, 1 + m, 1 + 2 * m, m)
        exchange(numeric, 1 + 3 * m, 1 + 3 * m + a, a)
        exchange(numeric, c - 2, c - 1, 1)
        if representation == "aggregate":
            signs[c:] = [-1.0] * 16
        else:
            p = metadata["dimensions"]["profile_width"] * 5
            exchange(numeric, c, c + p, p)
            exchange(categorical, 0, 15, 15)
    return {
        "numeric_indices": numeric,
        "numeric_signs": signs,
        "categorical_indices": categorical,
    }


def swapped(x, cat, spec):
    return (
        x[:, spec["numeric_indices"]]
        * torch.tensor(spec["numeric_signs"], device=x.device),
        cat[:, spec["categorical_indices"]],
    )


def representation_arrays(raw, task, representation, metadata, impute=None):
    if representation == "individual":
        return raw["numeric"].float(), raw["categorical"].long(), None
    c = metadata["dimensions"]["context_width"]
    features = raw["aggregate"][:, c:].reshape(-1, 10, 16)
    observed = features[:, :, 8:] > 0
    means = features[:, :, :8]
    if impute is None:
        impute = torch.nanmedian(means.flatten(0, 1), dim=0).values.nan_to_num(0)
    filled = torch.where(observed, means, impute)
    difference = filled[:, :5].mean(1) - filled[:, 5:].mean(1)
    coverage = observed[:, :5].float().mean(1) - observed[:, 5:].float().mean(1)
    result = torch.cat((raw["aggregate"][:, :c], difference, coverage), dim=1)
    if task == "map":
        result = torch.cat((result, raw["presence"]), dim=1)
    return (
        result,
        torch.empty((len(result), 0), dtype=torch.long, device=cuda()),
        impute,
    )


def prepare_data(directory, task, representation):
    metadata = json.loads((directory / "prepared.json").read_text())
    # Read the target-free inference file only; evaluation-targets is not opened here.
    raw = {
        role: load(directory / f"{role}.npz")
        for role in ("train", "validation", "evaluation")
    }
    for role, values in raw.items():
        if role == "evaluation" and "y" in values:
            raise ValueError("evaluation target present in inference input")
    tx, tc, impute = representation_arrays(raw["train"], task, representation, metadata)
    spec = swap_spec(task, representation, metadata, tx.shape[1], tc.shape[1])
    sx, sc = swapped(tx, tc, spec)
    training_numeric = torch.cat((tx, sx))
    median = torch.nanmedian(training_numeric, dim=0).values.nan_to_num(0)
    filled = torch.where(torch.isnan(training_numeric), median, training_numeric)
    scale = filled.square().mean(0).sqrt().clamp_min(1e-6)
    training_cat = torch.cat((tc, sc))
    n_agent_columns = tc.shape[1] - (1 if task == "map" and tc.shape[1] else 0)
    agent_vocab = (
        training_cat[:, :n_agent_columns].unique(sorted=True)
        if n_agent_columns
        else torch.tensor([0], device=cuda())
    )
    map_vocab = (
        training_cat[:, -1].unique(sorted=True)
        if tc.shape[1] > n_agent_columns
        else None
    )

    def encode(cat):
        if not cat.shape[1]:
            return cat
        outputs = []
        for j in range(cat.shape[1]):
            vocab = agent_vocab if j < n_agent_columns else map_vocab
            positions = torch.searchsorted(vocab, cat[:, j].contiguous())
            valid = (positions < len(vocab)) & (
                vocab[positions.clamp_max(len(vocab) - 1)] == cat[:, j]
            )
            outputs.append(torch.where(valid & (cat[:, j] != 0), positions + 1, 0))
        return torch.stack(outputs, dim=1)

    cardinalities = [len(agent_vocab) + 1] * n_agent_columns
    if map_vocab is not None:
        cardinalities.append(len(map_vocab) + 1)
    data, keys = {}, {}
    for role, values in raw.items():
        x, cat, _ = representation_arrays(
            values, task, representation, metadata, impute
        )
        x = (torch.where(torch.isnan(x), median, x) / scale).float()
        cat = encode(cat)
        if not torch.isfinite(x).all():
            raise ValueError("nonfinite transformed inputs")
        keys[role] = json.loads((directory / f"{role}-keys.json").read_text())
        if len(keys[role]) != len(x):
            raise ValueError("matrix/key membership mismatch")
        data[role] = {
            "numeric": x,
            "categorical": cat,
            "weight": event_weights(keys[role]),
        }
        if role != "evaluation":
            data[role]["y"] = values["y"].float()
    x, cat = data["train"]["numeric"], data["train"]["categorical"]
    sx, sc = swapped(x, cat, spec)
    # Imputation/scaling was fitted on mirrored training only, so swap commutes.
    data["train"] = {
        "numeric": torch.cat((x, sx)),
        "categorical": torch.cat((cat, sc)),
        "weight": data["train"]["weight"].repeat(2),
        "y": torch.cat((data["train"]["y"], 1 - data["train"]["y"])),
    }
    processor = {
        "median": median.cpu(),
        "scale": scale.cpu(),
        "aggregate_impute": None if impute is None else impute.cpu(),
        "agent_vocabulary": agent_vocab.cpu(),
        "map_vocabulary": None if map_vocab is None else map_vocab.cpu(),
        "cardinalities": cardinalities,
        "swap": spec,
        "metadata": metadata,
        "task": task,
        "representation": representation,
    }
    del raw
    return data, keys, processor


class ResidentDataset(Dataset):
    """Batch index adapter for library DataLoader; all numerical tensors stay on CUDA."""

    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values["numeric"])

    def __getitem__(self, index):
        indices = torch.as_tensor(index, device=cuda())
        return {key: value[indices] for key, value in self.values.items()}


def loader(values, batch_size, training, seed):
    dataset = ResidentDataset(values)
    generator = torch.Generator().manual_seed(seed)
    sampler = (
        RandomSampler(dataset, generator=generator)
        if training
        else SequentialSampler(dataset)
    )
    return (
        DataLoader(
            dataset,
            batch_size=None,
            sampler=BatchSampler(sampler, batch_size, drop_last=False),
        ),
        generator,
    )


class DurableCheckpointIO(TorchCheckpointIO):
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        super().save_checkpoint(checkpoint, path, storage_options)
        with open(path, "rb") as stream:
            os.fsync(stream.fileno())
        fd = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(fd)
        os.close(fd)


class EpochRecord(Callback):
    def __init__(self, output, seed, generator, identity):
        self.output, self.seed, self.generator, self.identity = (
            output,
            seed,
            generator,
            identity,
        )
        self.start = None

    def on_train_epoch_start(self, trainer, module):
        self.start = time.monotonic()
        self.generator.manual_seed(self.seed + trainer.current_epoch)
        if not all(p.is_cuda for p in module.parameters()):
            raise RuntimeError("model unexpectedly on CPU")

    def on_train_epoch_end(self, trainer, module):
        metrics = {
            key: float(value.detach().cpu())
            for key, value in trainer.callback_metrics.items()
        }
        free, total = torch.cuda.mem_get_info()
        row = {
            "epoch": trainer.current_epoch + 1,
            "seconds": time.monotonic() - self.start,
            "metrics": metrics,
            "device": str(module.device),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "reserved_gib": torch.cuda.memory_reserved() / 2**30,
            "free_gib": free / 2**30,
            "total_gib": total / 2**30,
            "identity": self.identity,
        }
        save_json(self.output / "progress.json", row)
        print(
            f"{self.output.parent.name}/{self.output.name}: epoch {row['epoch']}; "
            f"validation Brier {metrics.get('val_brier', float('nan')):.6f}; "
            f"epoch {row['seconds']:.1f}s; peak CUDA {row['peak_allocated_gib']:.2f} GiB",
            flush=True,
        )


class Predictor(pl.LightningModule):
    def __init__(self, family, n_numeric, cardinalities, swap, config):
        super().__init__()
        self.save_hyperparameters()
        self.family, self.swap, self.config = family, swap, config
        if family == "tabm":
            options = config["tabm"]
            self.model = TabM.make(
                n_num_features=n_numeric,
                cat_cardinalities=cardinalities or None,
                d_out=1,
                n_blocks=options["n_blocks"],
                d_block=options["d_block"],
                k=options["k"],
                dropout=0.1,
                num_embeddings=LinearReLUEmbeddings(n_numeric, 16),
            )
        else:
            options = FTTransformer.get_default_kwargs()
            options.update(
                {k: v for k, v in config["ft_transformer"].items() if k != "batch_size"}
            )
            self.model = FTTransformer(
                n_cont_features=n_numeric,
                cat_cardinalities=cardinalities,
                **options,
                d_out=1,
            )

    def logits(self, x, cat):
        return self.model(x, cat if cat.shape[1] else None)

    def probability(self, x, cat):
        raw = self.logits(x, cat).float().sigmoid().reshape(len(x), -1).mean(1)
        sx, sc = swapped(x, cat, self.swap)
        other = self.logits(sx, sc).float().sigmoid().reshape(len(x), -1).mean(1)
        return (raw + 1 - other) / 2

    def training_step(self, batch, batch_idx):
        logits = (
            self.logits(batch["numeric"], batch["categorical"])
            .float()
            .reshape(len(batch["y"]), -1)
        )
        y = batch["y"][:, None].expand_as(logits)
        loss = (
            F.binary_cross_entropy_with_logits(logits, y, reduction="none").mean(1)
            * batch["weight"]
        ).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite real training loss")
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=len(y))
        return loss

    def validation_step(self, batch, batch_idx):
        p = self.probability(batch["numeric"], batch["categorical"])
        brier = ((p - batch["y"]).square() * batch["weight"]).mean()
        # The symmetry-averaged probability is already FP32. Native probability
        # BCE must run outside autocast; training uses autocast-safe logits BCE.
        with torch.autocast("cuda", enabled=False):
            loss = (
                F.binary_cross_entropy(
                    p.float().clamp(1e-7, 1 - 1e-7),
                    batch["y"].float(),
                    reduction="none",
                )
                * batch["weight"]
            ).mean()
        self.log("val_brier", brier, on_step=False, on_epoch=True, batch_size=len(p))
        self.log("val_log_loss", loss, on_step=False, on_epoch=True, batch_size=len(p))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config["maximum_epochs"]
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_save_checkpoint(self, checkpoint):
        checkpoint["cuda_rng_states"] = torch.cuda.get_rng_state_all()
        checkpoint["cpu_rng_state"] = torch.get_rng_state()

    def on_load_checkpoint(self, checkpoint):
        if "cuda_rng_states" in checkpoint:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_states"]]
            )
            torch.set_rng_state(checkpoint["cpu_rng_state"].cpu())


def linear_design(x, cat, processor):
    def expand(numeric, categorical):
        return torch.cat(
            [numeric]
            + [
                F.one_hot(categorical[:, j], classes).float()
                for j, classes in enumerate(processor["cardinalities"])
            ],
            dim=1,
        )

    sx, sc = swapped(x, cat, processor["swap"])
    return (expand(x, cat) - expand(sx, sc)) / 2


def fit_linear(data, processor, config, output):
    train = data["train"]
    x = linear_design(train["numeric"], train["categorical"], processor)
    xv = linear_design(
        data["validation"]["numeric"], data["validation"]["categorical"], processor
    )
    best = None
    for penalty in config["linear_penalties"]:
        path = output / f"penalty-{penalty}.pt"
        model = torch.nn.Linear(x.shape[1], 1, bias=False, device=cuda())
        torch.nn.init.zeros_(model.weight)
        marker = output / f"penalty-{penalty}.json"
        if marker.exists():
            if (
                not path.exists()
                or digest(path) != json.loads(marker.read_text())["model_sha256"]
            ):
                raise RuntimeError("completed linear penalty checkpoint changed")
            model.load_state_dict(
                torch.load(path, map_location=cuda(), weights_only=True)
            )
        else:
            optimizer = torch.optim.LBFGS(
                model.parameters(),
                max_iter=300,
                history_size=20,
                tolerance_grad=1e-7,
                tolerance_change=1e-9,
                line_search_fn="strong_wolfe",
            )

            def closure():
                optimizer.zero_grad()
                loss = (
                    F.binary_cross_entropy_with_logits(
                        model(x).squeeze(1), train["y"], reduction="none"
                    )
                    * train["weight"]
                ).mean()
                loss = loss + penalty * model.weight.square().sum() / 2
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite GPU linear objective")
                loss.backward()
                return loss

            optimizer.step(closure)
            save_torch(path, model.state_dict())
        with torch.no_grad():
            p = model(xv).squeeze(1).sigmoid()
            score = float(
                (
                    (p - data["validation"]["y"]).square()
                    * data["validation"]["weight"]
                ).mean()
            )
        save_json(
            marker,
            {
                "validation_brier": score,
                "penalty": penalty,
                "device": "cuda:0",
                "model_sha256": digest(path),
            },
        )
        print(f"GPU linear penalty={penalty}: validation Brier={score:.6f}", flush=True)
        if best is None or score < best[0]:
            best = (score, penalty, model)
    save_torch(output / "model.pt", best[2].state_dict())
    save_json(
        output / "selected.json",
        {"penalty": best[1], "validation_brier": best[0], "device": "cuda:0"},
    )
    evaluation = data["evaluation"]
    with torch.no_grad():
        return (
            best[2](
                linear_design(
                    evaluation["numeric"], evaluation["categorical"], processor
                )
            )
            .squeeze(1)
            .sigmoid()
        )


def scores(p, y, keys):
    from torchmetrics.functional.classification import binary_auroc

    def calculate(indices):
        ix = torch.tensor(indices, device=p.device)
        q, labels = p[ix].float(), y[ix].float()
        result = {
            "rows": len(ix),
            "brier": float((q - labels).square().mean()),
            "log_loss": float(F.binary_cross_entropy(q.clamp(1e-7, 1 - 1e-7), labels)),
            "accuracy": float(((q > 0.5) == labels.bool()).float().mean()),
        }
        if labels.unique().numel() == 2:
            result["roc_auc"] = float(binary_auroc(q, labels.long()))
        return result

    groups = defaultdict(list)
    for i, row in enumerate(keys):
        groups[row["region"] + "/" + row["population"]].append(i)
    return {
        "all_regional": calculate(list(range(len(p)))),
        "groups": {key: calculate(indices) for key, indices in groups.items()},
    }


def fit_unit(root, task, representation, family, config):
    directory = root / "inputs" / task
    unit = root / "fits" / f"{task}-{representation}-{family}"
    unit.mkdir(parents=True, exist_ok=True)
    identity = {
        "inputs": digest(directory / "prepared.json"),
        "config": digest(root / "config.json"),
        "environment": digest(root / "environment.lock"),
        "code": code_identity(),
        "seed": config["seed"],
        "task": task,
        "representation": representation,
        "family": family,
    }
    if (unit / "identity.json").exists():
        if json.loads((unit / "identity.json").read_text()) != identity:
            raise RuntimeError("unit identity changed: new version required")
    else:
        save_json(unit / "identity.json", identity)
    if (unit / "completed.json").exists():
        previous = json.loads((unit / "completed.json").read_text())
        freeze = previous["freeze"]
        if digest(freeze["model_checkpoint"]) != freeze["model_sha256"]:
            raise RuntimeError("completed model artifact changed")
        if (
            digest(
                Path(freeze["model_checkpoint"]).parent.parent / "predictions.npz"
                if family != "linear"
                else Path(freeze["model_checkpoint"]).parent / "predictions.npz"
            )
            != freeze["prediction_sha256"]
        ):
            raise RuntimeError("completed prediction artifact changed")
        print(f"Reusing completed {unit.name}", flush=True)
        return previous
    pl.seed_everything(config["seed"], workers=True)
    torch.set_float32_matmul_precision("high")
    torch.cuda.reset_peak_memory_stats()
    data, keys, processor = prepare_data(directory, task, representation)
    save_torch(unit / "processor.pt", processor)
    batch_size = config.get(family, {}).get("batch_size", 2048)
    if family == "tabm" and task == "series" and representation == "individual":
        batch_size = config["tabm"].get("series_individual_batch_size", batch_size)
    started = time.monotonic()
    for attempt in range(1, 4):
        output = unit / f"attempt-{attempt}"
        output.mkdir(exist_ok=True)
        if (output / "failed.json").exists():
            failure = json.loads((output / "failed.json").read_text())
            if failure.get("cuda_oom") and family != "linear":
                batch_size = max(1, batch_size // 2)
                continue
            raise RuntimeError(
                "preserved terminal fit failure; inspect before a versioned repair"
            )
        actual = {**identity, "batch_size": batch_size, "attempt": attempt}
        if (output / "identity.json").exists() and json.loads(
            (output / "identity.json").read_text()
        ) != actual:
            raise RuntimeError("attempt identity mismatch")
        save_json(output / "identity.json", actual)
        save_json(
            root / "progress.json",
            {
                "status": "fitting",
                "unit": unit.name,
                "attempt": attempt,
                "batch_size": batch_size,
                "device": "cuda:0",
            },
        )
        print(
            f"START {unit.name}; batch={batch_size}; CUDA; max epochs={config['maximum_epochs']}",
            flush=True,
        )
        module = trainer = probabilities = train_loader = val_loader = None
        try:
            if family == "linear":
                probabilities = fit_linear(data, processor, config, output)
                best_path = str(output / "model.pt")
            else:
                module = Predictor(
                    family,
                    data["train"]["numeric"].shape[1],
                    processor["cardinalities"],
                    processor["swap"],
                    config,
                )
                train_loader, generator = loader(
                    data["train"], batch_size, True, config["seed"]
                )
                val_loader, _ = loader(
                    data["validation"], batch_size, False, config["seed"]
                )
                checkpoint = ModelCheckpoint(
                    dirpath=output / "checkpoints",
                    monitor="val_brier",
                    mode="min",
                    save_top_k=1,
                    save_last=True,
                    every_n_epochs=1,
                    filename="epoch-{epoch:03d}-brier-{val_brier:.6f}",
                )
                callbacks = [
                    EarlyStopping(
                        monitor="val_brier",
                        mode="min",
                        patience=config["early_stopping_patience"],
                        min_delta=1e-6,
                    ),
                    EpochRecord(output, config["seed"], generator, actual),
                    checkpoint,
                ]
                trainer = pl.Trainer(
                    accelerator="gpu",
                    devices=[0],
                    max_epochs=config["maximum_epochs"],
                    precision=(
                        "bf16-mixed" if torch.cuda.is_bf16_supported() else "32-true"
                    ),
                    callbacks=callbacks,
                    plugins=[DurableCheckpointIO()],
                    logger=[
                        CSVLogger(output, name="epochs"),
                        TensorBoardLogger(
                            root / "tensorboard",
                            name=unit.name,
                            version=f"attempt-{attempt}",
                        ),
                    ],
                    enable_progress_bar=False,
                    log_every_n_steps=10,
                    num_sanity_val_steps=0,
                )
                last = output / "checkpoints" / "last.ckpt"
                trainer.fit(
                    module,
                    train_loader,
                    val_loader,
                    ckpt_path=str(last) if last.exists() else None,
                )
                if trainer.interrupted:
                    raise KeyboardInterrupt(
                        "training interrupted; preserve last epoch for resume"
                    )
                best_path = checkpoint.best_model_path
                if not best_path:
                    raise RuntimeError("no best checkpoint saved")
                module = (
                    Predictor.load_from_checkpoint(
                        best_path, map_location=cuda(), weights_only=False
                    )
                    .to(cuda())
                    .eval()
                )
                probabilities = []
                with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()
                ):
                    values = data["evaluation"]
                    for begin in range(0, len(values["numeric"]), batch_size):
                        probabilities.append(
                            module.probability(
                                values["numeric"][begin : begin + batch_size],
                                values["categorical"][begin : begin + batch_size],
                            )
                        )
                probabilities = torch.cat(probabilities)
            if (
                not torch.isfinite(probabilities).all()
                or not ((probabilities >= 0) & (probabilities <= 1)).all()
            ):
                raise RuntimeError("invalid frozen forecast")
            path = output / "predictions.npz"
            with path.open("wb") as stream:
                np.savez_compressed(
                    stream, probability=probabilities.float().cpu().numpy()
                )
                stream.flush()
                os.fsync(stream.fileno())
            freeze = {
                "prediction_sha256": digest(path),
                "keys_sha256": digest(directory / "evaluation-keys.json"),
                "identity": actual,
                "model_checkpoint": best_path,
                "model_sha256": digest(best_path),
                "evaluation_targets_opened_before_freeze": False,
            }
            save_json(output / "prediction-freeze.json", freeze)
            # Evaluation labels first enter this training process after durable prediction freezing.
            target_file = directory / "evaluation-targets.npz"
            with np.load(target_file, allow_pickle=False) as targets:
                y = torch.tensor(targets["y"], device=cuda())
            if len(y) != len(probabilities):
                raise RuntimeError("frozen prediction/target key mismatch")
            result = {
                "unit": unit.name,
                "status": "completed",
                "seconds": time.monotonic() - started,
                "identity": actual,
                "freeze": freeze,
                "scores": scores(probabilities, y, keys["evaluation"]),
                "target_sha256": digest(target_file),
                "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            save_json(unit / "completed.json", result)
            print(
                f"COMPLETED {unit.name}: {json.dumps(result['scores']['all_regional'])}",
                flush=True,
            )
            return result
        except torch.OutOfMemoryError as error:
            save_json(
                output / "failed.json",
                {"reason": str(error), "cuda_oom": True, "identity": actual},
            )
            if family == "linear":
                raise RuntimeError(
                    "GPU linear full-data fit OOM; recorded terminal failure, no ineffective batch retry"
                ) from error
            print(
                f"Real fit OOM preserved in {output}; next attempt uses half batch size",
                flush=True,
            )
            module = trainer = probabilities = train_loader = val_loader = None
            gc.collect()
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
        except KeyboardInterrupt:
            save_json(
                output / "interrupted.json",
                {"identity": actual, "resume": "last completed epoch checkpoint"},
            )
            raise
        except Exception as error:
            save_json(
                output / "failed.json",
                {
                    "reason": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "cuda_oom": False,
                    "identity": actual,
                },
            )
            raise
    raise RuntimeError("three preserved real OOM attempts; cannot continue this unit")


def reuse_linear(prior, root, task, representation, config):
    prior_unit = prior / "fits" / f"{task}-{representation}-linear"
    previous = json.loads((prior_unit / "completed.json").read_text())
    previous_config = json.loads((prior / "config.json").read_text())
    if any(previous_config[k] != config[k] for k in ("seed", "linear_penalties")):
        raise RuntimeError("linear reuse numerical parameters differ")
    if digest(prior / "code/train.py") != previous["identity"]["code"]["train.py"]:
        raise RuntimeError("prior linear code identity changed")
    # Verify the actual numerical/transform code, not a caller's equivalence claim.
    names = {
        "save_torch",
        "load",
        "event_weights",
        "swap_spec",
        "swapped",
        "representation_arrays",
        "prepare_data",
        "linear_design",
        "fit_linear",
        "scores",
    }

    def functions(path):
        tree = ast.parse(path.read_text())
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        }

    if functions(prior / "code/train.py") != functions(Path(__file__)):
        raise RuntimeError("linear numerical or transform implementation differs")
    if digest(prior / "code/common.py") != digest(Path(__file__).parent / "common.py"):
        raise RuntimeError("linear shared identity/runtime code differs")
    if digest(prior / "inputs" / task / "prepared.json") != digest(
        root / "inputs" / task / "prepared.json"
    ):
        raise RuntimeError("linear prepared inputs differ")
    freeze = previous["freeze"]
    if digest(freeze["model_checkpoint"]) != freeze["model_sha256"]:
        raise RuntimeError("prior linear model changed")
    prediction = Path(freeze["model_checkpoint"]).parent / "predictions.npz"
    if digest(prediction) != freeze["prediction_sha256"]:
        raise RuntimeError("prior linear forecasts changed")
    output = root / "fits" / prior_unit.name
    output.mkdir(parents=True, exist_ok=True)
    save_json(
        output / "reuse.json",
        {
            "original_completed_record": str(prior_unit / "completed.json"),
            "original_record_sha256": digest(prior_unit / "completed.json"),
            "reason": "unchanged numerical functions, inputs, seed and linear penalties; neural autocast/resource repair only",
            "current_code": code_identity(),
            "current_config_sha256": digest(root / "config.json"),
        },
    )
    # Retain actual old fitted-code identity, never relabel it as a fresh fit.
    save_json(output / "completed.json", {**previous, "reused_from": str(prior_unit)})
    print(
        f"Reused completed CUDA {prior_unit.name} with original code identity",
        flush=True,
    )
    return previous


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reuse-inputs", type=Path)
    parser.add_argument("--reuse-linear", type=Path)
    args = parser.parse_args()
    cuda()
    config = json.loads((args.root / "config.json").read_text())
    prepared = json.loads((args.root / "inputs" / "prepared.json").read_text())
    if prepared["config_sha256"] != digest(args.root / "config.json"):
        if not args.reuse_inputs:
            raise RuntimeError("input configuration differs from fit configuration")
        prior_config = json.loads((args.reuse_inputs / "config.json").read_text())
        input_keys = (
            "tasks",
            "recent_observed_batches",
            "train_event_fraction",
            "validation_event_fraction",
            "reveal_delay_minutes",
            "prediction_lead_minutes",
            "history_policy",
            "series_roster_policy",
            "series_target",
        )
        if any(prior_config[k] != config[k] for k in input_keys):
            raise RuntimeError("input-generating policy changed")
        if prepared["config_sha256"] != digest(args.reuse_inputs / "config.json"):
            raise RuntimeError("input reuse origin configuration mismatch")
        if digest(args.root / "inputs/prepared.json") != digest(
            args.reuse_inputs / "inputs/prepared.json"
        ):
            raise RuntimeError("input reuse identity record differs")
        save_json(
            args.root / "input-reuse.json",
            {
                "from": str(args.reuse_inputs),
                "prepared_sha256": digest(args.root / "inputs/prepared.json"),
                "original_config_sha256": prepared["config_sha256"],
                "current_config_sha256": digest(args.root / "config.json"),
                "unchanged_input_policy": {k: config[k] for k in input_keys},
            },
        )
    for task in prepared["tasks"]:
        for name, expected in task["files"].items():
            if digest(args.root / "inputs" / task["task"] / name) != expected:
                raise RuntimeError("saved training input identity changed")
    device = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "name": torch.cuda.get_device_name(0),
        "vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
    }
    save_json(args.root / "device.json", device)
    completed, failed = [], []
    # Start both tasks before the more expensive attention fits.
    order = [
        (task, representation, family)
        for family in ("linear", "tabm", "ft_transformer")
        for task in config["tasks"]
        for representation in config["representations"]
    ]
    for task, representation, family in order:
        try:
            if family == "linear" and args.reuse_linear:
                completed.append(
                    reuse_linear(
                        args.reuse_linear, args.root, task, representation, config
                    )
                )
            else:
                completed.append(
                    fit_unit(args.root, task, representation, family, config)
                )
        except Exception as error:
            failed.append(
                {
                    "task": task,
                    "representation": representation,
                    "family": family,
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            print(traceback.format_exc(), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        save_json(
            args.root / "campaign-progress.json",
            {
                "completed": [r["unit"] for r in completed],
                "failed": failed,
                "scheduled_units": len(order),
            },
        )
    save_json(
        args.root / "campaign-completed.json",
        {
            "completed": [r["unit"] for r in completed],
            "failed": failed,
            "scheduled_units": len(order),
            "device": device,
            "scope": config["scope"],
        },
    )
    save_json(
        args.root / "progress.json",
        {
            "status": "completed_with_failures" if failed else "completed",
            "completed_units": len(completed),
            "failed_units": len(failed),
        },
    )


if __name__ == "__main__":
    main()
