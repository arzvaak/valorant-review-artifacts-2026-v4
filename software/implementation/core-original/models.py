"""Established CUDA model adapters, actual epochs and durable candidate receipts."""

import gc, json, math, time, traceback, subprocess, threading, os
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from rtdl_revisiting_models import MLP, FTTransformer
from rtdl_num_embeddings import LinearReLUEmbeddings
from tabm import TabM
import gpu_runtime as rt
from common import *
from preprocess import swap
from set_modules import SAB, PMA


class Net(pl.LightningModule):
    def __init__(self, family, width, player_width, proc, params, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.family = family
        self.proc = proc
        self.params = params
        self.cfg = cfg
        if family == "mlp":
            self.model = MLP(
                d_in=width,
                d_out=1,
                n_blocks=3,
                d_block=params["width"],
                dropout=params["dropout"],
            )
        elif family == "tabm":
            self.model = TabM.make(
                n_num_features=width,
                cat_cardinalities=None,
                d_out=1,
                n_blocks=params["blocks"],
                d_block=params["width"],
                k=16,
                dropout=params["dropout"],
                num_embeddings=LinearReLUEmbeddings(width, 16),
            )
        elif family == "ft_transformer":
            options = FTTransformer.get_default_kwargs()
            options.update(
                n_blocks=params["blocks"],
                d_block=params["width"],
                attention_dropout=params["dropout"],
                _is_default=False,
            )
            self.model = FTTransformer(
                n_cont_features=width, cat_cardinalities=[], d_out=1, **options
            )
        else:
            d = params["width"]
            if family == "deep_sets":
                self.encoder = MLP(
                    d_in=player_width,
                    d_out=d,
                    n_blocks=2,
                    d_block=d,
                    dropout=params["dropout"],
                )
                self.pool = None
            else:
                self.encoder = torch.nn.Sequential(
                    SAB(player_width, d, 4, ln=True), SAB(d, d, 4, ln=True)
                )
                self.pool = PMA(d, 4, 1, ln=True)
            self.model = MLP(
                d_in=width + 2 * d,
                d_out=1,
                n_blocks=2,
                d_block=d,
                dropout=params["dropout"],
            )

    def logits(self, x, players=None):
        if self.family in ("deep_sets", "set_transformer"):
            n = len(players)
            v = self.encoder(players.reshape(n * 2, 5, -1))
            z = v.sum(1) if self.pool is None else self.pool(v).squeeze(1)
            x = torch.cat((x, z.reshape(n, -1)), 1)
        return (
            self.model(x, None)
            if self.family in ("tabm", "ft_transformer")
            else self.model(x)
        )

    def probability(self, x, players=None):
        p = self.logits(x, players).float().sigmoid().reshape(len(x), -1).mean(1)
        q = (
            self.logits(
                swap(x, self.proc), players.flip(1) if players is not None else None
            )
            .float()
            .sigmoid()
            .reshape(len(x), -1)
            .mean(1)
        )
        return (p + 1 - q) / 2

    def training_step(self, b, batch_idx):
        z = self.logits(b["numeric"], b.get("players")).float().reshape(len(b["y"]), -1)
        y = b["y"][:, None].expand_as(z)
        v = (
            F.binary_cross_entropy_with_logits(z, y, reduction="none").mean(1)
            * b["weight"]
        ).mean()
        if not torch.isfinite(v):
            raise RuntimeError("Nonfinite CUDA fit")
        self.log("train_loss", v, on_epoch=True, on_step=False, batch_size=len(y))
        return v

    def validation_step(self, b, batch_idx):
        p = (
            self.logits(b["numeric"], b.get("players"))
            .float()
            .sigmoid()
            .reshape(len(b["y"]), -1)
            .mean(1)
            .clamp(1e-7, 1 - 1e-7)
        )
        with torch.autocast("cuda", enabled=False):
            ll = (
                F.binary_cross_entropy(p, b["y"].float(), reduction="none")
                * b["weight"]
            ).mean()
        self.log("val_log_loss", ll, on_step=False, on_epoch=True, batch_size=len(p))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.0003, weight_decay=0.001)

    def on_save_checkpoint(self, c):
        c["cuda_rng"] = torch.cuda.get_rng_state_all()
        c["cpu_rng"] = torch.get_rng_state()

    def on_load_checkpoint(self, c):
        if "cuda_rng" in c:
            torch.cuda.set_rng_state_all([x.cpu() for x in c["cuda_rng"]])
            torch.set_rng_state(c["cpu_rng"].cpu())


class Epoch(pl.Callback):
    def __init__(self, path, identity):
        self.path = path
        self.identity = identity
        self.started = 0

    def on_train_epoch_start(self, t, m):
        self.started = time.monotonic()
        t.train_dataloader.sampler.sampler.generator.manual_seed(
            self.identity["seed"] + t.current_epoch
        )
        if not all(p.is_cuda for p in m.parameters()):
            raise RuntimeError("Neural parameters moved to CPU")

    def on_train_epoch_end(self, t, m):
        sec = time.monotonic() - self.started
        r = {
            "epoch": t.current_epoch + 1,
            "seconds": sec,
            "examples_per_second": len(t.train_dataloader.dataset) / sec,
            "metrics": {
                k: float(v.detach().cpu()) for k, v in t.callback_metrics.items()
            },
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "reserved_gib": torch.cuda.memory_reserved() / 2**30,
            "identity": self.identity,
        }
        save_json(self.path / "progress.json", r)
        with (self.path / "epoch-telemetry.jsonl").open("a") as f:
            f.write(json.dumps(r) + "\n")


def loader(v, batch, shuffle, seed):
    data = {"numeric": v["x"], "weight": v["weight"], "y": v["y"]}
    if "players" in v:
        data["players"] = v["players"]
    return rt.loader(data, batch, shuffle, seed)[0]


def neural(data, proc, path, family, params, seed, cfg):
    batch = {
        "mlp": 1024,
        "tabm": 256,
        "ft_transformer": 64,
        "deep_sets": 256,
        "set_transformer": 128,
    }[family]
    train = data["train"]
    mirror = {
        "x": torch.cat((train["x"], swap(train["x"], proc))),
        "y": torch.cat((train["y"], 1 - train["y"])),
        "weight": train["weight"].repeat(2),
    }
    if "players" in train:
        mirror["players"] = torch.cat((train["players"], train["players"].flip(1)))
    for a in range(1, 9):
        out = path / f"neural-attempt-{a}"
        out.mkdir(exist_ok=True)
        if (out / "failed.json").exists():
            if not read(out / "failed.json")["cuda_oom"]:
                raise RuntimeError("Preserved terminal candidate failure")
            batch //= 2
            continue
        if batch < 16:
            raise RuntimeError("OOM batch floor reached")
        pl.seed_everything(seed, workers=True)
        torch.set_float32_matmul_precision("high")
        module = trainer = None
        try:
            pw = train["players"].shape[-1] if "players" in train else 0
            module = Net(family, train["x"].shape[1], pw, proc, params, cfg)
            stopping = data["stopping"]
            stopmirror = {
                "x": torch.cat((stopping["x"], swap(stopping["x"], proc))),
                "y": torch.cat((stopping["y"], 1 - stopping["y"])),
                "weight": stopping["weight"].repeat(2),
            }
            if "players" in stopping:
                stopmirror["players"] = torch.cat(
                    (stopping["players"], stopping["players"].flip(1))
                )
            cp = ModelCheckpoint(
                dirpath=out / "checkpoints",
                monitor="val_log_loss",
                mode="min",
                save_top_k=1,
                save_last=True,
                filename="epoch-{epoch:03d}",
            )
            trainer = pl.Trainer(
                accelerator="gpu",
                devices=[0],
                precision="bf16-mixed",
                max_epochs=cfg["maximum_epochs"],
                callbacks=[
                    cp,
                    EarlyStopping(
                        monitor="val_log_loss", patience=cfg["patience"], min_delta=1e-6
                    ),
                    Epoch(out, {"seed": seed, "batch": batch, "family": family}),
                ],
                plugins=[rt.DurableCheckpointIO()],
                logger=CSVLogger(out, name="epochs"),
                enable_progress_bar=False,
                enable_model_summary=False,
                num_sanity_val_steps=0,
                log_every_n_steps=20,
            )
            last = out / "checkpoints/last.ckpt"
            trainer.fit(
                module,
                loader(mirror, batch, True, seed),
                loader(stopmirror, batch, False, seed),
                ckpt_path=str(last) if last.exists() else None,
            )
            if trainer.interrupted:
                raise KeyboardInterrupt("Retain last checkpoint")
            modelpath = Path(cp.best_model_path)
            module = (
                Net.load_from_checkpoint(
                    modelpath, map_location=cuda(), weights_only=False
                )
                .to(cuda())
                .eval()
            )
            pred = {}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for phase in PHASES:
                    v = data[phase]
                    pp = []
                    for i in range(0, len(v["x"]), batch):
                        pp.append(
                            module.probability(
                                v["x"][i : i + batch],
                                v["players"][i : i + batch] if "players" in v else None,
                            )
                        )
                    pred[phase] = (
                        torch.cat(pp).float().cpu().numpy()
                        if pp
                        else np.empty(0, dtype="float32")
                    )
                # On real rows only: direct runtime invariance of the player pooling.
                if "players" in data["evaluation"] and len(data["evaluation"]["x"]):
                    v = data["evaluation"]
                    n = min(16, len(v["x"]))
                    p = module.probability(v["x"][:n], v["players"][:n])
                    q = module.probability(v["x"][:n], v["players"][:n].flip(2))
                    err = float((p - q).abs().max())
                    save_json(
                        out / "player-permutation.json",
                        {"real_rows": n, "maximum_error": err},
                    )
                    if err > 0.002:
                        raise RuntimeError(
                            "Player permutation invariance failed on real rows"
                        )
            return (
                pred,
                modelpath,
                {
                    "batch": batch,
                    "epochs": trainer.current_epoch,
                    "stopping_log_loss": float(cp.best_model_score),
                    "cuda": True,
                },
            )
        except torch.OutOfMemoryError as e:
            save_json(
                out / "failed.json",
                {"cuda_oom": True, "reason": str(e), "batch": batch},
            )
            batch //= 2
        except KeyboardInterrupt:
            raise
        except Exception as e:
            save_json(
                out / "failed.json",
                {
                    "cuda_oom": False,
                    "reason": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
        finally:
            module = trainer = None
            gc.collect()
            torch.cuda.empty_cache()
    raise RuntimeError("OOM attempts exhausted")


def linear(data, proc, path, params, cfg):
    x = (data["train"]["x"] - swap(data["train"]["x"], proc)).double() / 2
    y = data["train"]["y"].double()
    w = data["train"]["weight"].double()
    model = torch.nn.Linear(
        x.shape[1], 1, bias=False, device=cuda(), dtype=torch.float64
    )
    torch.nn.init.zeros_(model.weight)
    optim = torch.optim.LBFGS(
        model.parameters(),
        max_iter=10000,
        tolerance_grad=1e-8,
        tolerance_change=1e-12,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optim.zero_grad()
        v = (
            F.binary_cross_entropy_with_logits(model(x).flatten(), y, reduction="none")
            * w
        ).sum() / w.sum() + params["penalty"] * model.weight.square().sum() / 2
        if not torch.isfinite(v):
            raise RuntimeError("Nonfinite logistic fit")
        v.backward()
        return v

    optim.step(closure)
    closure()
    if (
        optim.state[model.weight].get("n_iter", 0) >= 10000
        and model.weight.grad.abs().max() > 1e-6
    ):
        raise RuntimeError("Logistic optimization budget exhausted")
    out = path / "model.pt"
    rt.save_torch(
        out,
        {
            "coefficient": model.weight.detach().flatten(),
            "proc": proc,
            "params": params,
        },
    )
    with torch.no_grad():
        pred = {
            p: ((v["x"] - swap(v["x"], proc)).double() / 2 @ model.weight.flatten())
            .sigmoid()
            .float()
            .cpu()
            .numpy()
            for p, v in data.items()
        }
    return (
        pred,
        out,
        {"cuda": True, "iterations": optim.state[model.weight].get("n_iter", 0)},
    )


def tree(data, proc, path, family, params, seed, cfg):
    tx = data["train"]["x"]
    x = torch.cat((tx, swap(tx, proc)))
    y = torch.cat((data["train"]["y"], 1 - data["train"]["y"]))
    w = data["train"]["weight"].repeat(2)
    sx = data["stopping"]["x"]
    sy = data["stopping"]["y"]
    sw = data["stopping"]["weight"]
    mirrors = torch.cat((sx, swap(sx, proc)))
    ys = torch.cat((sy, 1 - sy))
    ws = sw.repeat(2)
    if family == "xgboost":
        import xgboost as xgb

        if not xgb.build_info().get("USE_CUDA"):
            raise RuntimeError("XGBoost build lacks CUDA")
        d = xgb.QuantileDMatrix(x, label=y, weight=w)
        ds = xgb.QuantileDMatrix(mirrors, label=ys, weight=ws, ref=d)
        model = xgb.train(
            {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "device": "cuda:0",
                "tree_method": "hist",
                "max_depth": params["depth"],
                "min_child_weight": params["min_child"],
                "eta": 0.03,
                "lambda": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "seed": seed,
                "nthread": 4,
            },
            d,
            num_boost_round=2000,
            evals=[(ds, "stopping")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )
        if (
            json.loads(model.save_config())["learner"]["generic_param"]["device"]
            != "cuda:0"
        ):
            raise RuntimeError("XGBoost unexpectedly fitted on CPU")
        model = model[: model.best_iteration + 1]
        out = path / "model.ubj"
        model.save_model(out)

        # DMatrix prediction returns host arrays but fitted booster is CUDA.
        def infer(v):
            return model.predict(xgb.DMatrix(v))

        pred = {
            p: (
                (infer(v["x"]) + 1 - infer(swap(v["x"], proc))) / 2
                if len(v["x"])
                else np.empty(0)
            )
            for p, v in data.items()
        }
        rounds = model.num_boosted_rounds()
    else:
        from catboost import CatBoostClassifier, Pool

        model = CatBoostClassifier(
            task_type="GPU",
            devices="0",
            gpu_ram_part=0.6,
            loss_function="Logloss",
            eval_metric="Logloss",
            iterations=2000,
            learning_rate=0.03,
            depth=params["depth"],
            l2_leaf_reg=params["l2"],
            random_seed=seed,
            thread_count=4,
            allow_writing_files=False,
            verbose=False,
            od_type="Iter",
            od_wait=100,
        )
        model.fit(
            Pool(x.cpu().numpy(), y.cpu().numpy(), weight=w.cpu().numpy()),
            eval_set=Pool(
                mirrors.cpu().numpy(), ys.cpu().numpy(), weight=ws.cpu().numpy()
            ),
            use_best_model=True,
        )
        if model.get_all_params()["task_type"] != "GPU":
            raise RuntimeError("CatBoost unexpectedly fitted on CPU")
        out = path / "model.cbm"
        model.save_model(out)
        rounds = model.tree_count_
        pred = {
            p: (
                (
                    model.predict_proba(v["x"].cpu().numpy())[:, 1]
                    + 1
                    - model.predict_proba(swap(v["x"], proc).cpu().numpy())[:, 1]
                )
                / 2
                if len(v["x"])
                else np.empty(0)
            )
            for p, v in data.items()
        }
    return (
        pred,
        out,
        {
            "cuda_training": True,
            "rounds": rounds,
            "inference": "library CUDA XGBoost; CatBoost native host inference",
            "gpu_nondeterminism": family == "catboost",
        },
    )


def candidate(data, keys, proc, path, family, params, seed, cfg, identity):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    marker = path / "frozen.json"
    actual = {**identity, "family": family, "parameters": params, "seed": seed}
    if marker.exists():
        r = read(marker)
        if r["identity"] != actual:
            raise RuntimeError("Candidate identity changed")
        check(r["model"], r["model_sha256"])
        check(path / "predictions.npz", r["prediction_sha256"])
        check(path / "processor.json", r["processor_sha256"])
        return r
    if (path / "failed.json").exists():
        r = read(path / "failed.json")
        if r["identity"] != actual:
            raise RuntimeError("Failed candidate identity changed")
        return r
    save_json(path / "processor.json", proc)
    from recovery import reuse

    reused = reuse(path, actual, cfg, "candidate")
    if reused is not None:
        return reused
    n, record = attempt(path, actual)
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    monitor = MemoryMonitor(path)
    monitor.start()
    try:
        if family == "linear":
            pred, model, telemetry = linear(data, proc, path, params, cfg)
        elif family in ("catboost", "xgboost"):
            pred, model, telemetry = tree(data, proc, path, family, params, seed, cfg)
        else:
            pred, model, telemetry = neural(data, proc, path, family, params, seed, cfg)
        if any(
            not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() for p in pred.values()
        ):
            raise RuntimeError("Invalid forecast probability")
        save_npz(path / "predictions.npz", **pred)
        sc = loss(
            pred["selection"], data["selection"]["y"].cpu().numpy(), keys["selection"]
        )
        r = {
            "identity": actual,
            "model": str(model),
            "model_sha256": digest(model),
            "prediction_sha256": digest(path / "predictions.npz"),
            "processor_sha256": digest(path / "processor.json"),
            "selection": sc,
            "seconds": time.monotonic() - start,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "telemetry": {
                **telemetry,
                "process_vram_peak_mib_sampled": monitor.peak,
                "process_vram_samples": monitor.count,
                "process_vram_sampling_seconds": 1,
            },
            "status": "fitted",
            "evaluation_labels_read": False,
        }
        save_json(marker, r)
        record.update(status="completed", finished=time.time())
        save_json(path / f"attempt-{n}.json", record)
        print("FROZEN", path.name, family, seed, round(r["seconds"], 2), flush=True)
        return r
    except Exception as e:
        r = {
            "identity": actual,
            "status": "failed",
            "reason": str(e),
            "traceback": traceback.format_exc(),
            "seconds": time.monotonic() - start,
        }
        save_json(path / "failed.json", r)
        record.update(status="failed", finished=time.time())
        save_json(path / f"attempt-{n}.json", record)
        print("FAILED", family, params, repr(e), flush=True)
        return r
    finally:
        monitor.stop()
        gc.collect()
        torch.cuda.empty_cache()


class MemoryMonitor:
    def __init__(self, path):
        self.path = path
        self.peak = 0
        self.count = 0
        self.done = threading.Event()
        self.thread = None

    def start(self):
        def loop():
            while not self.done.is_set():
                try:
                    text = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-compute-apps=pid,used_gpu_memory",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        timeout=5,
                    )
                    for line in text.splitlines():
                        pid, value = line.split(",")
                        if int(pid) == os.getpid():
                            value = int(value.strip())
                            self.peak = max(self.peak, value)
                            self.count += 1
                            with (self.path / "process-vram.jsonl").open("a") as f:
                                f.write(
                                    json.dumps(
                                        {
                                            "time": time.time(),
                                            "pid": os.getpid(),
                                            "mib": value,
                                        }
                                    )
                                    + "\n"
                                )
                except Exception:
                    pass
                self.done.wait(1)

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.done.set()
        self.thread.join(timeout=6)
