"""Versioned saved-forecast revision analysis; no predictor refits or raw access.

Runs once into a new artifact directory. Per-task units are durable and reusable
only under identical code, methods and input identities. Numerical diagnostics
use CUDA; inputs remain read-only and incomplete attempts are retained.
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

STORE = Path(os.environ.get('VALORANT_ARTIFACT_ROOT', '.')).resolve()
CORE = STORE / 'strong-predictors-20261010-v6'
ONLINE = STORE / 'online-rich-predictor-20261010-v4'
DYNAMIC = STORE / 'dynamic-predictor-20261010-v1'
ELO = STORE / 'dynamic-matched-baselines-20261010-v1'
LATER = STORE / 'later-evaluation-20261011-v1'
TASKS = ('map_winner', 'series_winner')
PHASES = ('train', 'stopping', 'selection', 'calibration', 'evaluation')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class Inputs:
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = Path(path)
        actual = sha(path)
        if expected is not None and actual != expected:
            raise RuntimeError('Input identity mismatch: ' + str(path))
        self.hashes[str(path)] = actual
        return path

    def json(self, path, expected=None):
        return read(self.bind(path, expected))


def load_core(task, inputs):
    cfg = inputs.json(CORE / 'config.json')
    keys, labels = [], []
    probabilities = collections.defaultdict(list)
    decisions = collections.defaultdict(list)
    timing, support, calibrators = [], [], []
    online_index = inputs.json(ELO / 'keys.json')
    elo_receipt = inputs.json(ELO / 'frozen.json')
    inputs.bind(ELO / 'keys.json', elo_receipt['keys'])
    elo_values = np.load(inputs.bind(ELO / 'predictions.npz', elo_receipt['predictions']))['probability']
    lookup = {tuple(k): i for i, k in enumerate(online_index)}
    if len(lookup) != len(online_index):
        raise RuntimeError('Duplicate matched Elo keys')
    for fold in cfg['folds']:
        name = fold['name']
        block = CORE / 'blocks' / name
        prepared = inputs.json(block / 'inputs/prepared.json')
        phase_keys = {}
        for phase in PHASES:
            file = f'{task}/{phase}-keys.json'
            phase_keys[phase] = inputs.json(block / 'inputs' / file, prepared['files'][file])
        kk = phase_keys['evaluation']
        history = [k['history_end_timestamp'] for k in kk if k['history_end_timestamp'] is not None]
        if any(k['history_end_timestamp'] is not None and k['history_end_timestamp'] >= k['prediction_cutoff'] for k in kk):
            raise RuntimeError('Saved history cutoff violation')
        phase_summary = {}
        for phase, pk in phase_keys.items():
            phase_summary[phase] = dict(rows=len(pk), events=len({k['event_id'] for k in pk}),
                                       first_anchor=min((k['anchor'] for k in pk), default=None),
                                       last_anchor=max((k['anchor'] for k in pk), default=None))
        ages = [(k['prediction_cutoff'] - k['history_end_timestamp']) / 86400 for k in kk if k['history_end_timestamp'] is not None]
        timing.append(dict(block=name, task=task, phases=phase_summary,
                           state_age_days_quantiles=np.quantile(ages, [0, .25, .5, .75, 1]).tolist() if ages else None,
                           admitted_history_end=min(history) if history else None,
                           admitted_history_latest=max(history) if history else None,
                           absent_state_rows=len(kk) - len(history)))
        lp = CORE / 'scored-labels' / name / (task + '.npy')
        label_receipt = inputs.json(lp.with_suffix('.json'))
        yy = np.load(inputs.bind(lp, label_receipt['sha256']))
        inputs.bind(block / 'inputs' / task / 'evaluation-keys.json', label_receipt['keys_sha256'])
        z = np.load(inputs.bind(block / 'selector' / task / 'predictions.npz'))
        for label, field in [('selector', 'evaluation_calibrated'), ('block_elo', 'evaluation_elo_calibrated'),
                             ('prior', 'training_prior'), ('chance', 'chance')]:
            probabilities[label].extend(z[field])
            decisions[label].extend(z[field] > .5)
        for label, root in [('rich', ONLINE / 'rich_residual'), ('earlier', DYNAMIC)]:
            directory = root / 'blocks' / name / task
            receipt = inputs.json(directory / 'frozen.json')
            z = np.load(inputs.bind(directory / 'predictions.npz', receipt['predictions']))
            probabilities[label].extend(z['probability'])
            decisions[label].extend(z['pick'])
            if label == 'earlier':
                probabilities['dynamic'].extend(z['dynamic_probability'])
                decisions['dynamic'].extend(z['dynamic_pick'])
        ep = elo_values[[lookup[task, k['game_id']] for k in kk], 0]
        probabilities['online_elo'].extend(ep)
        decisions['online_elo'].extend(ep > .5)
        parent_file = block / f'removals-{task}-parent.json'
        support.append(dict(block=name, task=task, parent=inputs.json(parent_file) if parent_file.exists() else None))
        for marker in sorted((block / 'packages/core' / task).glob('*/selection.json')):
            rec = inputs.json(marker)
            calibrators.append(dict(block=name, task=task, package=marker.parent.name,
                                    calibration=rec.get('calibrator'), fallback=rec.get('fallback'), reason=rec.get('reason')))
        keys.extend(dict(k, block=name) for k in kk)
        labels.extend(yy)
    if len({k['game_id'] for k in keys}) != len(keys):
        raise RuntimeError('Repeated target key across outer blocks')
    if any(int(k['team_a_id']) >= int(k['team_b_id']) for k in keys):
        raise RuntimeError('Canonical team identity mismatch')
    p = {k: np.asarray(v, dtype=np.float64) for k, v in probabilities.items()}
    d = {k: np.asarray(v, dtype=np.int8) for k, v in decisions.items()}
    y = np.asarray(labels, dtype=np.float64)
    if any(len(a) != len(keys) or not np.isfinite(a).all() for a in p.values()):
        raise RuntimeError('Nonfinite or misaligned probabilities')
    return keys, y, p, d, dict(timing=timing, removal_support=support, calibrators=calibrators)


def grouping(keys):
    events = sorted({str(k['event_id']) for k in keys})
    index = {e: i for i, e in enumerate(events)}
    ix = np.array([index[str(k['event_id'])] for k in keys])
    counts = np.bincount(ix, minlength=len(events))
    return events, ix, counts


def metrics(keys, y, p, decision):
    events, ix, counts = grouping(keys)
    w = 1 / len(events) / counts[ix]
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    ties = p == .5
    accuracy = (decision == y).astype(float)
    tied_accuracy = np.where(ties, .5, accuracy)
    return dict(rows=len(y), events=len(events), event_size_quantiles=np.quantile(counts, [0, .25, .5, .75, 1]).tolist(),
                row_a_fraction=float(y.mean()), event_a_fraction=float(w @ y),
                row_accuracy=float(accuracy.mean()), event_accuracy=float(w @ accuracy),
                row_fixed_half_accuracy=float(((p > .5) == y).mean()),
                event_fixed_half_accuracy=float(w @ ((p > .5) == y)),
                row_tie_averaged_accuracy=float(tied_accuracy.mean()), event_tie_averaged_accuracy=float(w @ tied_accuracy),
                exact_ties=int(ties.sum()), tie_row_mass=float(ties.mean()), tie_event_mass=float(w @ ties),
                tied_a_fraction=float(y[ties].mean()) if ties.any() else None,
                event_brier=float(w @ ((clipped - y) ** 2)), row_brier=float(((clipped - y) ** 2).mean()),
                event_log_loss=float(w @ (-y * np.log(clipped) - (1-y) * np.log1p(-clipped))))


def paired(keys, y, pa, pb, draws, seed):
    events, ix, counts = grouping(keys)
    values = []
    for p in (pa, pb):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        values.append(((p-y)**2, -y*np.log(p)-(1-y)*np.log1p(-p)))
    sums = np.stack([np.bincount(ix, weights=values[0][j]-values[1][j], minlength=len(events)) for j in range(2)])
    means = torch.tensor(sums / counts, device='cuda', dtype=torch.float64)
    totals = torch.tensor(sums, device='cuda', dtype=torch.float64)
    n = torch.tensor(counts, device='cuda', dtype=torch.float64)
    gen = torch.Generator(device='cuda').manual_seed(seed)
    boots, rowboots = [], []
    for start in range(0, draws, 512):
        indices = torch.randint(len(events), (min(512, draws-start), len(events)), device='cuda', generator=gen)
        boots.append(means[:, indices].mean(-1))
        rowboots.append(totals[:, indices].sum(-1) / n[indices].sum(-1))
    quantiles = torch.tensor([.025, .975], device='cuda', dtype=torch.float64)
    ci = torch.quantile(torch.cat(boots, 1), quantiles, dim=1).cpu().numpy()
    rowci = torch.quantile(torch.cat(rowboots, 1), quantiles, dim=1).cpu().numpy()
    return {name: dict(event_difference=float((sums[j]/counts).mean()), event_ci95=ci[:, j].tolist(),
                       row_difference=float(sums[j].sum()/counts.sum()), row_ci95=rowci[:, j].tolist(),
                       draws=draws, seed=seed, events=len(events)) for j, name in enumerate(('brier', 'log_loss'))}


def diagnostic_bootstrap(keys, y, p, draws, seed, mirror):
    events, ix, counts = grouping(keys)
    p = np.clip(p, 1e-7, 1-1e-7)
    if np.ptp(p) < 1e-12:
        return dict(status='nonidentifiable_constant_forecast', coefficient_intervals=None)
    if mirror:
        ix = np.tile(ix, 2)
        p, y = np.r_[p, 1-p], np.r_[y, 1-y]
        counts = counts * 2
    z = torch.tensor(np.log(p)-np.log1p(-p), device='cuda', dtype=torch.float64)
    yt = torch.tensor(y, device='cuda', dtype=torch.float64)
    event_ix = torch.tensor(ix, device='cuda')
    counts_t = torch.tensor(counts, device='cuda', dtype=torch.float64)
    gen = torch.Generator(device='cuda').manual_seed(seed)
    estimates, eces, valid = [], [], []
    # First row is the full observed cohort. All later rows resample events.
    for start in range(-1, draws, 16):
        size = min(16, draws-start)
        if start == -1:
            samples = torch.randint(len(events), (size-1, len(events)), generator=gen, device='cuda')
            multiplicity = torch.nn.functional.one_hot(samples, len(events)).sum(1).double()
            multiplicity = torch.cat((torch.ones((1, len(events)), device='cuda', dtype=torch.float64), multiplicity))
        else:
            samples = torch.randint(len(events), (size, len(events)), generator=gen, device='cuda')
            multiplicity = torch.nn.functional.one_hot(samples, len(events)).sum(1).double()
        w = (multiplicity/counts_t/len(events))[:, event_ix]
        a = torch.zeros(size, device='cuda', dtype=torch.float64)
        b = torch.ones_like(a)
        alive = ((w @ yt) > 0) & ((w @ yt) < 1)
        for _ in range(60):
            eta = a[:, None]+b[:, None]*z
            fitted = eta.sigmoid()
            residual = (fitted-yt)*w
            g0, g1 = residual.sum(1), (residual*z).sum(1)
            curvature = w*fitted*(1-fitted)
            h0, h1, h2 = curvature.sum(1), (curvature*z).sum(1), (curvature*z*z).sum(1)
            determinant = h0*h2-h1*h1
            safe = determinant > 1e-14
            alive &= safe
            denominator = determinant.clamp_min(1e-14)
            da, db = (h2*g0-h1*g1)/denominator, (-h1*g0+h0*g1)/denominator
            scale = torch.maximum(torch.maximum(da.abs(), db.abs())/5, torch.ones_like(a))
            da, db = da/scale, db/scale
            current = ((torch.nn.functional.softplus(eta)-yt*eta)*w).sum(1)
            rate = torch.ones_like(a)
            for _ in range(16):
                proposed = (a-rate*da)[:, None]+(b-rate*db)[:, None]*z
                objective = ((torch.nn.functional.softplus(proposed)-yt*proposed)*w).sum(1)
                rate = torch.where(objective > current+1e-12, rate*.5, rate)
            a = torch.where(alive, a-rate*da, a)
            b = torch.where(alive, b-rate*db, b)
            if max(float(g0.abs().max()), float(g1.abs().max())) < 1e-7:
                break
        fitted = (a[:, None]+b[:, None]*z).sigmoid()
        gradient = torch.maximum(((fitted-yt)*w).sum(1).abs(), (((fitted-yt)*w)*z).sum(1).abs())
        good = alive & (gradient <= 1e-5) & (a.abs() < 100) & (b.abs() < 100)
        estimates.append(torch.stack((a, b), 1).cpu().numpy())
        valid.append(good.cpu().numpy())
        bin_eces = []
        for bins in (5, 10, 20):
            bin_ix = np.minimum((p*bins).astype(int), bins-1)
            biases = np.zeros((len(events), bins))
            np.add.at(biases, (ix, bin_ix), p-y)
            mean_bias = torch.tensor(biases / counts[:, None], device='cuda', dtype=torch.float64)
            bin_eces.append((multiplicity @ mean_bias / len(events)).abs().sum(1).cpu().numpy())
        eces.append(np.stack(bin_eces, 1))
    estimates, valid, eces = np.concatenate(estimates), np.concatenate(valid), np.concatenate(eces)
    if len(estimates) != draws+1:
        raise RuntimeError('Incorrect diagnostic resample count')
    bad = int((~valid[1:]).sum())
    return dict(status='resolved' if bad <= .05*draws and valid[0] else 'unresolved_diagnostic_draw_failures',
                mirror=mirror, events=len(events), rows=len(y), draws=draws, seed=seed,
                point=estimates[0].tolist() if valid[0] else None, point_converged=bool(valid[0]),
                failed_draws=bad, coefficient_ci95=np.quantile(estimates[1:][valid[1:]], [.025, .975], axis=0).T.tolist() if bad <= .05*draws and valid[0] else None,
                ece_point=dict(zip(('5', '10', '20'), eces[0].tolist())),
                ece_ci95=dict(zip(('5', '10', '20'), np.quantile(eces[1:], [.025, .975], axis=0).T.tolist())))


def analyze(task, args, inputs):
    keys, y, probabilities, decisions, metadata = load_core(task, inputs)
    scopes = {
        'broad_nonglobal': np.array([k['region'] != 'global' for k in keys]),
        'reviewed2025': np.array([k['region'] in ('americas', 'emea', 'pacific') and k['population'] in ('open', 'game_changers') and dt.datetime.fromtimestamp(k['anchor'], dt.timezone.utc).year == 2025 for k in keys]),
    }
    result = dict(task=task, metadata=metadata, metrics=[], contrasts=[], reliability=[])
    for scope, mask in scopes.items():
        mask &= np.isfinite(y)
        kk = [k for k, keep in zip(keys, mask) if keep]
        yy = y[mask]
        pp = {name: p[mask] for name, p in probabilities.items()}
        for name, p in pp.items():
            result['metrics'].append(dict(scope=scope, package=name, **metrics(kk, yy, p, decisions[name][mask])))
        for a, b in [('rich', 'earlier'), ('rich', 'dynamic'), ('rich', 'online_elo'), ('block_elo', 'online_elo'), ('selector', 'online_elo')]:
            result['contrasts'].append(dict(scope=scope, a=a, b=b, **paired(kk, yy, pp[a], pp[b], args.draws, 20261016)))
        if scope == 'broad_nonglobal':
            for name in ('selector', 'block_elo', 'prior'):
                for mirror in (False, True):
                    print(task, scope, 'reliability', name, mirror, flush=True)
                    result['reliability'].append(dict(scope=scope, package=name, **diagnostic_bootstrap(kk, yy, pp[name], args.diagnostic_draws, 20261017, mirror)))
            result['block_orientation'] = [dict(block=block, **metrics([k for k, keep in zip(kk, bm) if keep], yy[bm], pp['selector'][bm], decisions['selector'][mask][bm])) for block in sorted({k['block'] for k in kk}) if (bm := np.array([k['block'] == block for k in kk])).any()]
    result['inputs'] = inputs.hashes
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--methods', type=Path, required=True)
    ap.add_argument('--draws', type=int, default=131072)
    ap.add_argument('--diagnostic-draws', type=int, default=2048)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU numerical fallback')
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    identity = dict(code_sha256=sha(__file__), methods_sha256=sha(args.methods), draws=args.draws,
                    diagnostic_draws=args.diagnostic_draws, torch=torch.__version__, numpy=np.__version__)
    identity_file = args.output/'diagnostics-identity.json'
    if identity_file.exists() and read(identity_file) != identity:
        raise RuntimeError('Version changed: preserve this output and use a new root')
    save(identity_file, identity)
    for task in TASKS:
        output = args.output/(task+'-diagnostics.json')
        if output.exists():
            previous = read(output)
            for p, h in previous['inputs'].items():
                if sha(p) != h:
                    raise RuntimeError('Completed unit input changed')
            print('REUSE', task, flush=True)
            continue
        started = time.time()
        try:
            result = analyze(task, args, Inputs())
            result['elapsed_seconds'] = time.time()-started
            result['gpu_peak_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
            save(output, result)
            print('COMPLETED', task, result['elapsed_seconds'], flush=True)
        except Exception as error:
            save(args.output/f'failed-{task}-{time.time_ns()}.json', dict(error=repr(error), identity=identity))
            raise
    save(args.output/'diagnostics-completion.json', dict(identity=identity, outputs={task: sha(args.output/(task+'-diagnostics.json')) for task in TASKS}))


if __name__ == '__main__':
    main()
