"""Post-freeze later-cohort scoring with the finite twelve-contrast family."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta
from sklearn.metrics import roc_auc_score
from later_reference_state import read, sha


def event_weights(keys):
    counts = collections.Counter(k['event_id'] for k in keys)
    return np.array([1 / len(counts) / counts[k['event_id']] for k in keys])


def metrics(probability, pick, labels, keys, reliability=True):
    if not len(labels):
        return {'status': 'no_valid_targets', 'rows': 0, 'events': 0}
    weight = event_weights(keys)
    p = np.clip(probability.astype('float64'), 1e-7, 1 - 1e-7)
    correct = pick == labels
    losses = {'brier': (p - labels) ** 2, 'log_loss': -labels * np.log(p) - (1 - labels) * np.log1p(-p)}
    result = {'status': 'scored', 'rows': len(labels), 'events': len({k['event_id'] for k in keys}), 'matches': len({k['match_id'] for k in keys}), 'equal_event_brier': float(weight @ losses['brier']), 'equal_event_log_loss': float(weight @ losses['log_loss']), 'row_brier': float(losses['brier'].mean()), 'row_log_loss': float(losses['log_loss'].mean()), 'equal_event_accuracy': float(weight @ correct), 'row_accuracy': float(correct.mean()), 'row_team_a_win_fraction': float(labels.mean()), 'equal_event_team_a_win_fraction': float(weight @ labels), 'balanced_accuracy': float(np.mean([correct[labels == v].mean() for v in (0, 1)])) if len(np.unique(labels)) == 2 else None, 'equal_event_weighted_roc_auc': float(roc_auc_score(labels, p, sample_weight=weight)) if len(np.unique(labels)) == 2 else None}
    for bins in (5, 10, 20):
        ids = np.minimum((p * bins).astype(int), bins - 1)
        result['ece_' + str(bins)] = float(sum(abs(weight[ids == i] @ (p[ids == i] - labels[ids == i])) for i in range(bins)))
    if not reliability:
        return result
    if len(np.unique(labels)) < 2 or np.ptp(p) < 1e-12:
        result['calibration_diagnostic'] = {'status': 'unresolved_one_class_or_constant_forecast', 'intercept': None, 'slope': None}
        return result
    z = torch.logit(torch.tensor(p, dtype=torch.float64, device='cuda'))
    y = torch.tensor(labels, dtype=torch.float64, device='cuda')
    w = torch.tensor(weight, dtype=torch.float64, device='cuda')
    parameters = torch.tensor([0., 1.], dtype=torch.float64, device='cuda', requires_grad=True)
    optimizer = torch.optim.LBFGS([parameters], max_iter=500, line_search_fn='strong_wolfe', tolerance_grad=1e-9)
    def closure():
        optimizer.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(parameters[0] + parameters[1] * z, y, reduction='none') * w).sum()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite reporting-only calibration diagnostic')
        loss.backward()
        return loss
    try:
        optimizer.step(closure)
        a = parameters.detach().cpu().numpy()
        optimizer.zero_grad()
        closure()
        grad = parameters.grad.detach().abs().max().item()
        resolved = bool(np.isfinite(a).all() and grad <= 1e-5 and np.abs(a).max() < 100)
        result['calibration_diagnostic'] = {'status': 'point_estimate' if resolved else 'unresolved_optimization_or_separation', 'intercept': float(a[0]) if resolved else None, 'slope': float(a[1]) if resolved else None, 'maximum_gradient': grad, 'interval': None, 'use': 'reporting only; never applied to saved probabilities'}
    except RuntimeError as error:
        result['calibration_diagnostic'] = {'status': 'unresolved_optimization_failure', 'intercept': None, 'slope': None, 'preserved_failure': str(error)}
    return result


def bootstrap(values, draws, seed):
    """Columns are complete-event means; all arms share each resample."""
    v = torch.tensor(values, device='cuda', dtype=torch.float64)
    generator = torch.Generator(device='cuda').manual_seed(seed)
    chunks = []
    for start in range(0, draws, 1024):
        indices = torch.randint(values.shape[1], (min(1024, draws - start), values.shape[1]), generator=generator, device='cuda')
        chunks.append(v[:, indices].mean(2))
    return torch.cat(chunks, dim=1)


def paired_result(row, delta, samples, draws, seed, primary):
    count = len(delta)
    point = float(delta.mean()) if count else None
    result = row | {'events': count, 'difference': point, 'seed': seed, 'draws': draws if count >= 2 else 0, 'status': 'scored' if count >= 2 else 'insufficient_complete_event_clusters', 'ci95': None}
    if count < 2:
        if primary:
            result.update(p_approx=1., mc_upper_p=1., extreme_count=None)
        return result
    if np.ptp(delta) == 0:
        result.update(status='degenerate_complete_event_differences', ci95=[point, point])
        if primary:
            result.update(p_approx=1., mc_upper_p=1., extreme_count=None)
        return result
    result['ci95'] = torch.quantile(samples, torch.tensor([.025, .975], device='cuda', dtype=torch.float64)).cpu().tolist()
    if primary:
        extreme = int(((samples - point).abs() >= abs(point)).sum())
        result.update(extreme_count=extreme, p_approx=(extreme + 1) / (draws + 1), mc_upper_p=float(beta.ppf(1 - .01 / 12, extreme + 1, draws - extreme)) if extreme < draws else 1., mc_simultaneous_confidence=.99)
    return result


def main():
    ap = argparse.ArgumentParser()
    for name in ('context', 'forecasts', 'targets', 'method', 'output'):
        ap.add_argument('--' + name, type=Path, required=True)
    args = ap.parse_args()
    assert not args.output.exists(), 'Preserve prior scoring attempts'
    assert torch.cuda.is_available(), 'CUDA is required for uncertainty and reporting-only reliability fitting'
    torch.set_num_threads(2)
    method = read(args.method)
    context, forecasts, targets = read(args.context / 'projection.json'), read(args.forecasts / 'frozen.json'), read(args.targets / 'frozen.json')
    assert context['status'] == 'complete_real_source_context_projection' and forecasts['complete_cohort']
    assert targets['status'] == 'targets_projected_after_complete_forecast_freeze'
    assert targets['forecast_freeze_sha256'] == sha(args.forecasts / 'frozen.json')
    assert targets['context_projection_sha256'] == sha(args.context / 'projection.json')
    assert context['method_sha256'] == forecasts['method_sha256'] == targets['method_sha256'] == sha(args.method)
    for root, receipt in ((args.context, context), (args.forecasts, forecasts), (args.targets, targets)):
        for name, h in receipt['files'].items():
            assert sha(root / name) == h
    keys = read(args.forecasts / 'keys.json')
    assert keys == read(args.targets / 'keys.json')
    labels = np.load(args.targets / 'labels.npy')
    assert len(labels) == len(keys)
    assert set(labels[np.isfinite(labels)].tolist()) <= {0., 1.}
    identity = {(r['task'], r['game_id']): r for name in ('map-context.json', 'series-context.json') for r in read(args.context / name)}
    assert set(map(tuple, keys)) == set(identity) and len(keys) == len(identity)
    probabilities, picks = {}, {}
    with np.load(args.forecasts / 'predictions.npz') as z:
        for arm in method['arms']:
            probabilities[arm], picks[arm] = z[arm + '_probability'].copy(), z[arm + '_pick'].copy()
            assert len(probabilities[arm]) == len(picks[arm]) == len(keys)
            assert np.isfinite(probabilities[arm]).all() and ((probabilities[arm] >= 0) & (probabilities[arm] <= 1)).all()
            assert set(picks[arm].tolist()) <= {0, 1}
    arms = method['arms']
    all_metrics, event_metrics, primary, accuracy, denominators = [], [], [], [], []
    for task in method['tasks']:
        task_mask = np.array([t == task for t, g in keys])
        valid = task_mask & np.isfinite(labels)
        indices = np.flatnonzero(valid)
        kk = [identity[tuple(keys[i])] for i in indices]
        yy = labels[valid]
        pp, dd = {a: p[valid] for a, p in probabilities.items()}, {a: p[valid] for a, p in picks.items()}
        for arm in arms:
            all_metrics.append({'task': task, 'arm': arm, **metrics(pp[arm], dd[arm], yy, kk)})
        times = {e: min(k['prediction_cutoff'] for k in kk if k['event_id'] == e) for e in {k['event_id'] for k in kk}}
        events = sorted(times, key=lambda e: (times[e], e))
        for event in method['event_allowlist']:
            frozen = [i for i, (t, g) in enumerate(keys) if t == task and identity[t, g]['event_id'] == event]
            event_ix = [j for j, k in enumerate(kk) if k['event_id'] == event]
            denominators.append({'task': task, 'event_id': event, 'frozen_rows': len(frozen), 'valid_targets': len(event_ix), 'invalid_targets': len(frozen) - len(event_ix)})
            for arm in arms:
                event_metrics.append({'task': task, 'event_id': event, 'arm': arm, **metrics(pp[arm][event_ix], dd[arm][event_ix], yy[event_ix], [kk[i] for i in event_ix], reliability=False)})
        event_losses = {}
        boots = {}
        for metric in ('brier', 'log_loss', 'accuracy'):
            rows = []
            for arm in arms:
                p = np.clip(pp[arm].astype('float64'), 1e-7, 1 - 1e-7)
                values = (p - yy) ** 2 if metric == 'brier' else -yy * np.log(p) - (1 - yy) * np.log1p(-p) if metric == 'log_loss' else (dd[arm] == yy).astype('float64')
                rows.append([float(values[[i for i, k in enumerate(kk) if k['event_id'] == event]].mean()) for event in events])
            event_losses[metric] = np.array(rows, dtype='float64').reshape(len(arms), len(events))
            if len(events) >= 2:
                boots[metric] = bootstrap(event_losses[metric], 131072, method['uncertainty_seed'])
                for row in all_metrics:
                    if row['task'] == task:
                        row['equal_event_' + metric + '_ci95_pointwise'] = torch.quantile(boots[metric][arms.index(row['arm'])], torch.tensor([.025, .975], device='cuda', dtype=torch.float64)).cpu().tolist()
        for spec in method['primary_contrasts']:
            if spec['task'] != task:
                continue
            a, b = arms.index(spec['candidate']), arms.index(spec['parent'])
            delta = event_losses[spec['metric']][a] - event_losses[spec['metric']][b]
            samples = boots[spec['metric']][a] - boots[spec['metric']][b] if len(events) >= 2 else None
            row = {'id': task + '/' + spec['candidate'] + '-' + spec['parent'] + '/' + spec['metric'], **spec, 'direction': 'negative favors candidate', 'conditional_on_saved_fits': True}
            primary.append(paired_result(row, delta, samples, 131072, method['uncertainty_seed'], True))
        for parent in ('original_residual', 'dynamic_strength', 'online_elo_k20'):
            a, b = arms.index('rich_residual'), arms.index(parent)
            delta = event_losses['accuracy'][a] - event_losses['accuracy'][b]
            samples = boots['accuracy'][a] - boots['accuracy'][b] if len(events) >= 2 else None
            accuracy.append(paired_result({'task': task, 'candidate': 'rich_residual', 'parent': parent, 'metric': 'equal_event_accuracy', 'direction': 'positive favors candidate', 'multiplicity': 'pointwise secondary; no adjusted accuracy claim'}, delta, samples, 131072, method['uncertainty_seed'], False))
    assert len(primary) == 12
    for field, output in (('p_approx', 'holm_p'), ('mc_upper_p', 'holm_mc_upper_p')):
        previous = 0.
        for i, row in enumerate(sorted(primary, key=lambda r: (r[field], r['id']))):
            previous = max(previous, min(1., (12 - i) * row[field]))
            row[output] = previous
    for row in primary:
        row['conservative_rejection'] = row['status'] == 'scored' and row['holm_mc_upper_p'] <= .05
    result = {'status': 'later_retained_cohort_scored', 'metrics': all_metrics, 'per_event_metrics': event_metrics, 'primary_contrasts': primary, 'secondary_accuracy_contrasts': accuracy, 'denominators': denominators, 'input_quarantine': read(args.context / 'quarantine.json'), 'target_quarantine': read(args.targets / 'quarantine.json'), 'inference_scope': 'retrospective later competitions; conditional on retained historical fits; no independent external execution or prospective claim', 'new_feedback': False, 'source_policy': 'unreviewed cohort is score-only', 'mc_policy': 'centered complete-event paired bootstrap; 131072 draws; common task/metric resamples; 99 percent simultaneous binomial upper bounds and Holm over twelve contrasts', 'unsupported_inference': 'fewer than two valid complete-event clusters receives unresolved interval and p=1; exactly constant event differences receive descriptive degenerate interval and p=1; neither establishes equivalence', 'numerical_loss_policy': 'saved probabilities clipped to [1e-7,1-1e-7] for finite benchmark-compatible scoring', 'calibration_diagnostics': 'reporting-only CUDA optimization on evaluation labels; never modifies predictions; point estimates without intervals'}
    args.output.mkdir(parents=True)
    (args.output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    receipt = {'status': result['status'], 'method_sha256': sha(args.method), 'context_sha256': sha(args.context / 'projection.json'), 'forecasts_sha256': sha(args.forecasts / 'frozen.json'), 'targets_sha256': sha(args.targets / 'frozen.json'), 'code_sha256': sha(__file__), 'results_sha256': sha(args.output / 'results.json'), 'new_model_fit_selection_or_prediction_changes': False, 'runtime': {'torch': torch.__version__, 'numpy': np.__version__, 'gpu': torch.cuda.get_device_name(0)}}
    (args.output / 'completion.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'primary_contrasts': len(primary), 'metrics': len(all_metrics)}))


if __name__ == '__main__':
    main()
