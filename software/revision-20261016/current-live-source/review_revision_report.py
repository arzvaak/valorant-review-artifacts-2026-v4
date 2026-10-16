"""Assemble revision diagnostics and analyze saved later forecasts on CUDA."""
import argparse
import collections
import datetime as dt
from pathlib import Path
import shutil

import numpy as np
import torch

from review_revision_diagnostics import (
    CORE, LATER, TASKS, diagnostic_bootstrap, grouping, metrics, paired, read, save, sha,
)


def later(output):
    target = output/'later-diagnostics.json'
    inputs = {}
    for directory, marker in [('context', 'projection.json'), ('forecasts', 'frozen.json'), ('targets', 'frozen.json')]:
        receipt = read(LATER/directory/marker)
        inputs[str(LATER/directory/marker)] = sha(LATER/directory/marker)
        for name, expected in receipt['files'].items():
            path = LATER/directory/name
            if sha(path) != expected:
                raise RuntimeError('Later artifact changed: '+str(path))
            inputs[str(path)] = expected
    identity = dict(code=sha(__file__), diagnostics_code=sha(Path(__file__).with_name('review_revision_diagnostics.py')), inputs=inputs)
    if target.exists():
        if read(target)['identity'] != identity:
            raise RuntimeError('Later result identity changed; version output')
        return read(target)
    order = read(LATER/'forecasts/keys.json')
    if order != read(LATER/'targets/keys.json'):
        raise RuntimeError('Later forecast/target membership mismatch')
    keys = {(k['task'], k['game_id']): k for name in ('map-context.json', 'series-context.json') for k in read(LATER/'context'/name)}
    labels = np.load(LATER/'targets/labels.npy')
    forecasts = np.load(LATER/'forecasts/predictions.npz')
    result = dict(identity=identity, tasks=[])
    for task in TASKS:
        indices = [i for i, key in enumerate(order) if key[0] == task and np.isfinite(labels[i])]
        kk = [keys[tuple(order[i])] for i in indices]
        y = labels[indices].astype(float)
        p = {arm: forecasts[arm+'_probability'][indices].astype(float) for arm in ('rich_residual', 'original_residual', 'dynamic_strength', 'online_elo_k20')}
        task_result = dict(task=task, metrics={}, contrasts=[], reliability=[])
        for arm, prob in p.items():
            task_result['metrics'][arm] = metrics(kk, y, prob, forecasts[arm+'_pick'][indices])
        for reference in ('original_residual', 'dynamic_strength', 'online_elo_k20'):
            task_result['contrasts'].append(dict(a='rich_residual', b=reference, **paired(kk, y, p['rich_residual'], p[reference], 131072, 20261016)))
        for mirror in (False, True):
            task_result['reliability'].append(diagnostic_bootstrap(kk, y, p['rich_residual'], 2048, 20261017, mirror))
        result['tasks'].append(task_result)
    save(target, result)
    return result


def date(timestamp):
    return dt.datetime.fromtimestamp(timestamp, dt.timezone.utc).strftime('%Y-%m-%d') if timestamp is not None else 'absent'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diagnostics', type=Path, required=True)
    parser.add_argument('--docs', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA mandatory for uncertainty and diagnostic fits')
    torch.set_num_threads(2)
    result = later(args.diagnostics)
    lines = ['# Saved-forecast revision findings', '', 'Original forecasts remain unchanged. New intervals condition on saved fitted procedures and use complete-event resampling; they do not include refitting/selection uncertainty. Loss comparisons below are pointwise exploratory intervals.', '',
             '## Orientation and exact ties', '', '| Task / procedure | Rows | Exact ties | Ordinary accuracy | Tie-averaged accuracy | Event accuracy | Tie-averaged event accuracy |', '|---|---:|---:|---:|---:|---:|---:|']
    timing = ['# Executed historical timing and removal support', '', 'Core numeric, composition and player histories admit only training maps, released at scheduled anchor + two hours, strictly before the prediction cutoff. Stopping, selection and calibration outcomes never update those histories. Those phases can stop/select/calibrate models using their own labels. Historical recent-90-day windows age relative to the target cutoff, so they become empty when every admitted training observation is outside the window.', '', '| Block / task | Train last | Stopping first–last | Selection first–last | Calibration first–last | Evaluation first–last | Observed state age min–max (days) | Removal parent |', '|---|---|---|---|---|---|---:|---|']
    contrast_lines = ['## Matched probability-loss comparisons', '', 'Positive differences favor the reference. Online Elo in the original extension is uncalibrated with six-hour feedback; core Elo is separately scalar-calibrated with training-only two-hour history. Their direct difference is a procedure difference, not an isolated refresh effect.', '', '| Task / scope / candidate − reference | Event Brier difference [95% interval] | Event log-loss difference [95% interval] |', '|---|---:|---:|']
    reliabilities = ['## Calibration diagnostic uncertainty', '', 'Canonical diagnostics answer calibration on the actually ordered cohort. Mirrored diagnostics retain each pair in the same event resample; symmetry sets their intercept to zero without establishing canonical calibration. Coefficients are reporting only and never alter predictions.', '', '| Task / procedure / orientation | Intercept [95% interval] | Slope [95% interval] | Failed draws / 2048 |', '|---|---:|---:|---:|']
    for task in TASKS:
        data = read(args.diagnostics/(task+'-diagnostics.json'))
        for m in data['metrics']:
            if m['scope'] == 'broad_nonglobal' and m['package'] in ('selector', 'block_elo', 'prior', 'online_elo', 'rich'):
                lines.append(f"| {task} / {m['package']} | {m['rows']} | {m['exact_ties']} | {100*m['row_accuracy']:.3f}% | {100*m['row_tie_averaged_accuracy']:.3f}% | {100*m['event_accuracy']:.3f}% | {100*m['event_tie_averaged_accuracy']:.3f}% |")
        support = {r['block']: r['parent'] for r in data['metadata']['removal_support']}
        for row in data['metadata']['timing']:
            p = row['phases']
            spans = [date(p[phase]['first_anchor'])+'–'+date(p[phase]['last_anchor']) for phase in ('stopping', 'selection', 'calibration', 'evaluation')]
            ages = row['state_age_days_quantiles']
            timing.append(f"| {row['block']} / {task} | {date(p['train']['last_anchor'])} | {' | '.join(spans)} | {ages[0]:.2f}–{ages[-1]:.2f} | {support[row['block']]['family'] if support[row['block']] else 'unavailable'} |")
        for c in data['contrasts']:
            if c['scope'] in ('broad_nonglobal', 'reviewed2025'):
                cells = [f"{c[metric]['event_difference']:+.6f} [{c[metric]['event_ci95'][0]:+.6f}, {c[metric]['event_ci95'][1]:+.6f}]" for metric in ('brier', 'log_loss')]
                contrast_lines.append(f"| {task} / {c['scope']} / {c['a']} − {c['b']} | {' | '.join(cells)} |")
        for c in data['reliability']:
            ci = c.get('coefficient_ci95')
            cells = [f"{c['point'][i]:.4f} [{ci[i][0]:.4f}, {ci[i][1]:.4f}]" if ci else 'unresolved' for i in (0, 1)]
            reliabilities.append(f"| {task} / {c['package']} / {'mirrored' if c.get('mirror') else 'canonical'} | {' | '.join(cells)} | {c.get('failed_draws', 'unresolved')} |")
    lines += ['', *contrast_lines, '', *reliabilities, '', '## Later retained continuation', '', '| Task | Events | Event-size minimum / median / maximum | Rich Brier (event) | Rich Brier (row) |', '|---|---:|---:|---:|---:|']
    for task_result in result['tasks']:
        m = task_result['metrics']['rich_residual']
        q = m['event_size_quantiles']
        lines.append(f"| {task_result['task']} | {m['events']} | {q[0]:.0f} / {q[2]:.0f} / {q[-1]:.0f} | {m['event_brier']:.6f} | {m['row_brier']:.6f} |")
    original = read(LATER/'scoring/results.json')
    special = next(r for r in original['primary_contrasts'] if r['id']=='series_winner/rich_residual-original_residual/brier')
    lines += ['', f"The original later series rich–earlier Brier contrast used {special['draws']:,} draws and {special['events']} events: approximate centered-tail p={special['p_approx']:.6f}, Monte Carlo upper bound={special['mc_upper_p']:.6f}, conservative Holm value={special['holm_mc_upper_p']:.6f}. The percentile interval and centered-tail test are not inversions of the same construction, and the displayed adjusted decision also includes multiplicity. No specific skewness explanation is established by this arithmetic alone.", '', '## Identity and preservation', '', f'Artifact directory: `{args.diagnostics}`. Machine-readable summaries retain per-input identities, actual CUDA memory and elapsed time. All 131,072-draw loss intervals and 2,048-draw calibration intervals preserve event clustering. No predictive fits, threshold changes or source-target regeneration occurred.']
    (args.docs/'saved-forecast-findings.md').write_text('\n'.join(lines)+'\n')
    (args.docs/'history-timing-and-removal-support.md').write_text('\n'.join(timing)+'\n')
    for file in ('review_revision_report.py', 'review_revision_diagnostics.py'):
        shutil.copy2(Path(__file__).with_name(file), args.diagnostics/file)
    save(args.diagnostics/'report-completion.json', dict(code=sha(__file__), later=sha(args.diagnostics/'later-diagnostics.json'), report=sha(args.docs/'saved-forecast-findings.md'), timing=sha(args.docs/'history-timing-and-removal-support.md')))
    print('Saved revision findings and timing table', flush=True)


if __name__ == '__main__':
    main()
