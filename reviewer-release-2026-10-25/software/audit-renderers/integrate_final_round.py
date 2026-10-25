"""Render aggregate-only final sensitivity evidence; never fit or rescore forecasts."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/paper-study/professor-revision-2026-10-24'
RUN = ROOT / 'output/local-final-round-2026-10-25'
CONTROL = RUN / 'control-root/valorant-final-round/results'
OUT = DOC / 'final-sensitivities'
OUT.mkdir(exist_ok=True)


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


reports = {}
sources = {}
for family in ('refresh8', 'fresh20', 'change20', 'clocks40'):
    base = CONTROL if family == 'refresh8' else RUN / 'work/results'
    path = base / 'reports' / family / 'results.json'
    completion = load(path.parent / 'completion.json')
    if digest(path) != completion['results']:
        raise RuntimeError('Report binding mismatch: ' + family)
    reports[family] = load(path)
    sources[family] = digest(path)
audit_path = CONTROL / 'audits/results.json'
audits = load(audit_path)
if digest(audit_path) != load(audit_path.parent / 'completion.json')['results']:
    raise RuntimeError('Audit binding mismatch')
sources['audits'] = digest(audit_path)

contrasts = []
metrics = []
for family, report in reports.items():
    for row in report['contrasts']:
        out = {k: v for k, v in row.items() if k != 'ci95'}
        out.update(ci_low=row['ci95'][0], ci_high=row['ci95'][1])
        contrasts.append(out)
    metrics.extend(dict(report_family=family, **row) for row in report['metrics'])
csv_write('all-88-contrasts.csv', contrasts)
csv_write('procedure-metrics.csv', metrics)
csv_write('core-ties-by-block.csv', audits['ties'])
csv_write('removal-parent-metrics.csv', audits['removal_parent_absolute_losses'])
csv_write('schedule-gaps.csv', audits['turnarounds'])
calibrators = []
for row in audits['calibration_choices']:
    cal = row['calibrator']
    for candidate in cal.get('candidates', []):
        calibrators.append(dict(block=row['block'], task=row['task'], procedure=row['procedure'],
                                selected_scale=cal.get('selected', {}).get('scale'),
                                candidate_scale=candidate['scale'],
                                candidate_log_loss=candidate['log_loss']))
csv_write('calibration-candidates.csv', calibrators)

names = {'linear': 'Logistic', 'catboost': 'CatBoost', 'xgboost': 'XGBoost', 'mlp': 'MLP', 'tabm': 'TabM'}
policy_names = {'fresh2h15m': 'Advancing history (2 h / 15 min)',
                'frozen2h15m': 'Frozen history (2 h / 15 min)',
                'frozen6h30m': 'Frozen history (6 h / 30 min)',
                'frozen24h60m': 'Frozen history (24 h / 60 min)',
                'refresh': 'Fixed-weight input refresh'}


def predictor(name):
    if name in ('selector', 'elo_k20', 'refresh', 'earlier', 'rich_refit'):
        return {'selector': 'retained mixture', 'elo_k20': 'K20 Elo', 'refresh': 'refresh only',
                'earlier': 'earlier residual', 'rich_refit': 'online-rich refit'}[name]
    rung, family = name.split('-', 1)
    return rung + ' ' + names[family]


lines = ['## S22. Completed exploratory history and availability sensitivities', '',
         'These analyses supplement the original results without replacing their inference families. All use the broad nonglobal development population: 55,943 scored maps in 1,049 events and 25,186 scored series in 1,047 events. Evaluation keys and finite-target scoring masks are matched between compared procedures. They are retrospective exploratory analyses, not independent confirmation.', '',
         '### Design and uncertainty', '',
         'Advancing-history B/BN comparisons retain the original twelve phase allocations, selected model recipes and seeds, 15-minute schedule lead and two-hour release proxy. Historical summaries advance only from earlier eligible nonglobal observations. Whole Match IDs share a cutoff; equal-time batches use frozen pre-batch expectations and summed updates. Current/same-series outcomes and known global outcomes cannot advance history. Preprocessing and parameter fitting remain training-local; stopping, calibration and the retained mixtures use their assigned earlier phases. Six earlier outcomes were unavailable after their release guards passed; those six observations were logged and excluded from feedback updates, without imputing outcomes or removing prediction rows. The original finite-label scoring rule is unchanged.', '',
         'The 6-hour/30-minute and 24-hour/60-minute arms rebuild training-only histories, purge earlier decision events whose completion-plus-delay proxy reaches the next originally allocated nonempty phase, retain the original selected hyperparameters and mixture configurations, and preserve evaluation membership. Delay and lead change together, so their separate effects are not identified. Archived map lineups/picks remain conditional after-picks inputs; these analyses do not verify pre-series publication at either lead time.', '',
         'The fixed-weight control replaces only the earlier residual\'s specified ACS/star, economy, pistol and recent-90-ACS summaries and associated coverage with the corresponding online-rich inputs. Earlier estimators, layout, calibrators, thresholds, dynamic state and composition/proxy rules are retained; no model is fitted for this control. Comparing the original refitted online-rich procedure with this control still includes reselection and recalibration differences.', '',
         'Each contrast uses 131,072 paired complete-event resamples, percentile 95% intervals, the original centered unstudentized two-sided tail, and a Clopper–Pearson Monte Carlo upper bound with simultaneous confidence 99% within its declared family. Holm is applied separately to 8 fixed-weight contrasts, 20 advancing-history contrasts, 20 changes in the BN–B increment, and 40 stricter-clock contrasts. Decisions below use the adjusted Monte Carlo upper bound at 0.05. Intervals are pointwise and conditional on saved fits; they omit full-pipeline retraining/selection uncertainty and repeated-team dependence. Non-rejection does not establish equivalence.', '',
         '### Execution provenance and support', '',
         'Original two-hour frozen controls and the fixed-weight/audit results were retained from Tesla T4 execution. Advancing-history blocks 1–9 and part of block 10 were completed on T4; remaining units were completed on RTX 4080 SUPER. Both stricter-clock arms were fitted locally. Python was 3.13.16 on T4 and 3.12.14 locally; PyTorch was 2.11.0+cu130 and 2.14.0+cu130, NumPy 2.1.3 and 2.5.3, SciPy 1.16.3 and 1.18.1, and scikit-learn 1.6.1 and 1.9.1, respectively. Both used CUDA 13.0, XGBoost 3.4.1, CatBoost 1.2.10, Lightning 2.6.6 and explicit neural 16-mixed precision. Completed fits were hash-bound and reused under their original identities. These are mixed-environment sensitivities, not hardware-matched replication; especially the fresh-minus-frozen change cannot be attributed solely to history policy.', '',
         'The advancing and two-hour frozen arms each have 64 insufficient-history package selections out of 240 (32/120 per task). Each stricter-clock arm has 48/184 (24/92 per task). These denominators count package selections, not trained estimators; the clock arms fit a narrower declared subset. No candidate-seed failure marker was found. Completed inputs, inference-only packages and explicit rating fallbacks are not counted as fitted models. Failed and interrupted attempts were retained, including a missing-outcome preparation failure and a later interrupted input preparation. This preservation does not establish the cause of the computer interruption.', '',
         'The retained fixed-weight report results and completion hashes match their original binding. Its completion-listed scoring-activation artifact is absent from the retained evidence, so that historical activation receipt remains unverified; no replacement receipt is invented. New 20/20/40 families retain their activation records and complete forecast freezes.', '']

descriptions = {
    'refresh8': ('Fixed-weight input refresh: all 8 contrasts', 'Difference is first procedure minus second.'),
    'fresh20': ('Advancing-history BN versus B: all 20 contrasts', 'Difference is BN minus B within the advancing-history arm.'),
    'change20': ('Change in the BN–B increment: all 20 contrasts', 'Difference is (advancing BN − advancing B) − (frozen BN − frozen B). Negative values favor a more beneficial advancing-history increment, not necessarily a lower absolute loss for every model.'),
    'clocks40': ('Stricter-clock comparisons: all 40 contrasts', 'Difference is first procedure minus second within each clock arm.')}
for family, report in reports.items():
    title, desc = descriptions[family]
    lines += ['### ' + title, '', desc + ' Negative loss differences favor the first procedure. Brier and log loss are event-equal. “Resolved harm” is retained rather than counted as improvement.', '',
              '| Clock / task / contrast | Loss | Difference | Pointwise 95% interval | Holm MC upper p | Decision |',
              '|---|---|---:|---|---:|---|']
    for c in report['contrasts']:
        decision = ('Resolved gain' if c['difference'] < 0 else 'Resolved harm') if c['conservative_rejection'] else 'Unresolved'
        label = policy_names[c['policy']] + ' / ' + c['task'].replace('_winner', '') + ' / ' + predictor(c['a']) + ' − ' + predictor(c['b'])
        lines.append(f"| {label} | {c['metric'].replace('_', ' ')} | {c['difference']:+.6f} | [{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}] | {c['holm_mc_upper_p']:.6f} | {decision} |")
    lines += ['']

lines += ['### Complete procedure metrics', '',
          'Event-equal and row-weighted scores are separately labelled in `procedure-metrics.csv`. The accuracy fields in that aggregate report use the diagnostic calibrated-probability rule p > 0.5, with exact ties assigned to Team B; its separate half-credit diagnostic gives ties 0.5. For residuals these diagnostic accuracies are not the earlier-selected raw-probability threshold decisions retained in the original online table. No new accuracy superiority claim is based on these diagnostics.', '',
          '| Arm / task / procedure | Event Brier | Event log loss | Row Brier | Row log loss | Diagnostic row accuracy (%) |',
          '|---|---:|---:|---:|---:|---:|']
seen = set()
for m in metrics:
    key = m['policy'], m['task'], m['predictor']
    if key in seen:
        continue
    seen.add(key)
    label = policy_names[m['policy']] + ' / ' + m['task'].replace('_winner', '') + ' / ' + predictor(m['predictor'])
    lines.append(f"| {label} | {m['event_brier']:.6f} | {m['event_log_loss']:.6f} | {m['row_brier']:.6f} | {m['row_log_loss']:.6f} | {100*m['row_accuracy']:.3f} |")

lines += ['', '### Saved-output availability and decision audits', '',
          'There are 5,620 two-to-under-six-hour gaps among 49,158 consecutive distinct scheduled-start intervals for teams (11.43%), and 28,053 among 256,185 for players (10.95%). These are schedule gaps, not observed result release or contest duration. Equal scheduled times establish no order. The admitted nonglobal map inventory has 5,038/13,973/7,577/451/348 Match IDs with 1/2/3/4/5 observed maps. This is a realized source-inventory distribution, not verified best-of format, and is not used as a predictor.', '',
          'On the eight-block removal support, the absolute parent event-equal Brier/log loss is 0.231405/0.655828 for 36,624 maps in 720 events and 0.247576/0.689135 for 16,319 series in 720 events. Row-weighted scores remain separately available in `removal-parent-metrics.csv`.', '',
          'The core audit retains 358 procedure/block/task tie records and 298 calibration-choice records, including actual tied-subset Team A counts. The table below pools row counts only; complete block records preserve event-equal ordinary/half-credit accuracies and origins in `core-ties-by-block.csv`. Calibration candidate scales and losses are in `calibration-candidates.csv`. Zero-scale selection identifies a minimum separate-event calibration log loss among the recorded candidates, not a causal explanation for poor calibration. Explicit chance, sparse fallback and selected-rating/mixture states are distinguished where recorded; equal-rating versus absent-history causes and component origins of blended exact ties remain unresolved.', '',
          '| Task / procedure | Rows | Exact ties | Team A wins among ties | Tied Team A fraction | Ordinary row accuracy (%) | Half-credit row accuracy (%) |',
          '|---|---:|---:|---:|---:|---:|---:|']
groups = defaultdict(list)
for row in audits['ties']:
    groups[row['task'], row['procedure']].append(row)
for (task, procedure), rows in sorted(groups.items()):
    count = sum(x['rows'] for x in rows)
    ties = sum(x['ties'] for x in rows)
    wins = sum(x['tied_team_a_wins'] for x in rows)
    ordinary = sum(x['row_accuracy'] * x['rows'] for x in rows) / count
    half = sum(x['row_half_credit_accuracy'] * x['rows'] for x in rows) / count
    fraction = f'{wins/ties:.6f}' if ties else 'undefined'
    lines.append(f'| {task.replace("_winner", "")} / {procedure} | {count:,} | {ties:,} | {wins:,} | {fraction} | {ordinary*100:.3f} | {half*100:.3f} |')
lines += ['', 'The companion tables contain no source row keys or raw records. The new analyses do not repeat the later cohort or full online study under each clock, recover real publication timestamps, identify optimal hyperparameters, or demonstrate prospective performance.', '']
(OUT / 'supplement-section.md').write_text('\n'.join(lines))
supplement = DOC / 'supplement.md'
original = supplement.read_text().split('\n## S22. Completed exploratory history and availability sensitivities')[0]
supplement.write_text(original.rstrip() + '\n\n' + '\n'.join(lines))

spec = dict(inputs=sources, scope='broad nonglobal; exploratory; conditional saved fits',
            figure='Event-equal Brier differences with pointwise 95% intervals; negative favors BN',
            mappings=dict(x='BN minus B Brier', y='model family', color='history/clock arm',
                          panels='map and series', interval='pointwise event bootstrap 95%'),
            filtering='Brier BN-versus-B only; all five fresh families and three declared clock families',
            code_sha256=digest(Path(__file__)))
(OUT / 'specification.json').write_text(json.dumps(spec, indent=2) + '\n')



if __name__ == '__main__':
    from plot_final_sensitivities import render
    render(OUT / 'all-88-contrasts.csv', OUT)
