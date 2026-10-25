"""Descriptive later tie tables and rating figure inputs from frozen outputs only."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--later', type=Path, required=True, help='Lawfully accessible frozen later-cohort artifact root')
parser.add_argument('--aggregates', type=Path, required=True, help='Supplied original aggregate-table directory')
parser.add_argument('--output', type=Path, required=True, help='Destination for aggregate-only descriptive evidence')
arguments = parser.parse_args()
OUT = arguments.output
OUT.mkdir(exist_ok=True)
LATER = arguments.later
AGG = arguments.aggregates


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def write_csv(name, rows):
    with (OUT/name).open('w') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def later_ties():
    sources={}
    for folder, receipt in [('context','projection.json'),('forecasts','frozen.json'),('targets','frozen.json')]:
        path=LATER/folder/receipt
        data=read(path);sources[folder+'/'+receipt]=sha(path)
        for name, expected in data['files'].items():
            path=LATER/folder/name
            if sha(path)!=expected:
                raise RuntimeError('Frozen later input changed')
    targets=read(LATER/'targets/frozen.json')
    if targets['forecast_freeze_sha256'] != sha(LATER/'forecasts/frozen.json'):
        raise RuntimeError('Target-to-forecast binding changed')
    keys=read(LATER/'forecasts/keys.json')
    if keys!=read(LATER/'targets/keys.json'):
        raise RuntimeError('Later key alignment changed')
    identity={(r['task'],r['game_id']):r for n in ['map-context.json','series-context.json'] for r in read(LATER/'context'/n)}
    labels=np.load(LATER/'targets/labels.npy',allow_pickle=False)
    scores=read(LATER/'scoring/results.json')
    sources['scoring/results.json']=sha(LATER/'scoring/results.json')
    rows=[]
    with np.load(LATER/'forecasts/predictions.npz',allow_pickle=False) as z:
        arms=[name[:-12] for name in z.files if name.endswith('_probability')]
        for task in ['map_winner','series_winner']:
            mask=np.array([k[0]==task for k in keys]) & np.isfinite(labels)
            yy=labels[mask]
            events=[identity[tuple(k)]['event_id'] for k,m in zip(keys,mask) if m]
            counts=collections.Counter(events)
            weight=np.array([1/len(counts)/counts[e] for e in events])
            for arm in arms:
                p=z[arm+'_probability'][mask];pick=z[arm+'_pick'][mask]
                correct=(pick==yy).astype(float);tied=p==.5;half=np.where(tied,.5,correct)
                old=next(m for m in scores['metrics'] if m['task']==task and m['arm']==arm)
                if len(yy)!=old['rows'] or len(counts)!=old['events'] or abs(float(correct.mean())-old['row_accuracy'])>1e-12 or abs(float(weight@correct)-old['equal_event_accuracy'])>1e-12:
                    raise RuntimeError('Saved decision/denominator mismatch')
                rows.append(dict(task=task,procedure=arm,rows=len(yy),events=len(counts),
                                 exact_probability_ties=int(tied.sum()),tied_team_a_wins=int(yy[tied].sum()),
                                 tied_team_a_fraction=float(yy[tied].mean()) if tied.any() else None,
                                 tied_event_weight=float(weight[tied].sum()),
                                 ordinary_row_accuracy=float(correct.mean()),ordinary_event_accuracy=float(weight@correct),
                                 half_credit_row_accuracy=float(half.mean()),half_credit_event_accuracy=float(weight@half)))
    write_csv('later-exact-ties.csv',rows)
    text=['## S23. Later-cohort exact-tie diagnostic','',
          'This descriptive audit reads the original frozen later-cohort forecasts and targets; it performs no inference, fitting, calibration or selection. The original finite-target masks retain 1,613 maps and 734 series across 23 events. Ordinary accuracy exactly reproduces the saved decisions, including residual raw-probability thresholds. Half-credit replaces only decisions whose saved calibrated probability equals exactly 0.5 by credit 0.5; all other decisions remain unchanged. This is a diagnostic, not a replacement decision policy. No probability or probability-loss result changes. Zero ties have undefined tied-subset class balance. Event accuracy weights each of the original 23 events equally; row accuracy weights targets equally. The audit does not infer why a particular rating or mixture produces an exact tie.','',
          '| Task / procedure | Rows | Exact ties | Team A wins among ties | Tied Team A fraction | Ordinary row / event accuracy (%) | Half-credit row / event accuracy (%) |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        fraction='undefined' if r['tied_team_a_fraction'] is None else f"{r['tied_team_a_fraction']:.6f}"
        text.append(f"| {r['task'].replace('_winner','')} / {r['procedure'].replace('_',' ')} | {r['rows']} | {r['exact_probability_ties']} | {r['tied_team_a_wins']} | {fraction} | {100*r['ordinary_row_accuracy']:.3f} / {100*r['ordinary_event_accuracy']:.3f} | {100*r['half_credit_row_accuracy']:.3f} / {100*r['half_credit_event_accuracy']:.3f} |")
    (OUT/'later-ties.md').write_text('\n'.join(text)+'\n')
    return sources


def rating_inputs():
    rating_path=AGG/'rating-comparison_results_metrics.csv'
    with rating_path.open() as f:ratings=list(csv.DictReader(f))
    rows=[];sources={'rating_aggregate':sha(rating_path)}
    for task in ['map_winner','series_winner']:
        path=AGG/('diagnostics_'+task+'-diagnostics_metrics.csv')
        with path.open() as f:diagnostics=list(csv.DictReader(f))
        sources[task+'_diagnostic_aggregate']=sha(path)
        for model, label in [('earlier','Earlier residual'),('rich','Online-rich residual'),('dynamic','Dynamic strength')]:
            m=next(x for x in diagnostics if x['scope']=='broad_nonglobal' and x['package']==model)
            rows.append(dict(task=task,procedure=label,probability_policy='separate residual calibration' if model!='dynamic' else 'raw',
                             rows=int(m['rows']),events=int(m['events']),event_brier=float(m['event_brier']),event_log_loss=float(m['event_log_loss'])))
        for model, regime, label in [('elo_k20','online_raw','K20 raw'),('elo_k20','online_calibrated','K20 calibrated'),('glicko2','online_calibrated','Glicko-2 calibrated'),('player_gaussian','online_calibrated','Player Gaussian calibrated')]:
            m=next(x for x in ratings if x['task']==task and x['model']==model and x['regime']==regime)
            rows.append(dict(task=task,procedure=label,probability_policy='raw' if regime=='online_raw' else 'common separate-event calibration',
                             rows=int(m['rows']),events=int(m['events']),event_brier=float(m['event_brier']),event_log_loss=float(m['event_log_loss'])))
    write_csv('online-rating-display.csv',rows)
    return sources


if __name__=='__main__':
    sources=dict(later=later_ties(),rating=rating_inputs())
    (OUT/'saved-output-provenance.json').write_text(json.dumps(dict(code_sha256=sha(Path(__file__)),inputs=sources,
        scope='descriptive saved-output summaries only; no new experiments or inferential comparisons',
        exports='aggregate counts and metrics only; no source identifiers or row forecasts'),indent=2)+'\n')
