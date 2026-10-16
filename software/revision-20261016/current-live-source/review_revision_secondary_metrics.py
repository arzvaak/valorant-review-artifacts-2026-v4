"""Descriptive secondary inventory of completed, frozen probability vectors."""
import argparse
from pathlib import Path
import numpy as np
import torch
import later_score
from review_revision_diagnostics import CORE, STORE, TASKS, Inputs, read, save, sha


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--docs',type=Path,required=True)
    args=ap.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for reporting diagnostic')
    torch.set_num_threads(2)
    if args.output.exists():raise RuntimeError('Preserve prior output; choose a new directory')
    args.output.mkdir(parents=True)
    inputs=Inputs();inputs.bind(__file__);inputs.bind(later_score.__file__)
    inputs.bind(args.docs/'secondary-metrics-method.md')
    folds=inputs.json(CORE/'config.json')['folds']
    player=STORE/'manuscript-review-player-fits-20261016-v1'
    player_freeze=inputs.json(player/'all-predictions-frozen.json')
    for file,h in player_freeze['packages'].items():
        receipt=inputs.json(inputs.bind(file,h))
        inputs.bind(Path(file).parent/'predictions.npz',receipt['predictions'])
    removal=STORE/'manuscript-review-retuned-removals-20261016-v2'
    removal_freeze=inputs.json(removal/'all-predictions-frozen.json')
    complete=inputs.json(removal/'completion.json')
    inputs.bind(removal/'all-predictions-frozen.json',complete['forecasts'])
    records=[]
    def metric(task,group,name,keys,y,p):
        p=np.asarray(p)
        if len(p)!=len(y) or not np.isfinite(p).all():raise RuntimeError('Invalid aligned predictions')
        records.append(dict(task=task,group=group,condition=name,**later_score.metrics(p,p>.5,np.asarray(y),keys)))
        save(args.output/'progress.json',dict(completed=len(records),last=group+'/'+task+'/'+name))
    for task in TASKS:
        all_keys=[];all_y=[];player_predictions={}
        reduced_keys=[];reduced_y=[];reduced_predictions={}
        for fold in folds:
            block=fold['name']
            label_path=CORE/'scored-labels'/block/(task+'.npy')
            label_receipt=inputs.json(label_path.with_suffix('.json'))
            kk=inputs.json(inputs.bind(CORE/'blocks'/block/'inputs'/task/'evaluation-keys.json',label_receipt['keys_sha256']))
            yy=np.load(inputs.bind(label_path,label_receipt['sha256']))
            valid=np.isfinite(yy)&np.array([k['region']!='global' for k in kk])
            all_keys.extend(kk);all_y.extend(yy)
            for rep in ('BN','BNC','BNL','BNLC'):
                for pool in ('sum','mean','attention','summary'):
                    name='corrected-'+rep+'-'+pool
                    player_predictions.setdefault(name,[]).extend(np.load(player/block/task/name/'predictions.npz')['evaluation_calibrated'][valid])
            name='original-BNLC-sum'
            player_predictions.setdefault(name,[]).extend(np.load(player/block/task/name/'predictions.npz')['evaluation_calibrated'][valid])
            packages=[r for r in removal_freeze['packages'] if r['task']==task and r['block']==block]
            if not packages:continue
            if len(packages)!=10:raise RuntimeError('Incomplete reduced block')
            reduced_keys.extend(k for k,b in zip(kk,valid) if b);reduced_y.extend(yy[valid])
            parent=packages[0]['parent'];directory=Path(parent['directory'])
            parent_receipt=inputs.json(inputs.bind(directory/'frozen.json',parent['freeze_sha256']))
            inputs.bind(directory/'predictions.npz',parent_receipt['files']['predictions.npz'])
            reduced_predictions.setdefault('full_reference',[]).extend(np.load(directory/'predictions.npz')['evaluation_calibrated'][valid])
            for package in packages:
                record=inputs.json(inputs.bind(package['path'],package['sha256']));directory=Path(package['path']).parent
                inputs.bind(directory/'predictions.npz',record['files']['predictions.npz'])
                reduced_predictions.setdefault(package['condition'],[]).extend(np.load(directory/'predictions.npz')['evaluation_calibrated'][valid])
        all_y=np.asarray(all_y);valid=np.isfinite(all_y)&np.array([k['region']!='global' for k in all_keys])
        keys=[k for k,b in zip(all_keys,valid) if b];y=all_y[valid]
        for delay in (6,2):
            root=STORE/f'manuscript-review-rating-comparison-20261016-frozen{delay}-v1'
            completion=inputs.json(root/'completion.json')
            report=inputs.json(inputs.bind(root/'results.json',completion['results']))
            z=np.load(inputs.bind(root/(task+'-probabilities.npz')))
            # The report did not freeze its derived NPZ. Verify every value against
            # the original frozen forecasts and the result-bound calibration scale.
            rebuilt={name:[] for name in z.files}
            online_root=STORE/'manuscript-review-ratings-20261016-online-v1'
            frozen_root=STORE/f'manuscript-review-ratings-20261016-frozen{delay}-v2'
            inputs.bind(online_root/'completion.json',report['online'])
            inputs.bind(frozen_root/'completion.json',report['frozen'])
            for fold in folds:
                block=fold['name'];kk=read(CORE/'blocks'/block/'inputs'/task/'evaluation-keys.json')
                for regime,directory in [('online',online_root/'online'),('frozen',frozen_root/block)]:
                    receipt=inputs.json(directory/'frozen.json')
                    index={tuple(k):i for i,k in enumerate(inputs.json(inputs.bind(directory/'keys.json',receipt['keys'])))}
                    raw=np.load(inputs.bind(directory/'predictions.npz',receipt['predictions']))['probability']
                    raw=raw[[index[task,k['game_id']] for k in kk]]
                    rebuilt[regime+'_raw'].append(raw)
                    cols=[]
                    for column,name in enumerate(('elo_k20','glicko2','player_gaussian')):
                        chosen=[r for r in report['calibrations'] if r['block']==block and r['task']==task and r['regime']==regime and r['model']==name]
                        if len(chosen)!=1:raise RuntimeError('Ambiguous saved rating scale')
                        tensor=torch.tensor(raw[:,column],device='cuda',dtype=torch.float64).clamp(1e-7,1-1e-7)
                        cols.append((torch.logit(tensor)*chosen[0]['scale']).sigmoid().cpu().numpy())
                    rebuilt[regime+'_calibrated'].append(np.stack(cols,1))
            for name,parts in rebuilt.items():
                if not np.array_equal(z[name],np.concatenate(parts)):raise RuntimeError('Rating archive differs from original frozen forecasts/scales')
            for regime in (('online_raw','online_calibrated','frozen_raw','frozen_calibrated') if delay==6 else ('frozen_raw','frozen_calibrated')):
                for column,name in enumerate(('elo_k20','glicko2','player_gaussian')):
                    metric(task,'ratings',str(delay)+'h/'+regime+'/'+name,keys,y,z[regime][valid,column])
        for name,p in player_predictions.items():metric(task,'players',name,keys,y,p)
        for name,p in reduced_predictions.items():metric(task,'removals',name,reduced_keys,reduced_y,p)
    feedback_root=STORE/'manuscript-review-feedback-20261016-v3/scoring'
    feedback_completion=inputs.json(feedback_root/'completion.json')
    feedback=inputs.json(inputs.bind(feedback_root/'results.json',feedback_completion['results_sha256']))
    for row in feedback['metrics']:
        records.append(dict(group='later_feedback',condition=row['arm'],**row))
    save(args.output/'results.json',dict(inputs=inputs.hashes,metrics=records))
    lines=['# Complete additional secondary metric inventory','','These are descriptive point estimates conditional on saved fits. AUC and ECE use equal-event weights; balanced accuracy gives the two target classes equal row weight. Canonical-orientation intercept/slope diagnose the original probabilities and never modify them. No new hypothesis family or selection is introduced. Coefficient intervals for selected main procedures appear elsewhere; none is implied for these additional points. All threshold decisions here use probability greater than 0.5; feedback rows preserve their original saved decisions.','','| Task / group / procedure | Rows/events | Balanced accuracy | AUC | Intercept | Slope | ECE 5/10/20 |','|---|---:|---:|---:|---:|---:|---|']
    for r in records:
        d=r['calibration_diagnostic'];fmt=lambda x:'unresolved' if x is None else f'{x:.5f}'
        lines.append('| '+' | '.join([r['task']+'/'+r['group']+'/'+r['condition'],str(r['rows'])+'/'+str(r['events']),fmt(r['balanced_accuracy']),fmt(r['equal_event_weighted_roc_auc']),fmt(d['intercept']),fmt(d['slope']),'/'.join(fmt(r['ece_'+str(b)]) for b in (5,10,20))])+' |')
    (args.docs/'secondary-metrics-findings.md').write_text('\n'.join(lines)+'\n')
    save(args.output/'completion.json',dict(results=sha(args.output/'results.json'),records=len(records)))
    print('SECONDARY METRICS COMPLETE',len(records),flush=True)


if __name__=='__main__':main()
