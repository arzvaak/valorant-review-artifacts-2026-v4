"""Common separate-event scalar calibration and paired rating evaluation."""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from review_revision_diagnostics import CORE, PHASES, TASKS, grouping, metrics, paired, read, save, sha


def calibrate(probability, labels, keys):
    if len({k['event_id'] for k in keys})<5 or len(np.unique(labels))<2:
        return 1., dict(method='identity',reason='insufficient_calibration_events_or_classes')
    _, ix, counts=grouping(keys)
    w=torch.tensor(1/len(counts)/counts[ix],device='cuda',dtype=torch.float64)
    y=torch.tensor(labels,device='cuda',dtype=torch.float64)
    z=torch.logit(torch.tensor(probability,device='cuda',dtype=torch.float64).clamp(1e-7,1-1e-7))
    theta=torch.tensor(-2.1972245773362196,device='cuda',dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([theta],max_iter=1000,tolerance_grad=1e-9,tolerance_change=1e-12,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss=(F.binary_cross_entropy_with_logits(10*theta.sigmoid()*z,y,reduction='none')*w).sum()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite rating calibration')
        loss.backward()
        return loss
    optimizer.step(closure)
    candidates=[]
    with torch.no_grad():
        for scale in (0.,1.,float(10*theta.sigmoid())):
            candidates.append(dict(scale=scale,log_loss=float((F.binary_cross_entropy_with_logits(scale*z,y,reduction='none')*w).sum())))
    chosen=min(candidates,key=lambda r:(r['log_loss'],abs(r['scale']-1)))
    return chosen['scale'],dict(method='separate-event nonnegative scalar logit calibration',candidates=candidates,selected=chosen)


def scale_probability(p,scale):
    return (torch.logit(torch.tensor(p,device='cuda',dtype=torch.float64).clamp(1e-7,1-1e-7))*scale).sigmoid().cpu().numpy()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--online',type=Path,required=True)
    parser.add_argument('--frozen',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--docs',type=Path,required=True)
    args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory')
    torch.set_num_threads(2)
    if args.output.exists():raise RuntimeError('Use a new immutable rating report directory')
    args.output.mkdir(parents=True)
    online_record=read(args.online/'completion.json')
    frozen_record=read(args.frozen/'completion.json')
    lookup={tuple(k):i for i,k in enumerate(read(args.online/'online/keys.json'))}
    raw_online=np.load(args.online/'online/predictions.npz')['probability']
    result=dict(code=sha(__file__),online=sha(args.online/'completion.json'),frozen=sha(args.frozen/'completion.json'),metrics=[],contrasts=[],calibrations=[])
    names=('elo_k20','glicko2','player_gaussian')
    for task in TASKS:
        kk=[];yy=[]
        predictions={regime:[] for regime in ('online_raw','online_calibrated','frozen_raw','frozen_calibrated')}
        for fold in read(CORE/'config.json')['folds']:
            block=fold['name'];b=CORE/'blocks'/block
            calibration_keys=read(b/'inputs'/task/'calibration-keys.json')
            calibration_y=np.load(b/'inputs'/task/'calibration-labels.npy')
            evaluation_keys=read(b/'inputs'/task/'evaluation-keys.json')
            evaluation_y=np.load(CORE/'scored-labels'/block/(task+'.npy'))
            frozen_lookup={tuple(k):i for i,k in enumerate(read(args.frozen/block/'keys.json'))}
            raw_frozen=np.load(args.frozen/block/'predictions.npz')['probability']
            for regime,raw,index in [('online',raw_online,lookup),('frozen',raw_frozen,frozen_lookup)]:
                cp=raw[[index[task,k['game_id']] for k in calibration_keys]]
                ep=raw[[index[task,k['game_id']] for k in evaluation_keys]]
                calibrated=[]
                for column,name in enumerate(names):
                    scale,receipt=calibrate(cp[:,column],calibration_y,calibration_keys)
                    calibrated.append(scale_probability(ep[:,column],scale))
                    result['calibrations'].append(dict(block=block,task=task,regime=regime,model=name,scale=scale,receipt=receipt))
                predictions[regime+'_raw'].append(ep)
                predictions[regime+'_calibrated'].append(np.stack(calibrated,1))
            kk.extend(evaluation_keys);yy.extend(evaluation_y)
        yy=np.asarray(yy)
        pp={regime:np.concatenate(parts) for regime,parts in predictions.items()}
        valid=np.isfinite(yy)&np.array([k['region']!='global' for k in kk])
        keys=[k for k,keep in zip(kk,valid) if keep];y=yy[valid]
        for regime,matrix in pp.items():
            for column,name in enumerate(names):
                result['metrics'].append(dict(task=task,regime=regime,model=name,**metrics(keys,y,matrix[valid,column],matrix[valid,column]>.5)))
        for mode in ('raw','calibrated'):
            for column,name in enumerate(names):
                result['contrasts'].append(dict(task=task,mode=mode,a='online_'+name,b='frozen_'+name,**paired(keys,y,pp['online_'+mode][valid,column],pp['frozen_'+mode][valid,column],131072,20261018)))
            for name,column in [('glicko2',1),('player_gaussian',2)]:
                result['contrasts'].append(dict(task=task,mode=mode,a='online_'+name,b='online_elo_k20',**paired(keys,y,pp['online_'+mode][valid,column],pp['online_'+mode][valid,0],131072,20261018)))
        np.savez_compressed(args.output/(task+'-probabilities.npz'),**pp)
    result['interpretation']='Pointwise exploratory complete-event intervals; no primary historical decisions replaced. Check delay identity before isolating refresh.'
    save(args.output/'results.json',result)
    lines=['# Stronger rating references and matched refresh comparison','','All rows use the original valid broad nonglobal evaluation targets. Probabilities are raw or separately calibrated on the original calibration partition. No evaluation-outcome tuning. Player Gaussian is the declared diagonal TrueSkill-model adaptation; incomplete prior series rosters fall back to team Glicko-2.','','| Task / regime / rating | Event Brier | Event log loss | Ordinary accuracy | Event accuracy |','|---|---:|---:|---:|---:|']
    for r in result['metrics']:
        lines.append(f"| {r['task']} / {r['regime']} / {r['model']} | {r['event_brier']:.6f} | {r['event_log_loss']:.6f} | {100*r['row_accuracy']:.2f}% | {100*r['event_accuracy']:.2f}% |")
    lines+=['','## Paired differences','','Intervals are pointwise, 131,072 complete-event draws, seed 20261018. Negative loss differences favor the first procedure.','','| Task / mode / candidate − reference | Brier difference [95% interval] | Log-loss difference [95% interval] |','|---|---:|---:|']
    for r in result['contrasts']:
        values=[f"{r[m]['event_difference']:+.6f} [{r[m]['event_ci95'][0]:+.6f}, {r[m]['event_ci95'][1]:+.6f}]" for m in ('brier','log_loss')]
        lines.append(f"| {r['task']} / {r['mode']} / {r['a']} − {r['b']} | {' | '.join(values)} |")
    lines+=['',f'Bound outputs: `{args.output}`. Online identity: `{online_record["identity"]}`. Frozen identity: `{frozen_record["identity"]}`.']
    (args.docs/('rating-results-'+args.frozen.name+'.md')).write_text('\n'.join(lines)+'\n')
    save(args.output/'completion.json',dict(results=sha(args.output/'results.json'),code=sha(__file__)))
    print('RATING COMPARISONS COMPLETE',flush=True)


if __name__=='__main__':main()
