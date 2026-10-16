"""Matched later feedback effects on the original scoring membership."""
import argparse
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

from review_revision_diagnostics import (CORE,LATER,TASKS,grouping,metrics,
    paired as loss_intervals,read,save,sha)
sys.path.insert(0,str(CORE/'code'))
from score import paired,adjustment


def accuracy(keys,y,a,b):
    events,index,counts=grouping(keys)
    differences=(a==y).astype(float)-(b==y).astype(float)
    sums=np.bincount(index,weights=differences,minlength=len(events))
    total=torch.tensor(sums,device='cuda',dtype=torch.float64)
    n=torch.tensor(counts,device='cuda',dtype=torch.float64)
    generator=torch.Generator(device='cuda').manual_seed(20261021)
    row=[];event=[]
    for start in range(0,131072,512):
        ix=torch.randint(len(events),(min(512,131072-start),len(events)),device='cuda',generator=generator)
        row.append(total[ix].sum(1)/n[ix].sum(1));event.append((total/n)[ix].mean(1))
    q=torch.tensor([.025,.975],device='cuda',dtype=torch.float64)
    return dict(row_difference=float(differences.mean()),event_difference=float((sums/counts).mean()),
        row_ci95=torch.quantile(torch.cat(row),q).cpu().tolist(),
        event_ci95=torch.quantile(torch.cat(event),q).cpu().tolist(),draws=131072,seed=20261021)


def main():
    p=argparse.ArgumentParser();p.add_argument('--feedback',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--methods',type=Path,required=True)
    p.add_argument('--docs',type=Path,required=True);args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory')
    torch.set_num_threads(2)
    if args.output.exists():raise RuntimeError('Use new immutable feedback report path')
    args.output.mkdir(parents=True)
    identity=dict(code=sha(__file__),methods=sha(args.methods),inputs={})
    for root in (LATER/'forecasts',LATER/'targets',args.feedback/'forecasts',LATER/'reference',args.feedback/'reference'):
        marker=read(root/'frozen.json');identity['inputs'][str(root/'frozen.json')]=sha(root/'frozen.json')
        for name,h in marker['files'].items():
            if sha(root/name)!=h:raise RuntimeError('Changed bound feedback input')
    order=read(LATER/'targets/keys.json')
    if order!=read(LATER/'forecasts/keys.json') or order!=read(args.feedback/'forecasts/keys.json'):raise RuntimeError('Scoring membership changed')
    contexts={(r['task'],r['game_id']):r for file in ('map-context.json','series-context.json') for r in read(LATER/'context'/file)}
    y=np.load(LATER/'targets/labels.npy');old=np.load(LATER/'forecasts/predictions.npz');new=np.load(args.feedback/'forecasts/predictions.npz')
    oldstate=np.load(LATER/'reference/features.npz')['dynamic'];newstate=np.load(args.feedback/'reference/features.npz')['dynamic']
    result=dict(identity=identity,tasks=[],contrasts=[])
    arms=('rich_residual','original_residual','dynamic_strength','online_elo_k20')
    for task in TASKS:
        selected=np.array([t==task for t,g in order])&np.isfinite(y)
        keys=[contexts[tuple(k)] for k,b in zip(order,selected) if b];yy=y[selected]
        task_result=dict(task=task,metrics=[],accuracy=[],coverage={})
        predictions={}
        for arm in arms:
            predictions[arm+'_new']=new[arm+'_probability'][selected]
            predictions[arm+'_old']=old[arm+'_probability'][selected]
            for mode,z in (('feedback',new),('score_only',old)):
                task_result['metrics'].append(dict(arm=arm,mode=mode,**metrics(keys,yy,z[arm+'_probability'][selected],z[arm+'_pick'][selected])))
            task_result['accuracy'].append(dict(arm=arm,**accuracy(keys,yy,new[arm+'_pick'][selected],old[arm+'_pick'][selected])))
            row=loss_intervals(keys,yy,predictions[arm+'_new'],predictions[arm+'_old'],131072,20261021)
            for metric in ('brier','log_loss'):
                spec=dict(id=task+'/'+arm+'/'+metric,task=task,a=arm+'_new',b=arm+'_old',metric=metric,correction_family='later_feedback16')
                contrast=paired(spec,predictions,yy,keys,131072,16,20261021)
                contrast['row_weighted']=row[metric]
                result['contrasts'].append(contrast)
        for mode,z in (('score_only',oldstate),('feedback',newstate)):
            task_result['coverage'][mode]=dict(rows=int(selected.sum()),
                both_zero_team_exposure=int((z[selected,8:10]==0).all(1).sum()),
                both_full_player_state=int((z[selected,6:8]==1).all(1).sum()),
                either_zero_player_state=int((z[selected,6:8]==0).any(1).sum()))
        result['tasks'].append(task_result)
    adjustment(result['contrasts'])
    audit=read(args.feedback/'reference/history-audit.json')
    result['feedback_accounting']=dict(contexts=len(order),contexts_with_acquired_feedback=sum(bool(r['new_outcome_feedback']) for r in audit),
        admitted_events=read(args.feedback/'feedback-policy.json')['eligible_event_ids'],
        projected_map_observations=len(read(args.feedback/'observations.json')['records']),
        interpretation='all-context acquired-feedback count; performance metrics/coverage use valid-target rows only; source admission differs from feedback already released')
    save(args.output/'results.json',result);shutil.copy2(__file__,args.output/'code.py');shutil.copy2(args.methods,args.output/'methods.md')
    save(args.output/'completion.json',dict(results=sha(args.output/'results.json'),identity=identity))
    print('FEEDBACK COMPARISONS COMPLETE',flush=True)


if __name__=='__main__':main()
