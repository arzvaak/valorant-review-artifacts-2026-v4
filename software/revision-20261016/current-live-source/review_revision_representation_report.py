"""Complete predeclared controlled-pooling/channel/projection contrast families."""
import argparse
import collections
from pathlib import Path
import sys

import numpy as np
import torch

from review_revision_diagnostics import CORE, TASKS, metrics, read, save, sha
sys.path.insert(0,str(CORE/'code'))
from score import paired, adjustment

POOLS=('sum','mean','attention','summary')
REPS=('BN','BNC','BNL','BNLC')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fits',type=Path,required=True)
    parser.add_argument('--inference-method',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--docs',type=Path,required=True)
    args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory for inference')
    torch.set_num_threads(2)
    args.output.mkdir(parents=True,exist_ok=True)
    freeze=read(args.fits/'all-predictions-frozen.json')
    for file,expected in freeze['packages'].items():
        if sha(file)!=expected:raise RuntimeError('Frozen player package changed')
        receipt=read(file)
        if sha(Path(file).parent/'predictions.npz')!=receipt['predictions']:raise RuntimeError('Player probability freeze changed')
    identity=dict(code=sha(__file__),fits=sha(args.fits/'all-predictions-frozen.json'),method=sha(args.inference_method))
    if (args.output/'identity.json').exists() and read(args.output/'identity.json')!=identity:raise RuntimeError('Version changed; retain existing report')
    save(args.output/'identity.json',identity)
    result=dict(identity=identity,metrics=[],contrasts=[],counts={})
    for task in TASKS:
        keys=[];labels=[];predictions=collections.defaultdict(list)
        for fold in read(CORE/'config.json')['folds']:
            kk=read(CORE/'blocks'/fold['name']/'inputs'/task/'evaluation-keys.json')
            yy=np.load(CORE/'scored-labels'/fold['name']/(task+'.npy'))
            valid=np.isfinite(yy)&np.array([k['region']!='global' for k in kk])
            keys.extend(k for k,keep in zip(kk,valid) if keep);labels.extend(yy[valid])
            for rep,pool in [(rep,pool) for rep in REPS for pool in POOLS]+[('BNLC','original_sum')]:
                folder='original-BNLC-sum' if pool=='original_sum' else f'corrected-{rep}-{pool}'
                z=np.load(args.fits/fold['name']/task/folder/'predictions.npz')
                predictions[rep+'-'+pool].extend(z['evaluation_calibrated'][valid])
        y=np.asarray(labels);pp={k:np.asarray(p) for k,p in predictions.items()}
        for name,p in pp.items():result['metrics'].append(dict(task=task,package=name,**metrics(keys,y,p,p>.5)))
        contrasts=[]
        for rep in REPS:
            for a,b in [('mean','sum'),('attention','sum'),('attention','mean'),('sum','summary'),('attention','summary')]:
                contrasts.append(('pool80',rep+'-'+a,rep+'-'+b,80))
        for pool in POOLS:
            for a,b in [('BNC','BN'),('BNLC','BNL'),('BNL','BN'),('BNLC','BNC')]:
                contrasts.append(('channel64',a+'-'+pool,b+'-'+pool,64))
        contrasts.append(('projection4','BNLC-sum','BNLC-original_sum',4))
        for family,a,b,budget in contrasts:
            for metric in ('brier','log_loss'):
                identifier=task+'/'+family+'/'+a+'-'+b+'/'+metric
                marker=args.output/(identifier.replace('/','__')+'.json')
                if marker.exists():record=read(marker)
                else:
                    row=dict(task=task,correction_family=family,a=a,b=b,metric=metric,id=identifier)
                    record=paired(row,pp,y,keys,131072,budget,20261020+len(result['contrasts']))
                    save(marker,record)
                result['contrasts'].append(record)
        save(args.output/'progress.json',dict(task=task,contrasts=len(result['contrasts'])))
    adjustment(result['contrasts'])
    result['counts']={family:dict(contrasts=len(group),reject=sum(r['conservative_rejection'] for r in group),benefit=sum(r['conservative_rejection'] and r['difference']<0 for r in group),harm=sum(r['conservative_rejection'] and r['difference']>0 for r in group)) for family in ('pool80','channel64','projection4') if (group:=[r for r in result['contrasts'] if r['correction_family']==family])}
    result['interpretation']='Controlled fixed-recipe sensitivities, conditional on saved fits; summary head differs in width, and original player results remain unchanged.'
    save(args.output/'results.json',result)
    lines=['# Controlled player-pooling and channel results','','All completed conditions are retained. Sum/mean/attention share the encoder/head dimensions; summary pooling uses the same encoded information but a wider head. C and L channels are independently crossed. Original studies and their decisions remain unchanged.','','## Complete family accounting','','| Family | Contrasts | Corrected benefits | Corrected harms | Unresolved |','|---|---:|---:|---:|---:|']
    for family,count in result['counts'].items():lines.append(f"| {family} | {count['contrasts']} | {count['benefit']} | {count['harm']} | {count['contrasts']-count['reject']} |")
    lines+=['','## Full calibrated metrics','','| Task / package | Event Brier | Event log loss | Row accuracy | Event accuracy |','|---|---:|---:|---:|---:|']
    for r in result['metrics']:lines.append(f"| {r['task']} / {r['package']} | {r['event_brier']:.6f} | {r['event_log_loss']:.6f} | {100*r['row_accuracy']:.2f}% | {100*r['event_accuracy']:.2f}% |")
    lines+=['','## All contrasts','','Negative favors first procedure. 131072 complete-event draws; original centered-bootstrap/Monte-Carlo-bound/Holm arithmetic is applied within each new complete family. Intervals condition on the fitted recipes.','','| Task / family / candidate − reference / loss | Difference | Pointwise 95% interval | Adjusted upper p | Decision |','|---|---:|---:|---:|---|']
    for r in result['contrasts']:lines.append(f"| {r['id']} | {r['difference']:+.6f} | [{r['ci95'][0]:+.6f}, {r['ci95'][1]:+.6f}] | {r['holm_mc_upper_p']:.6f} | {'benefit' if r['difference']<0 else 'harm' if r['difference']>0 else 'unresolved'}{' supported' if r['conservative_rejection'] else ' unresolved'} |")
    lines+=['',f'Machine-readable source: `{args.output}/results.json`. Fitting receipts retain parameter counts, epoch/batch telemetry, failures and every checkpoint.']
    (args.docs/'controlled-player-results.md').write_text('\n'.join(lines)+'\n')
    save(args.output/'completion.json',dict(identity=identity,results=sha(args.output/'results.json')))
    print('CONTROLLED PLAYER COMPARISONS COMPLETE',result['counts'],flush=True)


if __name__=='__main__':main()
