"""Bounded reporting diagnostics for retained forecasts, without predictor refits."""
import argparse
import datetime as dt
from pathlib import Path
import shutil

import numpy as np
import torch

from review_revision_diagnostics import (CORE, ONLINE, TASKS, STORE, Inputs,
    load_core, metrics, paired, diagnostic_bootstrap, read, save, sha)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--methods',type=Path,required=True)
    parser.add_argument('--docs',type=Path,required=True)
    args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory for numerical diagnostics')
    torch.set_num_threads(2)
    args.output.mkdir(parents=True,exist_ok=True)
    identity=dict(code=sha(__file__),methods=sha(args.methods))
    if (args.output/'identity.json').exists() and read(args.output/'identity.json')!=identity:raise RuntimeError('Changed diagnostic identity')
    save(args.output/'identity.json',identity)
    shutil.copy2(__file__,args.output/'code.py');shutil.copy2(args.methods,args.output/'methods.md')
    inputs=Inputs();output=dict(identity=identity,tasks=[])
    policy=STORE/'accuracy-predictor-20261010-v1'
    for task in TASKS:
        keys,y,pp,decisions,meta=load_core(task,inputs)
        broad=np.isfinite(y)&np.array([k['region']!='global' for k in keys])
        bkeys=[k for k,b in zip(keys,broad) if b];by=y[broad]
        rating_root=STORE/'manuscript-review-rating-comparison-20261016-frozen6-v1'
        ratings=np.load(inputs.bind(rating_root/(task+'-probabilities.npz')))['online_calibrated']
        if len(ratings)!=len(keys):raise RuntimeError('Rating row alignment changed')
        row=dict(task=task,rating_comparisons=[],policy=[],orientation=[])
        for column,name in ((1,'glicko2'),(2,'player_gaussian')):
            row['rating_comparisons'].append(dict(reference=name,**paired(bkeys,by,pp['rich'][broad],ratings[broad,column],131072,20261022)))
        for fold in read(CORE/'config.json')['folds']:
            block=fold['name'];mask=broad&np.array([k['block']==block for k in keys])
            kk=[k for k,b in zip(keys,mask) if b];yy=y[mask]
            row['orientation'].append(dict(block=block,**metrics(kk,yy,pp['selector'][mask],decisions['selector'][mask])))
            directory=policy/'blocks'/block/task
            receipt=inputs.json(directory/'frozen.json')
            selection=inputs.json(directory/'selection.json',receipt['selection'])
            forecast=np.load(inputs.bind(directory/'predictions.npz',receipt['predictions']))
            local=np.isfinite(y[np.array([k['block']==block for k in keys])])&np.array([k['region']!='global' for k in keys if k['block']==block])
            chosen=selection['choices']['row_accuracy']['chosen']
            p=forecast['row_accuracy_probability'][local];pick=forecast['row_accuracy_pick'][local]
            prior=forecast['training_prior_probability'][local]
            prior_weight=chosen['components'].get('training_prior',0.)
            swapped=1-p+prior_weight*(2*prior-1)
            mismatch=(pick==(swapped>chosen['threshold']))
            row['policy'].append(dict(block=block,chosen=chosen,rows=len(yy),
                threshold_changed_decisions=int((pick!=(p>.5)).sum()),
                accuracy_difference_from_half=float(((pick==yy).astype(float)-((p>.5)==yy)).mean()),
                swap_noncomplementary_decisions=int(mismatch.sum()),
                prior_weight=prior_weight))
        if task=='series_winner':
            reviewed=broad&np.array([k['region'] in ('americas','emea','pacific') and k['population'] in ('open','game_changers') and dt.datetime.fromtimestamp(k['anchor'],dt.timezone.utc).year==2025 for k in keys])
            kk=[k for k,b in zip(keys,reviewed) if b];yy=y[reviewed]
            rr={name:p[reviewed] for name,p in pp.items() if name in ('rich','dynamic','online_elo')}
            feature_root=ONLINE/'rich';columns=inputs.json(feature_root/'columns.json')['numeric']
            feature_index={tuple(k):i for i,k in enumerate(inputs.json(feature_root/'keys.json'))}
            numeric=np.load(inputs.bind(feature_root/'features.npz'))['numeric']
            wanted=[n for n in columns if 'coverage' in n or 'roster_proxy_missing' in n or ('recent' in n and ('acs' in n.lower() or 'ACS' in n))]
            selected=numeric[[feature_index[task,k['game_id']] for k in kk]]
            events=[]
            for event in sorted({k['event_id'] for k in kk}):
                mask=np.array([k['event_id']==event for k in kk]);ek=[k for k,b in zip(kk,mask) if b]
                record=dict(event_id=event,rows=int(mask.sum()),metrics={})
                for name,p in rr.items():
                    record['metrics'][name]=metrics(ek,yy[mask],p[mask],decisions[name][reviewed][mask])
                    record['metrics'][name]['extreme_probability_fraction']=float(((p[mask]<.1)|(p[mask]>.9)).mean())
                record['coverage']={n:dict(observed=int(np.isfinite(selected[mask,columns.index(n)]).sum()),mean=float(np.nanmean(selected[mask,columns.index(n)])) if np.isfinite(selected[mask,columns.index(n)]).any() else None) for n in wanted}
                events.append(record)
            differences=np.array([e['metrics']['rich']['event_brier']-e['metrics']['dynamic']['event_brier'] for e in events])
            row['reviewed2025']=dict(events=events,leave_one_out_brier_difference=[dict(omitted=e['event_id'],difference=float(np.delete(differences,i).mean())) for i,e in enumerate(events)],
                reliability=[dict(package=n,**diagnostic_bootstrap(kk,yy,p,2048,20261023,False)) for n,p in rr.items()])
        output['tasks'].append(row)
    output['inputs']=inputs.hashes;save(args.output/'results.json',output)
    save(args.output/'completion.json',dict(results=sha(args.output/'results.json'),identity=identity))
    print('ADDITIONAL DIAGNOSTICS COMPLETE',flush=True)


if __name__=='__main__':main()
