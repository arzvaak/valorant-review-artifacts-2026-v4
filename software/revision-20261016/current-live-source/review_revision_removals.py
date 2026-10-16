"""Retune semantic removals within each original selected parent architecture."""
import argparse
import gc
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from review_revision_diagnostics import CORE, TASKS, read, save, sha

sys.path.insert(0,str(CORE/'code'))
from common import SEMANTIC
from fit import package
from score import adjustment, paired


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--methods',type=Path,required=True)
    args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory for removal fitting')
    torch.set_num_threads(2)
    args.output.mkdir(parents=True,exist_ok=True)
    identity=dict(code=sha(__file__),methods=sha(args.methods),original_release=sha(CORE/'release-identity.json'),torch=torch.__version__,original_fit_code=sha(CORE/'code/fit.py'))
    if (args.output/'identity.json').exists() and read(args.output/'identity.json')!=identity:raise RuntimeError('Changed method: use new version')
    save(args.output/'identity.json',identity);shutil.copy2(__file__,args.output/'code.py');shutil.copy2(args.methods,args.output/'methods.md')
    cfg=read(CORE/'config.json')
    # This extension searches its own candidates. The historical release's
    # migration helper applies only to its original parent release chain.
    for key in ('reuse_root','ft_reuse_root'):
        cfg.pop(key,None)
    complete=[]
    for fold in cfg['folds']:
        block=fold['name'];original=CORE/'blocks'/block;destination=args.output/'blocks'/block
        destination.mkdir(parents=True,exist_ok=True)
        for name in ('inputs','config.json','release-binding.json'):
            target=destination/name
            if not target.exists():target.symlink_to(original/name)
        prepared=read(original/'inputs/prepared.json')
        for filename,expected in prepared['files'].items():
            if sha(original/'inputs'/filename)!=expected:raise RuntimeError('Original fitting snapshot changed')
        for task in TASKS:
            parent_path=original/f'removals-{task}-parent.json'
            if not parent_path.exists():continue
            parent=read(parent_path)
            if sha(Path(parent['directory'])/'frozen.json')!=parent['freeze_sha256']:raise RuntimeError('Original parent changed')
            for condition in (*SEMANTIC,'coverage'):
                save(args.output/'progress.json',dict(block=block,task=task,condition=condition,completed=len(complete)))
                print('RETUNED REMOVAL',block,task,parent['family'],condition,flush=True)
                try:
                    record=package(destination,task,'BNLC',parent['family'],cfg,stage='retuned_removals',drop=None if condition=='coverage' else condition,coverage=condition=='coverage',params_fixed=None)
                except Exception as error:
                    save(args.output/f'failure-{time.time_ns()}.json',dict(block=block,task=task,condition=condition,error=repr(error)));raise
                marker=Path(record['directory'])/'frozen.json'
                complete.append(dict(block=block,task=task,condition=condition,path=str(marker),sha256=sha(marker),parent=parent))
                gc.collect();torch.cuda.empty_cache()
    save(args.output/'all-predictions-frozen.json',dict(identity=identity,packages=complete,evaluation_labels_read=False))
    if (args.output/'results.json').exists():return
    contrasts=[];denominators=[]
    for task in TASKS:
        keys=[];labels=[];predictions={name:[] for name in (*SEMANTIC,'coverage','full_reference')}
        for fold in cfg['folds']:
            entries=[r for r in complete if r['block']==fold['name'] and r['task']==task]
            if not entries:continue
            if len(entries)!=10:raise RuntimeError('Incomplete removal family')
            original=CORE/'blocks'/fold['name']
            kk=read(original/'inputs'/task/'evaluation-keys.json')
            y=np.load(CORE/'scored-labels'/fold['name']/(task+'.npy'))
            mask=np.isfinite(y)&np.array([k['region']!='global' for k in kk])
            keys.extend(k for k,keep in zip(kk,mask) if keep);labels.extend(y[mask])
            reference=Path(entries[0]['parent']['directory'])/'predictions.npz'
            predictions['full_reference'].extend(np.load(reference)['evaluation_calibrated'][mask])
            for entry in entries:
                file=Path(entry['path']).parent/'predictions.npz'
                predictions[entry['condition']].extend(np.load(file)['evaluation_calibrated'][mask])
            denominators.append(dict(block=fold['name'],task=task,rows=int(mask.sum()),events=len({k['event_id'] for k,keep in zip(kk,mask) if keep})))
        y=np.asarray(labels);pp={n:np.asarray(v) for n,v in predictions.items()}
        for condition in (*SEMANTIC,'coverage'):
            for metric in ('brier','log_loss'):
                row=dict(task=task,correction_family='retuned_removals40',a=condition,b='full_reference',metric=metric,id=task+'/'+condition+'/'+metric)
                contrasts.append(paired(row,pp,y,keys,131072,40,20261019+len(contrasts)))
    adjustment(contrasts)
    save(args.output/'results.json',dict(identity=identity,contrasts=contrasts,denominators=denominators,interpretation='conditional reliance with hyperparameter retuning inside fixed selected architecture; new family, original decisions unchanged'))
    save(args.output/'completion.json',dict(forecasts=sha(args.output/'all-predictions-frozen.json'),results=sha(args.output/'results.json')))
    print('RETUNED REMOVALS COMPLETE',flush=True)


if __name__=='__main__':main()
