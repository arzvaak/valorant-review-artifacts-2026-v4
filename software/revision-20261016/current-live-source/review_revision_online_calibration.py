"""Reporting-only online calibration intervals from immutable forecasts."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import torch
from review_revision_diagnostics import CORE,TASKS,Inputs,load_core,diagnostic_bootstrap,read,save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--methods',type=Path,required=True);args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory')
    torch.set_num_threads(2);args.output.mkdir(parents=True,exist_ok=True)
    identity=dict(code=sha(__file__),methods=sha(args.methods))
    if (args.output/'identity.json').exists() and read(args.output/'identity.json')!=identity:raise RuntimeError('Changed identity')
    save(args.output/'identity.json',identity);shutil.copy2(__file__,args.output/'code.py');shutil.copy2(args.methods,args.output/'methods.md')
    result=[];inputs=Inputs()
    for task in TASKS:
        keys,y,p,d,metadata=load_core(task,inputs)
        valid=np.isfinite(y)&np.array([k['region']!='global' for k in keys]);kk=[k for k,b in zip(keys,valid) if b]
        for name in ('rich','earlier','dynamic','online_elo'):
            file=args.output/(task+'-'+name+'.json')
            if not file.exists():
                r=dict(task=task,package=name,**diagnostic_bootstrap(kk,y[valid],p[name][valid],2048,20261024,False))
                save(file,r)
            result.append(read(file));print('ONLINE RELIABILITY',task,name,flush=True)
    save(args.output/'results.json',dict(identity=identity,inputs=inputs.hashes,diagnostics=result))
    save(args.output/'completion.json',dict(results=sha(args.output/'results.json')))


if __name__=='__main__':main()
