"""Exact raw/calibrated tie inventory from original retained forecast packages."""
import argparse
import collections
from pathlib import Path
import shutil
import numpy as np
from review_revision_diagnostics import CORE,TASKS,Inputs,read,save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise RuntimeError('Preserve existing tie inventory')
    args.output.mkdir(parents=True);inputs=Inputs();records=[]
    for fold in inputs.json(CORE/'config.json')['folds']:
        block=CORE/'blocks'/fold['name']
        for task in TASKS:
            keys=inputs.json(block/'inputs'/task/'evaluation-keys.json')
            label_file=CORE/'scored-labels'/fold['name']/(task+'.npy');receipt=inputs.json(label_file.with_suffix('.json'))
            y=np.load(inputs.bind(label_file,receipt['sha256']))
            mask=np.isfinite(y)&np.array([k['region']!='global' for k in keys])
            directories=list((block/'packages/core'/task).glob('*/frozen.json'))+[block/'selector'/task/'frozen.json']
            for marker in directories:
                frozen=inputs.json(marker);root=marker.parent
                selection=inputs.json(root/'selection.json',frozen['files']['selection.json'])
                z=np.load(inputs.bind(root/'predictions.npz',frozen['files']['predictions.npz']))
                for name,raw,cal in ([('selector','evaluation_raw','evaluation_calibrated'),('block_elo','evaluation_elo_raw','evaluation_elo_calibrated')] if root.parent.name=='selector' else [(root.name,'evaluation_raw','evaluation_calibrated')]):
                    a=z[raw][mask]==.5;b=z[cal][mask]==.5
                    records.append(dict(task=task,block=fold['name'],package=name,rows=int(mask.sum()),raw_ties=int(a.sum()),calibrated_ties=int(b.sum()),new_calibrated_ties=int((b&~a).sum()),removed_raw_ties=int((a&~b).sum()),
                        prediction_fallback=selection.get('fallback'),fallback_reason=selection.get('reason'),calibrator=selection.get('calibrator')))
    save(args.output/'results.json',dict(code=sha(__file__),inputs=inputs.hashes,records=records));shutil.copy2(__file__,args.output/'code.py')
    save(args.output/'completion.json',dict(results=sha(args.output/'results.json')))
    print('RAW/CALIBRATED TIE INVENTORY COMPLETE',len(records),flush=True)


if __name__=='__main__':main()
