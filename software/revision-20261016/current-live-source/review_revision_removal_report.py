"""Report the complete retuned sensitivity after all predictions are frozen."""
import argparse
from pathlib import Path
import shutil
import numpy as np
from review_revision_diagnostics import CORE,TASKS,metrics,read,save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--fits',type=Path,required=True)
    p.add_argument('--docs',type=Path,required=True);args=p.parse_args()
    freeze=read(args.fits/'all-predictions-frozen.json');results=read(args.fits/'results.json')
    completion=read(args.fits/'completion.json')
    if completion['forecasts']!=sha(args.fits/'all-predictions-frozen.json') or completion['results']!=sha(args.fits/'results.json'):raise RuntimeError('Changed removal results')
    if len(freeze['packages'])!=160 or len(results['contrasts'])!=40:raise RuntimeError('Incomplete retuned sensitivity')
    rows=[];inventory=[]
    for task in TASKS:
        kk=[];yy=[];predictions={}
        for fold in read(CORE/'config.json')['folds']:
            packages=[r for r in freeze['packages'] if r['task']==task and r['block']==fold['name']]
            if not packages:continue
            if len(packages)!=10:raise RuntimeError('Incomplete supported block')
            keys=read(CORE/'blocks'/fold['name']/'inputs'/task/'evaluation-keys.json')
            y=np.load(CORE/'scored-labels'/fold['name']/(task+'.npy'))
            valid=np.isfinite(y)&np.array([k['region']!='global' for k in keys])
            kk.extend(k for k,b in zip(keys,valid) if b);yy.extend(y[valid])
            parent=packages[0]['parent'];directory=Path(parent['directory'])
            if sha(directory/'frozen.json')!=parent['freeze_sha256']:raise RuntimeError('Full parent changed')
            predictions.setdefault('full_reference',[]).extend(np.load(directory/'predictions.npz')['evaluation_calibrated'][valid])
            for package in packages:
                if sha(package['path'])!=package['sha256']:raise RuntimeError('Retuned package changed')
                record=read(package['path']);directory=Path(package['path']).parent
                for name,h in record['files'].items():
                    if sha(directory/name)!=h:raise RuntimeError('Retuned package file changed')
                predictions.setdefault(package['condition'],[]).extend(np.load(directory/'predictions.npz')['evaluation_calibrated'][valid])
                selection=read(directory/'selection.json')
                inventory.append(dict(block=fold['name'],task=task,condition=package['condition'],family=parent['family'],trained=record['trained'],fallback=selection['fallback'],reason=selection.get('reason'),candidates=len(record['model_records']),failed_candidates=sum(r['record']['status']=='failed' for r in record['model_records'])))
        for condition,values in predictions.items():
            prob=np.asarray(values);rows.append(dict(task=task,condition=condition,**metrics(kk,np.asarray(yy),prob,prob>.5)))
    summary=dict(code=sha(__file__),fit_completion=sha(args.fits/'completion.json'),metrics=rows,inventory=inventory,contrasts=results['contrasts'])
    save(args.fits/'report-summary.json',summary);shutil.copy2(__file__,args.fits/'report-code.py')
    lines=['# Complete retuned-removal sensitivity','','Every reduced condition searches the original selected full parent\'s architecture grid using the original train/stopping/selection/calibration phases and seeds. Full reference is the original already-searched parent. Original fixed-configuration removals remain in S4. All 160 new packages froze before evaluation labels were read. Eight supported blocks use 36,624 maps and 16,319 series across 720 events; the same four unsupported blocks remain excluded by the original eligibility rule. The new 40 contrasts receive their own 131,072-draw MonteCarlo/Holm family; original decisions are unchanged.','','## Absolute calibrated scores','','| Task/condition | Event Brier | Event log loss | Row accuracy | Event accuracy |','|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['task']}/{r['condition']} | {r['event_brier']:.6f} | {r['event_log_loss']:.6f} | {r['row_accuracy']:.6f} | {r['event_accuracy']:.6f} |")
    lines+=['','## All paired reduced-minus-full contrasts','','Negative loss difference favors the reduced procedure. Pointwise intervals condition on saved fits; a favorable unadjusted estimate does not supersede the complete corrected family.','','| Contrast | Difference | Pointwise95% interval | Conservative Holm | Decision |','|---|---:|---|---:|---|']
    for r in results['contrasts']:
        decision=('benefit' if r['difference']<0 else 'harm') if r['conservative_rejection'] else 'unresolved'
        lines.append(f"| {r['id']} | {r['difference']:+.6f} | [{r['ci95'][0]:+.6f}, {r['ci95'][1]:+.6f}] | {r['holm_mc_upper_p']:.6f} | {decision} |")
    lines+=['','## Real fitting/failure accounting','',f"Packages: {len(inventory)}; fitted packages: {sum(r['trained'] for r in inventory)}; retained fallback packages: {sum(r['fallback'] for r in inventory)}; candidate records: {sum(r['candidates'] for r in inventory)}; failed candidate records: {sum(r['failed_candidates'] for r in inventory)}. Every candidate/checkpoint/failure remains in its fitting directory."]
    (args.docs/'retuned-removal-findings.md').write_text('\n'.join(lines)+'\n')
    print('RETUNED SENSITIVITY REPORT COMPLETE',len(inventory),'packages',flush=True)


if __name__=='__main__':main()
