"""Post-freeze event inference for the four declared Colab families."""
import argparse
import collections
import numpy as np
from context import CORE, INPUT, OUTPUT, CONTROL, PERSIST, read, atomic, digest, bind
from common import TASKS, check, weights
from score import event_losses, adjustment

FAMILIES=('linear','catboost','xgboost','mlp','tabm')


def arrays(policy,task):
    keys=[];labels=[];probs=collections.defaultdict(list);receipts=[]
    for fold in read(CORE/'config.json')['folds']:
        block=fold['name'];original=CORE/'blocks'/block
        dest=(CONTROL if policy=='frozen2h15m' else OUTPUT)/policy/'blocks'/block
        if policy=='refresh':
            freeze=read(OUTPUT/'refresh/all-forecasts-frozen.json')
        else:
            freeze=read(dest/'forecasts-frozen.json')
            for relative,expected in freeze['files'].items():check(dest/relative,expected)
            receipts.append(dict(block=block,sha256=digest(dest/'forecasts-frozen.json')))
        kk=read(original/'inputs'/task/'evaluation-keys.json')
        labelpath=INPUT/'labels'/block/(task+'.npy');binding=read(labelpath.with_suffix('.json'))
        check(labelpath,binding['sha256']);check(original/'inputs'/task/'evaluation-keys.json',binding['keys_sha256'])
        if binding['release_sha256']!=digest(CORE/'release-identity.json'):raise RuntimeError('Scoring label release mismatch')
        y=np.load(labelpath,allow_pickle=False)
        if len(y)!=len(kk):raise RuntimeError('Scoring label alignment')
        keys.extend(kk);labels.append(y)
        if policy=='refresh':
            paths={'refresh':dest/task/'predictions.npz','earlier':INPUT/'dynamic/blocks'/block/task/'predictions.npz','rich_refit':INPUT/'rich_residual/blocks'/block/task/'predictions.npz'}
            for name,path in paths.items():
                with np.load(path,allow_pickle=False) as z:probs[name].append(z['probability'].copy())
        else:
            names=[r+'-'+f for r in ('B','BN') for f in FAMILIES] if policy in ('frozen2h15m','fresh2h15m') else [r+'-'+f for r in ('B','BN') for f in ('catboost','xgboost','tabm')]+['B-linear']
            for name in names:
                path=dest/'packages/cloudfixed'/task/name/'predictions.npz'
                with np.load(path,allow_pickle=False) as z:probs[name].append(z['evaluation_calibrated'].copy())
            selector=dest/'selector'/task
            if selector.exists():
                check(selector/'predictions.npz',read(selector/'frozen.json')['predictions'])
                with np.load(selector/'predictions.npz',allow_pickle=False) as z:
                    probs['selector'].append(z['evaluation_calibrated'].copy());probs['elo_k20'].append(z['evaluation_elo_calibrated'].copy())
    y=np.concatenate(labels)
    mask=np.isfinite(y)&np.asarray([k['region']!='global' for k in keys])
    selected=[k for k,m in zip(keys,mask) if m]
    if len({k['game_id'] for k in selected})!=len(selected):raise RuntimeError('Duplicate scored development unit')
    result={}
    for name,values in probs.items():
        if len(values)!=len(read(CORE/'config.json')['folds']):raise RuntimeError('Incomplete predictor across blocks: '+name)
        p=np.concatenate(values)
        if len(p)!=len(y) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise RuntimeError('Invalid frozen probabilities')
        result[name]=p[mask]
    return selected,y[mask],result,receipts


def bootstrap(delta,row,nrows,family_size,seed):
    import torch
    from scipy.stats import beta
    draws=131072;n=len(delta)
    if n<2:raise RuntimeError('Insufficient whole events')
    d=torch.as_tensor(delta,device='cuda',dtype=torch.float64)
    generator=torch.Generator(device='cuda').manual_seed(seed);samples=[]
    for start in range(0,draws,256):
        ix=torch.randint(n,(min(256,draws-start),n),device='cuda',generator=generator)
        samples.append(d[ix].mean(1))
    boot=torch.cat(samples);point=float(d.mean());extreme=int(((boot-point).abs()>=abs(point)).sum())
    ci=torch.quantile(boot,torch.tensor([.025,.975],device='cuda',dtype=torch.float64)).cpu().tolist()
    upper=float(beta.ppf(1-.01/family_size,extreme+1,draws-extreme)) if extreme<draws else 1.
    return dict(row,difference=point,ci95=ci,rows=nrows,events=n,draws=draws,seed=seed,extreme_count=extreme,p_approx=(extreme+1)/(draws+1),mc_upper_p=upper,mc_simultaneous_confidence=.99,interpretation='negative favors first procedure or a more favorable fresh BN-minus-B increment; conditional on saved fits')


def metrics(p,y,keys):
    w=weights(keys);clipped=np.clip(p,1e-7,1-1e-7);ll=-y*np.log(clipped)-(1-y)*np.log1p(-clipped)
    return dict(rows=len(y),events=len({k['event_id'] for k in keys}),event_brier=float(w@((clipped-y)**2)),event_log_loss=float(w@ll),row_brier=float(np.mean((clipped-y)**2)),row_log_loss=float(ll.mean()),event_accuracy=float(w@((p>.5)==y)),row_accuracy=float(np.mean((p>.5)==y)),exact_ties=int((p==.5).sum()),row_half_credit_accuracy=float(np.mean(np.where(p==.5,.5,(p>.5)==y))))


def report(family,identity):
    required={'fresh20':['fresh2h15m'],'change20':['fresh2h15m','frozen2h15m'],'clocks40':['frozen6h30m','frozen24h60m'],'refresh8':['refresh']}[family]
    # Check every block before any scalar scoring-label read in arrays().
    records=[]
    for policy in required:
        base=CONTROL if policy=='frozen2h15m' else OUTPUT
        paths=[OUTPUT/'refresh/all-forecasts-frozen.json'] if policy=='refresh' else [base/policy/'blocks'/f['name']/'forecasts-frozen.json' for f in read(CORE/'config.json')['folds']]
        expected_identity=identity['retained_control']['identity'] if policy=='frozen2h15m' and identity.get('retained_control') else identity
        for path in paths:
            if read(path).get('identity')!=expected_identity:raise RuntimeError('Missing/changed complete-family forecast freeze')
            records.append(dict(path=str(path),sha256=digest(path)))
    out=OUTPUT/'reports'/family;out.mkdir(parents=True,exist_ok=True)
    atomic(out/'scoring-activated.json',dict(identity=identity,forecasts=records,scope='broad_nonglobal',family=family))
    rows=[];summaries=[];ordinal=0;size=int(''.join(c for c in family if c.isdigit()))
    for task in TASKS:
        loaded={p:arrays(p,task) for p in required}
        first=loaded[required[0]];keys,y=first[:2]
        for policy,(kk,yy,pp,_) in loaded.items():
            if kk!=keys or not np.array_equal(yy,y):raise RuntimeError('Policy scoring-row mismatch')
            for name,p in pp.items():summaries.append(dict(policy=policy,task=task,predictor=name,**metrics(p,y,keys)))
        times={}
        for k in keys:times[k['event_id']]=min(times.get(k['event_id'],float('inf')),k['prediction_cutoff'])
        events=sorted(times,key=lambda e:(times[e],e))
        pairs=[]
        if family in ('fresh20','change20'):
            pairs=[(required[0],'BN-'+f,'B-'+f) for f in FAMILIES]
        elif family=='clocks40':
            pairs=[(policy,a,b) for policy in required for a,b in [('BN-'+f,'B-'+f) for f in ('catboost','xgboost','tabm')]+[('selector','elo_k20'),('selector','B-linear')]]
        else:pairs=[('refresh','refresh','earlier'),('refresh','rich_refit','refresh')]
        for policy,a,b in pairs:
            for metric in ('brier','log_loss'):
                pred=loaded[policy][2];aa=event_losses(pred[a],y,keys,metric);bb=event_losses(pred[b],y,keys,metric)
                delta=np.asarray([aa[e]-bb[e] for e in events])
                if family=='change20':
                    control=loaded['frozen2h15m'][2];ca=event_losses(control[a],y,keys,metric);cb=event_losses(control[b],y,keys,metric)
                    delta-=np.asarray([ca[e]-cb[e] for e in events])
                row=dict(id='/'.join((family,policy,task,a,b,metric)),correction_family=family,policy=policy,task=task,a=a,b=b,metric=metric)
                rows.append(bootstrap(delta,row,len(y),size,20261024+ordinal));ordinal+=1
                atomic(out/'partial.json',dict(contrasts=rows,metrics=summaries))
    if len(rows)!=size:raise RuntimeError('Declared family incomplete')
    adjustment(rows)
    result=dict(identity=identity,scope='broad_nonglobal',contrasts=rows,metrics=summaries,conditional_saved_fit_uncertainty=True,exploratory=True)
    atomic(out/'results.json',result);atomic(PERSIST/'reports'/family/'results.json',result)
    atomic(out/'completion.json',dict(results=digest(out/'results.json'),scoring_activation=digest(out/'scoring-activated.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--family',choices=('fresh20','change20','clocks40','refresh8'),required=True)
    args=parser.parse_args();report(args.family,bind())
