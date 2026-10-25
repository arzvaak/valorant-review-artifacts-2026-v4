"""Exact saved-estimator input refresh on the Colab CUDA runtime."""
import numpy as np
from context import CORE, INPUT, OUTPUT, read, atomic, digest, bind, configure_adapters
from common import TASKS, check
from storage import run_unit
from fit import scale_probs

DYNAMIC=('log_odds','team_difference','player_difference','map_difference','agent_context_difference','series_difference','coverage_a','coverage_b','exposure_a','exposure_b','age_a','age_b','roster_change_a','roster_change_b')


def forecast(block,task):
    import xgboost as xgb
    old=INPUT/'dynamic'
    rich=INPUT/'rich'
    recipe=read(old/'blocks'/block/task/'recipe.json')
    keys=read(CORE/'blocks'/block/'inputs'/task/'evaluation-keys.json')
    meta=read(CORE/'blocks'/block/'inputs/prepared.json')['tasks'][task]
    names=meta['feature_names'];semantics=meta['semantic']
    dynkeys=read(old/'online/keys.json')
    lookup={(k['task'],k['game_id']):i for i,k in enumerate(dynkeys)}
    with np.load(old/'online/features.npz',allow_pickle=False) as z:
        dyn=z['x'][[lookup[task,k['game_id']] for k in keys]]
    lookup={tuple(k):i for i,k in enumerate(read(rich/'keys.json'))}
    richnames=read(rich/'columns.json')['numeric'];rix={n:i for i,n in enumerate(richnames)}
    with np.load(rich/'features.npz',allow_pickle=False) as z:
        refreshed=z['numeric'][[lookup[task,k['game_id']] for k in keys]]
    with np.load(CORE/'blocks'/block/'inputs'/task/'evaluation.npz',allow_pickle=False) as z:
        num=z['numeric'].copy();composition=z['composition'].copy()
    changed=[]
    for j,name in enumerate(names):
        if (semantics[j] in ('stars','economy','pistols') or name.startswith('recent90_ACS')) and name in rix:
            num[:,j]=refreshed[:,rix[name]];changed.append(name)
    for j,name in enumerate(names):
        if '_coverage_' not in name:continue
        family=name.split('_')[0]
        if family not in ('stars','economy','pistols'):continue
        group=meta['groups'][j];side=name[-1]
        cc=[i for i,n in enumerate(names) if semantics[i]==family and meta['groups'][i]==group and n.endswith('_'+side) and '_coverage_' not in n]
        num[:,j]=np.isfinite(num[:,cc]).mean(1) if cc else 0
    num[:,names.index('elo_log_odds_difference')]=dyn[:,0]
    data=np.column_stack((num,composition,dyn)).astype('float32')
    expected=names+['composition_'+str(i) for i in range(meta['composition_width'])]+['dynamic_'+s for s in DYNAMIC]
    if recipe['feature_names']!=expected or data.shape[1]!=len(expected):raise RuntimeError('Saved residual feature layout mismatch')
    records=[]
    if recipe['depth'] is None:
        raw=1/(1+np.exp(-np.clip(dyn[:,0],-30,30)))
    else:
        predictions=[]
        for seed in read(old/'config.json')['seeds']:
            path=old/'blocks'/block/task/f'depth-{recipe["depth"]}-seed-{seed}'
            frozen=read(path/'frozen.json');check(path/'model.ubj',frozen['model'])
            model=xgb.Booster();model.load_model(path/'model.ubj');model.set_param({'device':'cuda:0'})
            predictions.append(model.predict(xgb.DMatrix(data,base_margin=dyn[:,0]),iteration_range=(0,frozen['best_iteration']+1)))
            records.append(dict(seed=seed,model=digest(path/'model.ubj'),best_iteration=frozen['best_iteration']))
        raw=np.mean(predictions,axis=0)
    scale=recipe['calibrator'].get('selected',{}).get('scale',1.)
    probability=scale_probs(raw,scale)
    out=OUTPUT/'refresh/blocks'/block/task;out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'predictions.npz',probability=probability,pick=(raw>recipe['decision']['threshold']).astype('int8'),raw=raw)
    atomic(out/'frozen.json',dict(predictions=digest(out/'predictions.npz'),recipe=digest(old/'blocks'/block/task/'recipe.json'),models=records,changed_features=changed,calibrator_retained=recipe['calibrator'],threshold_retained=recipe['decision'],evaluation_scoring_labels_read=False,weights_refitted=False))


def run(identity):
    for fold in read(CORE/'config.json')['folds']:
        for task in TASKS:
            block=fold['name'];out=OUTPUT/'refresh/blocks'/block/task
            def verify():check(out/'predictions.npz',read(out/'frozen.json')['predictions'])
            run_unit('refresh-'+block+'-'+task,out,identity,lambda:forecast(block,task),verify)
    atomic(OUTPUT/'refresh/all-forecasts-frozen.json',dict(identity=identity,complete=True,evaluation_scoring_labels_read=False))


if __name__=='__main__':
    identity=bind();configure_adapters();run(identity)
