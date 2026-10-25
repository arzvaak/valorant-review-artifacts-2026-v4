"""Bounded real fits on Colab, with complete-unit durable recovery."""
import argparse
import gc
from pathlib import Path
import shutil
import time
import numpy as np
from context import CORE, INPUT, OUTPUT, PERSIST, WORK, read, atomic, digest, bind, configure_adapters
from storage import run_unit
from history import prepare
from common import PHASES, TASKS, verify_files, save_json, save_npz, check
import fit

POLICIES = {'frozen2h15m':(7200,900,False), 'fresh2h15m':(7200,900,True),
            'frozen6h30m':(21600,1800,False), 'frozen24h60m':(86400,3600,False)}


def verify_package(directory):
    receipt=read(directory/'frozen.json')
    verify_files(directory,receipt['files'])
    for item in receipt['model_records']:
        record=item['record']
        if record['status']!='fitted':
            continue
        path=Path(item['path'])
        check(record['model'],record['model_sha256'])
        check(path/'predictions.npz',record['prediction_sha256'])
        check(path/'processor.json',record['processor_sha256'])
    return receipt


def fit_one(dest,original,task,name,cfg):
    representation,family=name.split('-',1)
    selection=read(original/'packages/core'/task/name/'selection.json')
    parameters=selection['selected']['record']['identity']['parameters'] if selection['selected'] else None
    eligible=fit.eligible
    if parameters is None:
        fit.eligible=lambda *args,**kwargs:False
    try:
        receipt=fit.package(dest,task,representation,family,cfg,stage='cloudfixed',params_fixed=parameters)
        outcome=read(Path(receipt['directory'])/'selection.json')
        if outcome.get('reason')=='candidate_or_selected_seed_failure':
            raise RuntimeError('Preserved numerical candidate failure; repair/version required, not a support fallback')
        return receipt
    finally:
        fit.eligible=eligible


def selector(dest,original,task):
    keys={p:read(dest/'inputs'/task/f'{p}-keys.json') for p in PHASES}
    components=read(original/'selector'/task/'selection.json')['chosen']['components']
    rating=fit.baselines(dest,task)
    raw={p:np.zeros(len(keys[p]),np.float64) for p in PHASES}
    for name,weight in components.items():
        if name=='elo_k20':
            probability=rating
        else:
            with np.load(dest/'packages/cloudfixed'/task/name/'predictions.npz') as z:
                probability={p:z[p+'_raw'] for p in PHASES}
        for phase in PHASES:
            raw[phase]+=weight*probability[phase]
    labels=np.load(dest/'inputs'/task/'calibration-labels.npy',allow_pickle=False)
    scale,calibrator=fit.calibrated(raw['calibration'],labels,keys['calibration'])
    rating_scale,rating_calibrator=fit.calibrated(rating['calibration'],labels,keys['calibration'])
    out=dest/'selector'/task
    save_npz(out/'predictions.npz',**{p+'_calibrated':fit.scale_probs(v,scale) for p,v in raw.items()},
             **{p+'_elo_calibrated':fit.scale_probs(v,rating_scale) for p,v in rating.items()})
    save_json(out/'frozen.json',dict(components=components,calibrator=calibrator,rating_calibrator=rating_calibrator,
                                    predictions=digest(out/'predictions.npz'),evaluation_labels_read=False))


def run(policy, block_name, identity):
    import torch
    delay,lead,fresh=POLICIES[policy]
    cfg=read(CORE/'config.json')
    cfg.pop('reuse_root',None);cfg.pop('ft_reuse_root',None)
    cfg.update(reveal_minutes=delay/60,lead_minutes=lead/60)
    original=CORE/'blocks'/block_name
    dest=OUTPUT/policy/'blocks'/block_name
    dest.mkdir(parents=True,exist_ok=True)
    atomic(dest/'config.json',cfg)
    binding=dict(config_sha256=digest(dest/'config.json'),colab_identity=identity,policy=policy)
    atomic(dest/'release-binding.json',binding)
    def prepare_action():
        if policy=='frozen2h15m':
            if not (dest/'inputs').exists():
                shutil.copytree(original/'inputs',dest/'inputs')
        else:
            prepare(dest,original,delay,lead,identity,fresh=fresh)
    def prepare_verify():
        receipt=read(dest/'inputs/prepared.json')
        verify_files(dest/'inputs',receipt['files'])
        for task in TASKS:
            a=read(original/'inputs'/task/'evaluation-keys.json')
            b=read(dest/'inputs'/task/'evaluation-keys.json')
            if [(k['game_id'],k['event_id'],k['team_a_id'],k['team_b_id']) for k in a]!=[(k['game_id'],k['event_id'],k['team_a_id'],k['team_b_id']) for k in b]:
                raise RuntimeError('Evaluation key alignment changed')
    run_unit(f'{policy}-{block_name}-inputs',dest/'inputs',binding,prepare_action,prepare_verify)
    for task in TASKS:
        original_selection=read(original/'inputs'/task/'selection-keys.json')
        evaluation=read(dest/'inputs'/task/'evaluation-keys.json')
        if original_selection and evaluation and max(k['event_completion'] for k in original_selection)+delay>=min(k['prediction_cutoff'] for k in evaluation):
            raise RuntimeError('Retained original selection labels are not eligible by stricter evaluation')
        if policy in ('frozen2h15m','fresh2h15m'):
            units={f'{r}-{f}' for r in ('B','BN') for f in ('linear','catboost','xgboost','mlp','tabm')}
        else:
            components=read(original/'selector'/task/'selection.json')['chosen']['components']
            units={f'{r}-{f}' for r in ('B','BN') for f in ('catboost','xgboost','tabm')}|{'B-linear'}|{n for n in components if n!='elo_k20'}
        for name in sorted(units):
            directory=dest/'packages/cloudfixed'/task/name
            atomic(PERSIST/'progress.json',dict(policy=policy,block=block_name,task=task,package=name,status='fitting',updated=time.time()))
            print('REAL COLAB FIT',policy,block_name,task,name,flush=True)
            run_unit(f'{policy}-{block_name}-{task}-{name}',directory,binding,
                     lambda:fit_one(dest,original,task,name,cfg),lambda:verify_package(directory))
            gc.collect();torch.cuda.empty_cache()
        if policy not in ('frozen2h15m','fresh2h15m'):
            directory=dest/'selector'/task
            def verify_selector():
                check(directory/'predictions.npz',read(directory/'frozen.json')['predictions'])
            run_unit(f'{policy}-{block_name}-{task}-selector',directory,binding,
                     lambda:selector(dest,original,task),verify_selector)
    frozen_files={str(path.relative_to(dest)):digest(path) for base in (dest/'packages',dest/'selector') if base.exists() for path in base.rglob('*') if path.is_file() and path.name in ('predictions.npz','frozen.json','selection.json','processor.json')}
    atomic(dest/'forecasts-frozen.json',dict(identity=identity,policy=policy,block=block_name,files=frozen_files,evaluation_scoring_labels_read=False,finished=time.time()))
    atomic(PERSIST/'progress.json',dict(policy=policy,block=block_name,status='forecasts_complete',updated=time.time()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--policy',choices=POLICIES,required=True)
    parser.add_argument('--block',required=True)
    args=parser.parse_args()
    identity=bind();configure_adapters()
    if args.block not in {f['name'] for f in read(CORE/'config.json')['folds']}:
        raise RuntimeError('Unknown original development block')
    run(args.policy,args.block,identity)


if __name__=='__main__':
    main()
