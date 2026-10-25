"""Descriptive saved-output and schedule audits; no new model selection."""
import collections
import numpy as np
from context import CORE, INPUT, OUTPUT, PERSIST, read, atomic, digest, bind
from common import TASKS, weights
from report import metrics
from fit import map_baseline


def ties(p,y,keys):
    tied=p==.5;correct=((p>.5)==y).astype(float);half=np.where(tied,.5,correct);w=weights(keys)
    return dict(rows=len(y),events=len({k['event_id'] for k in keys}),ties=int(tied.sum()),tied_team_a_wins=int(y[tied].sum()),row_accuracy=float(correct.mean()),event_accuracy=float(w@correct),row_half_credit_accuracy=float(half.mean()),event_half_credit_accuracy=float(w@half),tied_team_a_fraction=float(y[tied].mean()) if tied.any() else None)


def main(identity):
    records=[];calibrators=[];parent_rows=collections.defaultdict(lambda:dict(keys=[],y=[],p=[]))
    for fold in read(CORE/'config.json')['folds']:
        block=fold['name'];b=CORE/'blocks'/block
        for task in TASKS:
            kk=read(b/'inputs'/task/'evaluation-keys.json');y=np.load(INPUT/'labels'/block/(task+'.npy'),allow_pickle=False)
            mask=np.isfinite(y)&np.asarray([k['region']!='global' for k in kk]);keys=[k for k,m in zip(kk,mask) if m];yy=y[mask]
            directories=list((b/'packages/core'/task).glob('*/selection.json'))+[b/'selector'/task/'selection.json']
            for path in directories:
                selection=read(path);folder=path.parent
                with np.load(folder/'predictions.npz',allow_pickle=False) as z:
                    pairs=[(folder.name,'evaluation_raw','evaluation_calibrated')]
                    if folder.parent.name=='selector':
                        pairs=[('selector','evaluation_raw','evaluation_calibrated'),('block_elo','evaluation_elo_raw','evaluation_elo_calibrated'),('training_prior',None,'training_prior'),('chance',None,'chance')]
                        if 'evaluation_map_history' in z.files:pairs.append(('map_history',None,'evaluation_map_history'))
                    for name,rawfield,calfield in pairs:
                        if calfield not in z.files:raise RuntimeError('Missing original baseline column: '+calfield)
                        p=z[calfield][mask];raw=z[rawfield][mask] if rawfield else p
                        origin='unresolved exact-tie origin'
                        if name=='chance':origin='explicit chance baseline'
                        elif name=='training_prior':origin='training-only prior'
                        elif selection.get('fallback'):origin='explicit sparse-history rating fallback; individual rating tie origin unresolved'
                        elif name=='selector':origin=('explicit sparse selector fallback' if selection.get('sparse_fixed_fallback') else 'selected K20 rating' if selection.get('chosen',{}).get('components')=={'elo_k20':1.} else 'selected mixture')+'; individual component tie origins not uniquely identified'
                        cal=None if name in ('chance','training_prior','map_history') else selection.get('calibrator') if name!='block_elo' else selection.get('elo_calibrator',selection.get('rating_calibrator'))
                        if cal and cal.get('selected',{}).get('scale')==0:origin+='; selected zero-scale calibration yields 0.5'
                        records.append(dict(block=block,task=task,procedure=name,origin=origin,raw_ties=int((raw==.5).sum()),new_calibrated_ties=int(((p==.5)&(raw!=.5)).sum()),**ties(p,yy,keys)))
                        if cal:calibrators.append(dict(block=block,task=task,procedure=name,calibrator=cal,selection_objective='minimum separate calibration-event log loss; insufficient support uses identity'))
            if task=='map_winner':
                p=map_baseline(b)['evaluation'][mask]
                records.append(dict(block=block,task=task,procedure='map_history',origin='block-local map-history logit; absent versus equal history not uniquely identified',raw_ties=int((p==.5).sum()),new_calibrated_ties=0,**ties(p,yy,keys)))
            parent=INPUT/'removals'/block/task/'parent'
            if parent.exists():
                with np.load(parent/'predictions.npz',allow_pickle=False) as z:p=z['evaluation_calibrated'][mask]
                target=parent_rows[task];target['keys'].extend(keys);target['y'].extend(yy);target['p'].extend(p)
    parents=[dict(task=task,**metrics(np.asarray(v['p']),np.asarray(v['y']),v['keys'])) for task,v in parent_rows.items()]
    registry=read(INPUT/'registry/maps.json');matches={};maps=collections.defaultdict(set)
    for row in registry:
        if row['region']=='global':continue
        mid=row['match_id'];maps[mid].add(row['game_id'])
        identity_fields=(row['anchor'],row['team_a_id'],row['team_b_id'])
        if mid in matches and matches[mid]!=identity_fields:raise RuntimeError('Inconsistent schedule/team match identity')
        matches[mid]=identity_fields
    team_times=collections.defaultdict(dict);player_times=collections.defaultdict(dict)
    for mid,(anchor,ta,tb) in matches.items():
        for team in (ta,tb):team_times[team][mid]=anchor
    import duckdb
    con=duckdb.connect(str(INPUT/'source.duckdb'),read_only=True)
    game_to_match={r['game_id']:r['match_id'] for r in registry if r['match_id'] in matches}
    try:
        for g,player in con.execute('select distinct game_id,player_id from core.lineups').fetchall():
            if g in game_to_match:
                mid=game_to_match[g];player_times[player][mid]=matches[mid][0]
    finally:con.close()
    turnarounds=[]
    for entity,lookup in (('team',team_times),('player',player_times)):
        gaps=[];entities=set()
        for key,appearances in lookup.items():
            ordered=sorted(set(appearances.values()))
            for a,b in zip(ordered,ordered[1:]):
                gaps.append(b-a)
                if 7200<=b-a<21600:entities.add(key)
        turnarounds.append(dict(entity=entity,entities=len(lookup),entities_with_2_to_under_6_hour_interval=len(entities),consecutive_distinct_schedule_intervals=len(gaps),two_to_under_six_hours=sum(7200<=x<21600 for x in gaps),two_to_under_six_fraction=float(np.mean([(7200<=x<21600) for x in gaps])) if gaps else None,meaning='scheduled start gaps, not verified result-publication or turnaround duration; equal scheduled timestamps do not establish order'))
    result=dict(identity=identity,ties=records,calibration_choices=calibrators,removal_parent_absolute_losses=parents,turnarounds=turnarounds,observed_maps_per_series=dict(sorted(collections.Counter(len(v) for v in maps.values()).items())),series_length_scope='observed admitted nonglobal development maps grouped by Match ID; descriptive realized counts, not verified format or a predictor',tie_origin_limit='unrecorded rating state/zero-input mechanisms cannot be uniquely reconstructed from probabilities; blends remain unresolved',exploratory=True)
    out=OUTPUT/'audits';atomic(out/'results.json',result);atomic(PERSIST/'audits/results.json',result);atomic(out/'completion.json',dict(results=digest(out/'results.json')))


if __name__=='__main__':main(bind())
