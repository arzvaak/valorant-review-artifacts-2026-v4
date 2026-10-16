"""Repair six omitted historical side-headshot tensor channels, training only.

All other original arrays are read-only links. No targets or performance from
held-out maps enter the correction; map lineups are admissible horizon inputs.
"""
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import shutil

import duckdb
import numpy as np

from review_revision_diagnostics import CORE, PHASES, TASKS, Inputs, read, save, sha


def prepare(block, output):
    original = CORE/'blocks'/block
    inputs = Inputs()
    prepared = inputs.json(original/'inputs/prepared.json')
    keys = {t: {p: inputs.json(original/'inputs'/t/(p+'-keys.json'), prepared['files'][t+'/'+p+'-keys.json']) for p in PHASES} for t in TASKS}
    train = keys['map_winner']['train']
    labels = np.load(inputs.bind(original/'inputs/map_winner/train-labels.npy', prepared['files']['map_winner/train-labels.npy']))
    if not np.isfinite(labels).all():
        raise RuntimeError('Repair requires the exact valid admitted training inventory')
    cfg = read(CORE/'config.json')
    train_ids = [k['game_id'] for k in train]
    map_ids = sorted({k['game_id'] for phase in PHASES for k in keys['map_winner'][phase]})
    connection = duckdb.connect(cfg['source_database_path'], read_only=True)
    try:
        rows = connection.execute('select game_id,team_id,player_id,agent_alias from core.lineups where game_id in (select unnest(?))', [map_ids]).fetchall()
    finally:
        connection.close()
    lineups = collections.defaultdict(lambda: collections.defaultdict(list))
    for g,t,p,a in rows:
        lineups[str(g)][str(t)].append((str(p),a.lower()))
    connection = duckdb.connect(cfg['database_path'], read_only=True)
    try:
        values = connection.execute("select game_id,team_id,player_id,side,value from corrected.player_side_statistics where game_id in (select unnest(?)) and field='headshot_percentage_display'", [train_ids]).fetchall()
    finally:
        connection.close()
    cells = collections.defaultdict(set)
    for g,t,p,side,value in values:
        if side not in ('t','ct'):
            raise RuntimeError('Unrecognized side')
        if value is not None and np.isfinite(value):
            cells[str(g),str(t),str(p),side].add(float(value))
    unique = {k: next(iter(v)) if len(v)==1 else np.nan for k,v in cells.items()}
    conflicts = sum(len(v)>1 for v in cells.values())
    sums = collections.defaultdict(lambda: np.zeros(2))
    counts = collections.defaultdict(lambda: np.zeros(2))
    recent = collections.defaultdict(list)
    usage = collections.defaultdict(collections.Counter)
    rosters = {}
    arrays = {}
    field_columns = {}
    for task in TASKS:
        meta = prepared['tasks'][task]
        fields = meta['player_fields']
        field_columns[task] = [fields.index(side+'_headshot_percentage_display') for side in ('t','ct')]
        for phase in PHASES:
            file = task+'/'+phase+'-players.npz'
            with np.load(inputs.bind(original/'inputs'/file, prepared['files'][file])) as z:
                arrays[task,phase] = dict(players=z['players'].copy(), agents=z['agents'].copy())
    stream = sorted((k['prediction_cutoff'],task,phase,i,k) for task in TASKS for phase in PHASES for i,k in enumerate(keys[task][phase]))
    train = sorted(train, key=lambda k:(k['anchor'], k['game_id']))
    cursor = 0
    nonmissing = collections.Counter()
    for cutoff,task,phase,index,k in stream:
        while cursor<len(train) and train[cursor]['anchor']+7200<cutoff:
            end = cursor+1
            while end<len(train) and train[end]['anchor']==train[cursor]['anchor']:
                end += 1
            appeared = collections.defaultdict(set)
            for row in train[cursor:end]:
                for team in (row['team_a_id'],row['team_b_id']):
                    own = lineups[row['game_id']][team]
                    if len(own)!=5 or len({p for p,a in own})!=5:
                        raise RuntimeError('Invalid admitted training lineup')
                    appeared[team].add(tuple(sorted(p for p,a in own)))
                    for player,agent in own:
                        vector = np.array([unique.get((row['game_id'],team,player,side),np.nan) for side in ('t','ct')])
                        finite = np.isfinite(vector)
                        for key in [('player',player),('context',player,row['map_entity_id'],agent),('agent',player,agent)]:
                            sums[key] += np.nan_to_num(vector)
                            counts[key] += finite
                        usage[player][agent] += 1
                        recent[player].append((row['anchor']+7200,vector))
            for team, observed in appeared.items():
                rosters[team] = next(iter(observed)) if len(observed)==1 else ()
            cursor = end
        latest = train[cursor-1]['anchor']+7200 if cursor else None
        if latest != k['history_end_timestamp']:
            raise RuntimeError('Repair history differs from original admissible history')
        fields = prepared['tasks'][task]['player_fields']
        for side,team in enumerate((k['team_a_id'],k['team_b_id'])):
            slots = sorted(lineups[k['game_id']][team]) if task=='map_winner' else [(p,None) for p in rosters.get(team,())]
            for slot,(player,agent) in enumerate(slots):
                if task=='series_winner':
                    agent = min(usage[player],key=lambda a:(-usage[player][a],a)) if usage[player] else None
                def mean(key):
                    return np.divide(sums[key],counts[key],out=np.full(2,np.nan),where=counts[key]>0)
                general = mean(('player',player))
                prior = [v for when,v in recent[player] if cutoff-90*86400<when<cutoff]
                rs = np.nansum(prior,axis=0) if prior else np.zeros(2)
                rn = np.isfinite(prior).sum(0) if prior else np.zeros(2)
                recent_mean = np.divide(rs,rn,out=np.full(2,np.nan),where=rn>0)
                linked = mean(('context',player,k['map_entity_id'],agent)) if task=='map_winner' else mean(('agent',player,agent))
                for channel,vector in enumerate((general,recent_mean,linked)):
                    columns = [channel*len(fields)+j for j in field_columns[task]]
                    arrays[task,phase]['players'][index,side,slot,columns] = vector
                    nonmissing[task,phase,channel] += int(np.isfinite(vector).sum())
    output.mkdir(parents=True,exist_ok=False)
    for source in (original/'inputs').rglob('*'):
        if not source.is_file():
            continue
        relative = source.relative_to(original/'inputs')
        target = output/'inputs'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        if source.name!='prepared.json' and not source.name.endswith('-players.npz'):
            target.symlink_to(source)
    revised = copy.deepcopy(prepared)
    for (task,phase),data in arrays.items():
        file = task+'/'+phase+'-players.npz'
        np.savez_compressed(output/'inputs'/file,**data)
        revised['files'][file] = sha(output/'inputs'/file)
    revised['revision'] = dict(code=sha(__file__),original_prepared=sha(original/'inputs/prepared.json'), corrected='six historical attack/defence headshot channels only')
    save(output/'inputs/prepared.json',revised)
    save(output/'correction.json',dict(code=sha(__file__), inputs=inputs.hashes, source_database_expected=cfg['source_database_sha256'], warehouse_expected=cfg['database_sha256'],
                                      headshot_projection_sha256=hashlib.sha256(json.dumps(sorted((list(k),v) for k,v in unique.items() if np.isfinite(v))).encode()).hexdigest(),
                                      conflicting_cells=conflicts,nonmissing={str(k):v for k,v in nonmissing.items()}, training_maps=len(train), labels_or_performance_from_evaluation=False))
    print('CORRECTED HISTORICAL HEADSHOTS',block,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    identity=dict(code=sha(__file__),core=sha(CORE/'release-identity.json'))
    if (args.output/'identity.json').exists() and read(args.output/'identity.json')!=identity:
        raise RuntimeError('Changed correction code/input requires new version')
    save(args.output/'identity.json',identity)
    shutil.copy2(__file__,args.output/'projection-code.py')
    for fold in read(CORE/'config.json')['folds']:
        output=args.output/fold['name']
        if (output/'correction.json').exists():
            continue
        if output.exists():
            raise RuntimeError('Interrupted correction retained; use a new version')
        prepare(fold['name'],output)
    save(args.output/'completion.json',dict(identity=identity,blocks=12))


if __name__=='__main__':
    main()
