"""Real source-bound history replay inside Colab; never reads raw score HTML."""
import collections
import copy
import math
from pathlib import Path
import duckdb
import numpy as np
from context import CORE, INPUT, REGISTRY
import legacy_prepare as old
from common import PHASES, TASKS, read, save_json, save_npz, files
from player_snapshot import preferred


def bound_rounds(source, rows, labelmap, fresh=False):
    """Reuse frozen targets and validate admitted round identities/counts."""
    plan=read(REGISTRY/'target-plan.json')
    sequences=collections.defaultdict(list)
    ids=[r['game_id'] for r in rows]
    for g,n,t,h in source.execute('select game_id,round_number,winner_team_id,raw_sha256 from main.map_rounds where game_id in (select unnest(?))',[ids]).fetchall():
        sequences[g].append((n,t,h))
    valid={};reasons={}
    for r in rows:
        g=r['game_id'];sequence=sequences[g];n=len(sequence)
        bound={z['parent_sha256'] for z in plan[g]}
        if not n or any(not isinstance(z[0],int) for z in sequence) or sorted(z[0] for z in sequence)!=list(range(1,n+1)) or any(t not in (r['team_a_id'],r['team_b_id']) or h not in bound for _,t,h in sequence):
            reasons[g]='missing_or_invalid_source_bound_round_sequence';continue
        wins=sum(t==r['team_a_id'] for _,t,_ in sequence)
        if wins*2==n or g not in labelmap or (not fresh and int(wins*2>n)!=labelmap[g]):
            reasons[g]='source_round_frozen_target_disagreement';continue
        valid[g]=(float(2*wins-n),wins/n,n)
    return valid,reasons


class ReleasedLabels:
    """Mmap target references; scalar access fails before its release cutoff."""
    def __init__(self, original, delay):
        self.refs={};self.arrays=[];self.cutoff=float('-inf');self.delay=delay
        for phase in PHASES:
            keys=read(original/'inputs/map_winner'/f'{phase}-keys.json')
            path=original/'inputs/map_winner'/f'{phase}-labels.npy' if phase!='evaluation' else INPUT/'labels'/original.name/'map_winner.npy'
            values=np.load(path,allow_pickle=False,mmap_mode='r')
            if len(values)!=len(keys):raise RuntimeError('Feedback target alignment')
            self.arrays.append(values)
            for i,k in enumerate(keys):self.refs[k['game_id']]=(values,i,k['anchor'])
    def __contains__(self,key):return key in self.refs
    def released(self,key):
        values,i,anchor=self.refs[key]
        if anchor+self.delay>=self.cutoff:raise RuntimeError('Attempted unreleased feedback label access')
        value=float(values[i])
        return value if math.isfinite(value) else None
    def __getitem__(self,key):
        value=self.released(key)
        if value is None:raise RuntimeError('Unavailable outcome reached historical update')
        return value


def phase_keys(original, delay, lead):
    keys = {t: {p: read(original/'inputs'/t/f'{p}-keys.json') for p in PHASES} for t in TASKS}
    exclusions = []
    # Both tasks share event allocation. Use map rows to define full extents.
    for i, p in enumerate(PHASES[:-1]):
        later = next((q for q in PHASES[i+1:] if keys['map_winner'][q]), None)
        if later is None:
            continue
        boundary = min(k['anchor']-lead for k in keys['map_winner'][later])
        bad = {k['event_id'] for k in keys['map_winner'][p] if k['event_completion']+delay >= boundary}
        for event in sorted(bad):
            exclusions.append(dict(phase=p,event_id=event,next_phase=later,reason='event_release_not_before_next_phase'))
        for t in TASKS:
            keys[t][p] = [k for k in keys[t][p] if k['event_id'] not in bad]
    for t in TASKS:
        for p in PHASES:
            for k in keys[t][p]:
                k['prediction_cutoff'] = k['anchor']-lead
                k['source_availability'] = f'schedule_plus_{delay}_seconds_proxy'
    return keys, exclusions


def prepare(dest, original, delay, lead, identity, fresh=False):
    marker = dest/'inputs/prepared.json'
    if marker.exists():
        receipt = read(marker)
        if receipt['sensitivity_identity'] != identity:
            raise RuntimeError('Changed input identity; preserve and version repair')
        return receipt
    if (dest/'inputs').exists():
        raise RuntimeError('Unfinished source preparation preserved; version repair required')
    cfg = read(CORE/'config.json')
    cfg['source_database_path'] = str(INPUT/'source.duckdb')
    cfg['database_path'] = str(INPUT/'warehouse.duckdb')
    source_meta = read(original/'inputs/prepared.json')
    # The original target binding is retained; only chronological admission changes.
    if fresh:
        # The fresh-history arm changes only the historical update policy;
        # stricter-clock event purges belong to the separate clock arms.
        keys={t:{p:read(original/'inputs'/t/f'{p}-keys.json') for p in PHASES} for t in TASKS}
        exclusions=[]
        if any(k['prediction_cutoff']!=k['anchor']-lead for t in TASKS for p in PHASES for k in keys[t][p]):
            raise RuntimeError('Fresh policy must retain the original cutoffs')
    else:
        keys, exclusions = phase_keys(original,delay,lead)
    registry = read(REGISTRY/'maps.json')
    byid = {k['game_id']: k for k in registry}
    trainkeys = keys['map_winner']['train']
    original_train = read(original/'inputs/map_winner/train-keys.json')
    yy = np.load(original/'inputs/map_winner/train-labels.npy',allow_pickle=False)
    labelmap = {k['game_id']:float(y) for k,y in zip(original_train,yy)}
    rows = sorted([byid[k['game_id']] for k in trainkeys],key=lambda k:(k['anchor'],k['game_id']))
    if fresh:
        labelmap=ReleasedLabels(original,delay)
        admitted={k['game_id'] for phase in PHASES for k in keys['map_winner'][phase] if k['region']!='global'}
        rows=sorted([byid[g] for g in admitted if g in labelmap],key=lambda k:(k['anchor'],k['game_id']))
    ids = {k['game_id'] for p in PHASES for k in keys['map_winner'][p]}
    source = duckdb.connect(cfg['source_database_path'],read_only=True)
    warehouse = duckdb.connect(cfg['database_path'],read_only=True)
    source.execute('set threads=4'); warehouse.execute('set threads=4')
    dest.joinpath('inputs').mkdir(parents=True)
    try:
        lineups=collections.defaultdict(list)
        for g,t,p,a in source.execute('select game_id,team_id,player_id,agent_alias from core.lineups where game_id in (select unnest(?))',[sorted(ids)]).fetchall():
            lineups[g].append((t,p,a))
        vocabrows=[byid[k['game_id']] for k in trainkeys]
        agents=sorted({a.lower() for r in vocabrows for _,_,a in lineups[r['game_id']]})
        maps=sorted({r['map_entity_id'] for r in vocabrows})
        # Bound transient duplicate-cell SQL materialization on standard Colab RAM.
        # The immutable numerical loader is unchanged; chunks never split an
        # equal-anchor update batch and their inert lookups never update state.
        loaded_until=0;rounds={};observed={};teamobs={};duels={};reasons={};missing=[]
        history=old.History(agents,maps)
        history.player_agent=collections.defaultdict(lambda:collections.defaultdict(old.Mean))
        arrays={}; stream=[]; meta=copy.deepcopy(source_meta['tasks']); audit=[]; coverage=[]; feedback_exclusions=[]; applied_history_rows=0
        for t in TASKS:
            m=meta[t]; half=len(agents)+len(maps)+2
            m.update(composition_width=2*half,swap_composition_indices=list(range(half,2*half))+list(range(half)),swap_composition_signs=[1]*(2*half))
            for p in PHASES:
                arrays[t,p]=dict(numeric=np.empty((len(keys[t][p]),len(m['feature_names'])),np.float32),composition=np.empty((len(keys[t][p]),2*half),np.float32))
                stream.extend((k['prediction_cutoff'],t,p,i,k) for i,k in enumerate(keys[t][p]))
        cursor=0
        for ordinal,(cutoff,t,p,i,k) in enumerate(sorted(stream,key=lambda z:(z[0],z[4]['game_id'],z[1]))):
            while cursor<len(rows) and rows[cursor]['anchor']+delay<cutoff:
                if cursor>=loaded_until:
                    loaded_until=min(len(rows),cursor+512)
                    while loaded_until<len(rows) and rows[loaded_until]['anchor']==rows[loaded_until-1]['anchor']:loaded_until+=1
                    chunk=rows[cursor:loaded_until]
                    rounds,chunk_reasons=bound_rounds(source,chunk,labelmap,fresh=fresh)
                    reasons.update(chunk_reasons)
                    observed,teamobs,duels,chunk_missing=old.load_observations(warehouse,source,chunk,lineups,rounds)
                    missing.extend(chunk_missing)
                end=cursor+1
                while end<len(rows) and rows[end]['anchor']==rows[cursor]['anchor']:end+=1
                batch=rows[cursor:end]
                if any(r['match_id']==k['match_id'] for r in batch):raise RuntimeError('Same-Match history')
                if fresh:
                    labelmap.cutoff=cutoff
                    eligible=[]
                    for r in batch:
                        g=r['game_id']
                        outcome=labelmap.released(g)
                        if outcome is None:
                            feedback_exclusions.append(dict(game_id=g,match_id=r['match_id'],event_id=r['event_id'],release_timestamp=r['anchor']+delay,first_eligible_cutoff=cutoff,reason='original_frozen_outcome_unavailable_no_feedback_update'))
                            continue
                        if g in rounds and int(rounds[g][0]>0)!=outcome:
                            raise RuntimeError('Released round/frozen-target disagreement')
                        eligible.append(r)
                    batch=eligible
                if batch:history.update(batch,lineups,observed,teamobs,duels,labelmap,delay)
                for r in batch:
                    for bt,bp,ba in lineups[r['game_id']]:history.player_agent[bp][ba.lower()].add(observed.get((r['game_id'],bt,bp),{}))
                applied_history_rows+=len(batch)
                cursor=end
            if history.last is not None and history.last>=cutoff:raise RuntimeError('History cutoff')
            series=t=='series_winner'
            raw,comp,cols=history.features(k,lineups,cutoff,series)
            values=dict(zip(cols.names,raw)); slots=[]; linked=[]
            for side,team in enumerate((k['team_a_id'],k['team_b_id'])):
                if series:
                    roster,_=history.rosters.get(team,(None,None)); ss=[(z,None) for z in sorted(roster)] if roster else []
                else:ss=sorted((z,a.lower()) for tm,z,a in lineups[k['game_id']] if tm==team)
                slots.append(ss); ls=[]; rr=[]
                for player,agent in ss:
                    if series:agent,state=preferred(history,player)
                    else:state=history.player_context[player,k['map_entity_id'],agent]
                    ls.append(state)
                    recent=old.Mean()
                    for timestamp,mp,ag,v in history.recent[player]:
                        if cutoff-90*86400<timestamp<cutoff and ag==agent and (series or mp==k['map_entity_id']):recent.add(v)
                    rr.append(recent)
                linked.append(ls)
                if series:
                    suffix=('a','b')[side]
                    for f in ('first_kills','first_deaths','opening_balance'):values['matched_'+f+'_'+suffix]=old.finite_mean([s.value(f) for s in ls])
                    values['matched_player_map_agent_maps_'+suffix]=old.finite_mean([s.maps for s in ls])
                    for f in ('ACS','ADR','KAST','opening_balance'):values['recent90_matched_'+f+'_'+suffix]=old.finite_mean([s.value(f) for s in rr])
            m=meta[t]; vector=[]
            for name,group,family in zip(m['feature_names'],m['groups'],m['semantic']):
                if '_coverage_' in name and name not in values:
                    suffix=name[-1]; ns=[n for n,g,f in zip(m['feature_names'],m['groups'],m['semantic']) if g==group and f==family and n.endswith('_'+suffix) and '_coverage_' not in n]
                    vector.append(float(np.mean([math.isfinite(values[n]) for n in ns])) if ns else 0.)
                else:vector.append(values[name])
            arrays[t,p]['numeric'][i]=vector; arrays[t,p]['composition'][i]=comp
            k['history_end_timestamp']=history.last
            team=[history.team[tm].maps>0 for tm in (k['team_a_id'],k['team_b_id'])]
            player=[sum(history.player[z].maps>0 for z,_ in ss) for ss in slots]
            agent=[sum(s.maps>0 for s in ss) for ss in linked]
            coverage.append(dict(task=t,phase=p,game_id=k['game_id'],both_team_history=all(team),either_team_history=any(team),both_roster=len(slots[0])==len(slots[1])==5,both_full_player=min(player)==5,either_player=max(player)>0,both_full_linked=min(agent)==5,either_linked=max(agent)>0))
            audit.append(dict(task=t,phase=p,game_id=k['game_id'],history_end_timestamp=history.last,prediction_cutoff=cutoff,training_only=not fresh))
            if ordinal%10000==0:print('AVAILABILITY HISTORY',dest.name,ordinal,len(stream),flush=True)
        for t in TASKS:
            for p in PHASES:
                out=dest/'inputs'/t
                save_npz(out/f'{p}.npz',**arrays[t,p]);save_json(out/f'{p}-keys.json',keys[t][p])
                if p!='evaluation':
                    original_keys=read(original/'inputs'/t/f'{p}-keys.json'); labels=np.load(original/'inputs'/t/f'{p}-labels.npy',allow_pickle=False)
                    lookup={k['game_id']:y for k,y in zip(original_keys,labels)}
                    np.save(out/f'{p}-labels.npy',np.asarray([lookup[k['game_id']] for k in keys[t][p]],np.float32))
        save_json(dest/'inputs/history-audit.json',audit);save_json(dest/'inputs/coverage.json',coverage)
        save_json(dest/'inputs/source-missingness.json',dict(round_reasons=reasons,numeric_conflicts=missing))
        save_json(dest/'inputs/feedback-exclusions.json',feedback_exclusions)
        counts={t:{p:dict(rows=len(keys[t][p]),events=len({k['event_id'] for k in keys[t][p]})) for p in PHASES} for t in TASKS}
        receipt=dict(tasks=meta,files=files(dest/'inputs'),sensitivity_identity=identity,counts=counts,exclusions=exclusions,delay_seconds=delay,lead_seconds=lead,evaluation_scoring_labels_read=False,eligible_prior_feedback_permitted=fresh,source_history_ids=len(rows),applied_history_rows=applied_history_rows,unavailable_feedback_rows=len(feedback_exclusions),training_only=not fresh,agent_vocabulary=agents,map_vocabulary=maps)
        save_json(marker,receipt);return receipt
    finally:source.close();warehouse.close()

