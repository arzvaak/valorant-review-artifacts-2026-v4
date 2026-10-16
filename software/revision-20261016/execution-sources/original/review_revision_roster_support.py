"""Source-only roster availability replay; no rating or predictive fitting."""
import argparse
import collections
import heapq
from pathlib import Path
import shutil

import numpy as np

from review_revision_diagnostics import CORE,PHASES,TASKS,Inputs,read,save,sha
from review_revision_ratings import sources


def replay(meta,labels,lineups,admitted,delay):
    groups=collections.defaultdict(list)
    for uid,row in meta.items():groups[row['prediction_cutoff']].append(uid)
    rosters={};players=set();pending=[];saved={};support={}
    for cutoff,units in sorted(groups.items()):
        while pending and pending[0][0]<cutoff:
            when=pending[0][0];released=[]
            while pending and pending[0][0]==when:
                _,uid=heapq.heappop(pending)
                if uid in admitted and meta[uid]['region']!='global' and np.isfinite(labels[uid]):released.append(uid)
            appeared=collections.defaultdict(set)
            for uid in released:
                row=meta[uid];sides=saved[uid]
                if all(len(side)==5 for side in sides):
                    players.update((uid[0],p) for side in sides for p in side)
                if uid[0]=='map_winner':
                    for team,roster in lineups[uid[1]].items():appeared[team].add(tuple(sorted(roster)))
            for team,rr in appeared.items():rosters[team]=next(iter(rr)) if len(rr)==1 else ()
        for uid in units:
            row=meta[uid]
            sides=[tuple(sorted(lineups[uid[1]][t])) if uid[0]=='map_winner' else rosters.get(t,()) for t in (row['team_a_id'],row['team_b_id'])]
            saved[uid]=sides
            support[uid]=dict(fallback=not all(len(side)==5 for side in sides),
                historical_player_counts=[sum((uid[0],p) in players for p in side) for side in sides])
            heapq.heappush(pending,(row['anchor']+delay,uid))
    return support


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise RuntimeError('Preserve completed support audit')
    args.output.mkdir(parents=True)
    inputs=Inputs();meta,labels,teams,players,lineups=sources(inputs)
    cfg=read(CORE/'config.json');result=dict(code=sha(__file__),inputs=inputs.hashes,units=[])
    online=replay(meta,labels,lineups,set(meta),21600)
    for fold in cfg['folds']:
        selected={};admitted=set();evaluation={}
        for task in TASKS:
            for phase in PHASES:
                for k in read(CORE/'blocks'/fold['name']/'inputs'/task/(phase+'-keys.json')):
                    uid=task,k['game_id'];selected[uid]=dict(k,task=task)
                    if phase=='train':admitted.add(uid)
                    if phase=='evaluation' and k['region']!='global' and np.isfinite(labels[uid]):evaluation[uid]=k
        for hours in (2,6):
            frozen=replay(selected,labels,lineups,admitted,hours*3600)
            for task in TASKS:
                keys=[uid for uid in evaluation if uid[0]==task]
                for regime,inventory in ((f'frozen_{hours}h',frozen),('online_6h',online)):
                    if regime=='online_6h' and hours==2:continue
                    values=[inventory[u] for u in keys]
                    result['units'].append(dict(block=fold['name'],task=task,regime=regime,rows=len(values),
                        incomplete_proxy_fallbacks=sum(v['fallback'] for v in values),
                        full_roster_without_any_task_player_history=sum(not v['fallback'] and sum(v['historical_player_counts'])==0 for v in values)))
    save(args.output/'results.json',result);shutil.copy2(__file__,args.output/'code.py')
    save(args.output/'completion.json',dict(results=sha(args.output/'results.json')))
    print('ROSTER SUPPORT AUDIT COMPLETE',flush=True)


if __name__=='__main__':main()
