"""Source-bound later feedback extension; immutable parents stay untouched.

Prepare field-whitelisted performance observations and explicit event admission,
then execute reviewable derivatives of the original retained-state builders.
"""
import argparse
import collections
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import yaml
from selectolax.parser import HTMLParser

from review_revision_diagnostics import LATER, STORE, read, save, sha

REGIONAL_FLAGS={'3143':'au','3150':'us','3148':'us','3147':'jo','3142':'us','3140':'eu','3116':'fr','3107':'au','3125':'br','3110':'eu','3135':'us','3115':'eu','3141':'eu','3138':'br','3097':'au','3127':'us','3137':'vn','3120':'us'}
FIELDS={'rating2':'rating_display','acs':'ACS','kills':'kills','deaths':'deaths','assists':'assists','kd-diff':'kill_death_difference','kast':'KAST','adr':'ADR','hsp':'headshot_percentage_display','fb':'first_kills','fd':'first_deaths','fk-diff':'first_kill_first_death_difference'}
TABLE_FIELDS=list(FIELDS.values())


def number(node):
    token=' '.join(node.text(separator=' ',strip=True).split()) if node else ''
    if re.fullmatch(r'[+-]?\d+(?:\.\d+)?%?',token):return float(token.rstrip('%'))
    return None


def stats(panel, context, aliases):
    expected={p['player_id']:(p['team_id'],p['agent_entity_id']) for p in context['lineups']}
    source_teams=context['source_team_a_id'],context['source_team_b_id']
    tables=panel.css('.ovw-table');modern=bool(tables)
    if not tables:tables=panel.css('table.mod-overview')
    if len(tables)!=2:raise ValueError('missing_two_team_stat_tables')
    players={};observed=set()
    for team,table in zip(source_teams,tables):
        for row in table.css('.ovw-row' if modern else 'tbody tr'):
            link=row.css_first('a[href^="/player/"]')
            if not link:continue
            match=re.match(r'^/player/(\d+)(?:/|$)',link.attributes.get('href',''))
            if not match:raise ValueError('invalid_player_link')
            player=match[1]
            if player in observed or player not in expected or expected[player][0]!=team:raise ValueError('player_team_identity_conflict')
            observed.add(player)
            images=row.css('.ovw-agents img') if modern else row.css('td')[1].css('img')
            if len(images)!=1:raise ValueError('ambiguous_player_agent')
            alias=(images[0].attributes.get('title') or images[0].attributes.get('alt','')).lower()
            if aliases.get(alias)!=expected[player][1]:raise ValueError('agent_identity_conflict')
            cells=row.css('[data-col]') if modern else row.css('td')[2:]
            columns={n.attributes['data-col']:n for n in cells} if modern else dict(zip(FIELDS,cells))
            values={}
            for source,field in FIELDS.items():
                cell=columns.get(source)
                if cell is None:continue
                for side in ('both','t','ct'):
                    nodes=cell.css('.side.mod-'+side)
                    if len(nodes)>1:raise ValueError('ambiguous_stat_cell')
                    values[(side+'_' if side!='both' else '')+field]=number(nodes[0]) if nodes else None
            players[team,player]=values
    if observed!=set(expected):raise ValueError('incomplete_frozen_player_inventory')
    rounds=[]
    for column in panel.css('.vlr-rounds-row-col'):
        n=column.css_first('.rnd-num')
        if not n:continue
        token=n.text(strip=True)
        if not token.isdigit():raise ValueError('invalid_round_index')
        squares=column.css('.rnd-sq')
        winners=[i for i,square in enumerate(squares) if 'mod-win' in square.attributes.get('class','').split()]
        if len(squares)==2 and not winners and not column.attributes.get('title') and all(not square.css('img') and not square.text(strip=True) for square in squares):
            # VLR pads unplayed future round slots with structurally empty cells.
            # No winner is inferred; the retained played sequence must remain contiguous.
            continue
        if len(squares)!=2 or len(winners)!=1:raise ValueError('ambiguous_round_winner')
        winner=winners[0];classes=squares[winner].attributes.get('class','').split()
        side='t' if 'mod-t' in classes else 'ct' if 'mod-ct' in classes else None
        rounds.append((int(token),source_teams[winner],side))
    rounds.sort()
    valid_rounds=bool(rounds) and [r[0] for r in rounds]==list(range(1,len(rounds)+1))
    teamobs={}
    if valid_rounds:
        for team in source_teams:
            wins=[winner==team for n,winner,side in rounds]
            attack=[i for i,(n,winner,side) in enumerate(rounds) if side is not None and ((winner==team and side=='t') or (winner!=team and side=='ct'))]
            defence=[i for i,(n,winner,side) in enumerate(rounds) if side is not None and ((winner==team and side=='ct') or (winner!=team and side=='t'))]
            pistol=[n-1 for n,winner,side in rounds if n in (1,13)]
            own=dict(rounds=len(rounds),margin=2*sum(wins)-len(rounds),round_fraction=(sum(wins),len(rounds)),attack_fraction=(sum(wins[i] for i in attack),len(attack)),defence_fraction=(sum(wins[i] for i in defence),len(defence)),attack_rounds=len(attack),defence_rounds=len(defence),pistol_win_fraction=(sum(wins[i] for i in pistol),len(pistol)),pistol_opportunities=len(pistol))
            for won,label in [(True,'pistol_win_followup'),(False,'pistol_loss_followup')]:
                follow=[i+1 for i in pistol if i+1<len(wins) and bool(wins[i])==won]
                own[label]=(sum(wins[i] for i in follow),len(follow))
            teamobs[team]=own
    rates={'kills','deaths','assists','first_kills','first_deaths','multikill_2','multikill_3','multikill_4','multikill_5','clutch_1v1','clutch_1v2','clutch_1v3','clutch_1v4','clutch_1v5','plants','defuses'}
    for team in source_teams:
        own=[v for (t,p),v in players.items() if t==team]
        total=sum(v['kills'] for v in own) if all(v.get('kills') is not None for v in own) else None
        for v in own:
            if total and v.get('kills') is not None:
                v['kill_share']=v['kills']/total;v['kill_share_squared']=v['kill_share']**2
            fk,fd=v.get('first_kills'),v.get('first_deaths')
            v['opening_balance']=(fk-fd,len(rounds)) if valid_rounds and fk is not None and fd is not None else None
            for side,denom in [('t',teamobs.get(team,{}).get('attack_rounds')),('ct',teamobs.get(team,{}).get('defence_rounds'))]:
                for field in ('kills','deaths','assists','first_kills','first_deaths'):
                    value=v.get(side+'_'+field);v[side+'_'+field]=(value,denom) if value is not None and denom else None
            for field in rates:
                value=v.get(field);v[field]=(value,len(rounds)) if value is not None and valid_rounds else None
    return players,teamobs,valid_rounds


def prepare(args):
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    dependencies=('later_reference_state.py','later_residual_features.py','later_retained_inference.py','later_score.py')
    job=read(LATER/'job.json')
    identity=dict(code=sha(__file__),parent=sha(LATER/'job.json'),source_evidence=sha(args.docs/'later-header-region-evidence.json'),
        dependencies={name:sha(Path(__file__).parent/name) for name in dependencies},
        method=sha(job['method']),agent_registry=sha('configs/entities/vlr-agents-9e9ec894.yaml'),
        inputs={str(p):sha(p) for p in (LATER/'context/projection.json',LATER/'context/map-context.json',LATER/'context/series-context.json',LATER/'targets/frozen.json',LATER/'targets/keys.json',LATER/'targets/labels.npy',LATER/'forecasts/frozen.json')})
    if (root/'identity.json').exists() and read(root/'identity.json')!=identity:raise RuntimeError('Version changed: preserve prior attempt')
    save(root/'identity.json',identity)
    if (root/'code.py').exists():
        if sha(root/'code.py')!=identity['code']:raise RuntimeError('Saved generator changed')
    else:shutil.copy2(__file__,root/'code.py')
    evidence=read(args.docs/'later-header-region-evidence.json')['records']
    decisions=[]
    for row in evidence:
        if sha(row['raw_path'])!=row['raw_sha256'] or sha(row['metadata_path'])!=row['metadata_sha256']:raise RuntimeError('Event header evidence changed')
        eid=row['event_id'];expected=REGIONAL_FLAGS.get(eid)
        if expected and row['observed_region_flags']!=[expected]:raise RuntimeError('Explicit region decision conflicts with observed header')
        decisions.append(row|dict(feedback_allowed=expected is not None,decision='source_header_explicit_regional_scope' if expected else 'unresolved_global_scope_or_location_only',classification_scope='feedback admission only; no professional tier, circuit or geographic population reassignment'))
    policy=dict(schema='later-feedback-admission/1',decisions=decisions,protected_event_denylist=['2766'],eligible_event_ids=sorted(REGIONAL_FLAGS),source_bound=True,method='Keep all 23 events for scoring; allow feedback from 18 source-explicit regional events; five unresolved events remain score-only. Retain model weights, thresholds/calibrators, six-hour schedule proxy and equal-cutoff freezing.',source_evidence_sha256=identity['source_evidence'])
    save(root/'feedback-policy.json',policy)
    save(args.docs/'later-feedback-policy.json',policy)
    if not (root/'observations.json').exists():
        keys=read(LATER/'forecasts/keys.json');labels=np.load(LATER/'targets/labels.npy')
        lookup={tuple(k):float(y) for k,y in zip(keys,labels)}
        observations=[];quarantine=[];sources=[];body_cache={}
        aliases={r['observed_alias'].lower():r['agent_entity_id'] for r in yaml.safe_load(Path('configs/entities/vlr-agents-9e9ec894.yaml').read_text())['agents'] if r['review_state']=='reviewed'}
        roots=[STORE/'later-event-capture-20261011-v1/captures',STORE/'later-event-archive-lineup-recovery-20261011-v1/captures']
        for k in read(LATER/'context/map-context.json'):
            if k['event_id'] not in REGIONAL_FLAGS or not np.isfinite(lookup[k['task'],k['game_id']]):continue
            mid=k['match_id'];path=next((r/(mid+'.html') for r in reversed(roots) if (r/(mid+'.html')).exists()),None)
            if path is None:
                quarantine.append(dict(game_id=k['game_id'],reason='no_retained_overview_html'));continue
            if mid not in body_cache:
                receipt=read(path.with_name(mid+'-receipt.json'))
                if sha(path)!=receipt['sha256'] or receipt['match_id']!=mid or receipt['event_id']!=k['event_id']:raise RuntimeError('Performance source identity conflict')
                body_cache[mid]=HTMLParser(path.read_bytes())
                sources.append(dict(match_id=mid,path=str(path),sha256=receipt['sha256']))
            doc=body_cache[mid];panels=doc.css('.vm-stats-game[data-game-id="'+k['game_id']+'"]')
            try:
                if len(panels)!=1:raise ValueError('missing_exact_game_panel')
                players,teamobs,rounds=stats(panels[0],k,aliases)
                if rounds:
                    margin=teamobs[k['team_a_id']]['margin']
                    if not margin or int(margin>0)!=lookup[k['task'],k['game_id']]:raise ValueError('round_sequence_target_disagreement')
                observations.append(dict(game_id=k['game_id'],match_id=mid,event_id=k['event_id'],players=[dict(team_id=t,player_id=p,values=v,missing_fields=[f for f,value in v.items() if value is None],missing_reason='unavailable_or_unparsed_overview_cell') for (t,p),v in players.items()],teamobs=teamobs,valid_round_sequence=rounds,missing_economy_reason='equipment/buy/ECON sections not projected from retained overview; unavailable values remain missing'))
            except ValueError as error:
                quarantine.append(dict(game_id=k['game_id'],reason=str(error)))
        save(root/'observations.json',dict(records=observations,quarantine=quarantine,sources=sources,code=sha(__file__),policy=sha(root/'feedback-policy.json'),future_predictor_access='cached observations enter only via strictly prior released History.update; no current-target performance'))
    observations=read(root/'observations.json')
    if observations['code']!=identity['code'] or observations['policy']!=sha(root/'feedback-policy.json'):raise RuntimeError('Observation projection identity changed')
    for source in observations['sources']:
        if sha(source['path'])!=source['sha256']:raise RuntimeError('Retained overview capture changed')
    generate(args)
    print('PREPARED FEEDBACK',len(REGIONAL_FLAGS),'eligible events;',len(read(root/'observations.json')['records']),'map performance records',flush=True)


def patch_once(source,old,new):
    if source.count(old)!=1:raise RuntimeError('Derivative source anchor missing/ambiguous: '+old[:60])
    return source.replace(old,new,1)


def generate(args):
    root=args.output
    base=Path(__file__).parent
    marker=root/'generated-code.json'
    if marker.exists():
        frozen=read(marker)
        if frozen['generator']!=sha(__file__) or frozen['policy']!=sha(root/'feedback-policy.json') or frozen['observations']!=sha(root/'observations.json'):raise RuntimeError('Generated code inputs changed')
        for name,expected in frozen['files'].items():
            if sha(root/name)!=expected:raise RuntimeError('Preserved generated code changed')
        return
    if any((root/name).exists() for name in ('reference','features','forecasts','scoring')):raise RuntimeError('Existing stages lack a generated-code freeze')
    # Exact generated derivatives are saved for inspection and source-bound reuse.
    ref=(base/'later_reference_state.py').read_text()
    ref=patch_once(ref,'    method, projection = read(args.method), read(args.context / \'projection.json\')',f'''    method, projection = read(args.method), read(args.context / 'projection.json')
    feedback_policy = read({str(root/'feedback-policy.json')!r})
    allowed_events = set(feedback_policy['eligible_event_ids'])
    later_keys = read({str(LATER/'targets/keys.json')!r})
    later_labels = np.load({str(LATER/'targets/labels.npy')!r})
    later_refs = {{tuple(k): ('later_revision', i) for i,k in enumerate(later_keys)}}''')
    ref=patch_once(ref,'    state, pending, stored = State(lineups, maps), [], {}', '''    release_metadata = {**metadata, **later}
    refs.update(later_refs)
    arrays['later_revision'] = later_labels
    state, pending, stored = State(lineups, maps), [], {}''')
    ref=patch_once(ref,'state.release(when, batch, metadata, refs, arrays, stored)','state.release(when, batch, release_metadata, refs, arrays, stored)')
    ref=patch_once(ref,"            else:\n                stored[uid] = (x, slots, mp, elo)","            if not is_later or k['event_id'] in allowed_events:\n                stored[uid] = (x, slots, mp, elo)")
    ref=ref.replace("'new_outcome_feedback': False","'new_outcome_feedback': state.history_end is not None and state.history_end > endpoint + 21600, 'source_feedback_eligible': k['event_id'] in allowed_events")
    ref=ref.replace("'new_targets_or_performance_read': False, 'new_feedback': False","'new_targets_or_performance_read': True, 'new_feedback': True, 'feedback_policy_sha256': sha("+repr(str(root/'feedback-policy.json'))+")")
    ref=ref.replace("    print(json.dumps(receipt, indent=2))","    print('Reference feedback features frozen', len(order), flush=True)")
    (root/'reference_feedback.py').write_text(ref)
    features=(base/'later_residual_features.py').read_text()
    injection=f'''    feedback_policy = read({str(root/'feedback-policy.json')!r})
    allowed_events = set(feedback_policy['eligible_event_ids'])
    new_observations = read({str(root/'observations.json')!r})
    def restored(values):
        return {{field: tuple(value) if isinstance(value, list) and len(value)==2 else value for field,value in values.items()}}
    for new_record in new_observations['records']:
        for player in new_record['players']:
            observed[new_record['game_id'],player['team_id'],player['player_id']] = restored(player['values'])
        for team,values in new_record['teamobs'].items():
            teamobs[new_record['game_id'],team] = restored(values)
    new_keys = read({str(LATER/'targets/keys.json')!r})
    new_labels = np.load({str(LATER/'targets/labels.npy')!r})
    new_label_lookup = {{tuple(k): float(y) for k,y in zip(new_keys,new_labels)}}
'''
    features=patch_once(features,"    print('OLD_PERFORMANCE_LOADED', flush=True)","    print('OLD_PERFORMANCE_LOADED', flush=True)\n"+injection)
    features=patch_once(features,'    reference_keys = [tuple(k) for k in read(args.reference / \'keys.json\')]', '''    for uid, context_row in contexts.items():
        if uid[0]=='map_winner' and context_row['event_id'] in allowed_events and np.isfinite(new_label_lookup[uid]):
            labels[uid[1]] = new_label_lookup[uid]
            pending.append(context_row)
    pending.sort(key=lambda r:r['anchor'])
    reference_keys = [tuple(k) for k in read(args.reference / 'keys.json')]''')
    features=features.replace("'new_feedback': False","'new_feedback': True")
    features=features.replace("'new_targets_or_performance_read': False","'new_targets_or_performance_read': True, 'feedback_policy_sha256': sha("+repr(str(root/'feedback-policy.json'))+"), 'new_observations_sha256': sha("+repr(str(root/'observations.json'))+")")
    features=features.replace('    print(json.dumps(receipt, indent=2))',"    print('Residual feedback features frozen', len(order), flush=True)")
    (root/'residual_feedback.py').write_text(features)
    for file in ('later_reference_state.py','later_retained_inference.py','later_score.py'):
        shutil.copy2(base/file,root/file)
    inference=(root/'later_retained_inference.py').read_text()
    inference=inference.replace("'predictions_frozen_before_new_targets' if complete else 'partial_predictions_frozen_before_new_targets'", "'predictions_frozen_under_declared_feedback' if complete else 'partial_predictions_under_declared_feedback'")
    inference=inference.replace("'new_feedback': False", "'new_feedback': True, 'feedback_policy_sha256': sha("+repr(str(root/'feedback-policy.json'))+"), 'upstream_cohort_labels_used_for_prior_feedback': True")
    (root/'later_retained_inference.py').write_text(inference)
    scorer=(root/'later_score.py').read_text()
    scorer=patch_once(scorer,"    assert targets['forecast_freeze_sha256'] == sha(args.forecasts / 'frozen.json')",f"    assert targets['forecast_freeze_sha256'] == sha({str(LATER/'forecasts/frozen.json')!r})\n    assert forecasts['feedback_policy_sha256'] == sha({str(root/'feedback-policy.json')!r})")
    scorer=scorer.replace("'new_feedback': False", "'new_feedback': True, 'original_target_freeze_sha256': sha("+repr(str(LATER/'targets/frozen.json'))+")")
    scorer=scorer.replace("'source_policy': 'unreviewed cohort is score-only'", "'source_policy': '18 source-header regional events admitted for feedback; five unresolved events remain score-only; observed overview fields refreshed, unavailable economy fields retained missing'")
    (root/'later_score.py').write_text(scorer)
    save(root/'generated-code.json',dict(generator=sha(__file__),files={p.name:sha(p) for p in root.glob('*.py')},policy=sha(root/'feedback-policy.json'),observations=sha(root/'observations.json')))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--docs',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for retained inference; no silent CPU fallback')
    prepare(args)
    if args.prepare_only:return
    job=read(LATER/'job.json');method=job['method']
    stages=[('reference_feedback.py','reference',['--context',str(LATER/'context'),'--method',method]),
            ('residual_feedback.py','features',['--context',str(LATER/'context'),'--reference',str(args.output/'reference'),'--method',method]),
            ('later_retained_inference.py','forecasts',['--features',str(args.output/'features'),'--reference',str(args.output/'reference'),'--method',method]),
            ('later_score.py','scoring',['--context',str(LATER/'context'),'--forecasts',str(args.output/'forecasts'),'--targets',str(LATER/'targets'),'--method',method])]
    for script,output,arguments in stages:
        marker=args.output/output/('completion.json' if output=='scoring' else 'frozen.json')
        if marker.exists():
            receipt=read(marker)
            if receipt['code_sha256']!=sha(args.output/script) or receipt['method_sha256']!=sha(method):raise RuntimeError('Completed stage code/method changed')
            for name,expected in receipt.get('files',{}).items():
                if sha(args.output/output/name)!=expected:raise RuntimeError('Completed stage file changed')
            expected_upstream={'context_sha256':LATER/'context/projection.json','context_projection_sha256':LATER/'context/projection.json','reference_sha256':args.output/'reference/frozen.json','features_sha256':args.output/'features/frozen.json','forecasts_sha256':args.output/'forecasts/frozen.json','targets_sha256':LATER/'targets/frozen.json','feedback_policy_sha256':args.output/'feedback-policy.json','new_observations_sha256':args.output/'observations.json'}
            for field,path in expected_upstream.items():
                if field in receipt and receipt[field]!=sha(path):raise RuntimeError('Completed stage upstream changed: '+field)
            if output=='scoring' and sha(args.output/output/'results.json')!=receipt['results_sha256']:raise RuntimeError('Completed scores changed')
            continue
        log=args.output/(output+'.log')
        with log.open('a') as stream:
            subprocess.run([sys.executable,str(args.output/script),*arguments,'--output',str(args.output/output)],stdout=stream,stderr=subprocess.STDOUT,check=True)
        print('COMPLETED FEEDBACK STAGE',output,flush=True)


if __name__=='__main__':main()
