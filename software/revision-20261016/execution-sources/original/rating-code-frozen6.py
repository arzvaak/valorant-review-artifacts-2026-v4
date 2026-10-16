"""CUDA team Glicko-2 / Gaussian player-team ratings on admitted snapshots.

This is real fitting, with explicit chronological feedback, no CPU fallback,
no current-contest player statistics and no synthetic fitting campaign.
"""
import argparse
import collections
import heapq
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import time

import duckdb
import numpy as np
import torch

from review_revision_diagnostics import CORE, DYNAMIC, PHASES, TASKS, Inputs, read, save, sha


class Ratings:
    def __init__(self, team_ids, player_ids, lineups):
        self.teams, self.players, self.lineups = team_ids, player_ids, lineups
        n, m = 2*len(team_ids), 2*len(player_ids)
        self.mu = torch.zeros(n, device='cuda', dtype=torch.float64)
        self.phi = torch.full_like(self.mu, 350/173.7178)
        self.sigma = torch.full_like(self.mu, .06)
        self.last = torch.full_like(self.mu, float('nan'))
        self.elo = torch.zeros_like(self.mu)
        self.pm = torch.full((m,), 25., device='cuda', dtype=torch.float64)
        self.pv = torch.full_like(self.pm, (25/3)**2)
        self.plast = torch.full_like(self.pm, float('nan'))
        self.rosters = {}
        self.update_count = 0

    def team_index(self, task, team):
        return task*len(self.teams)+self.teams[team]

    def player_index(self, task, player):
        return task*len(self.players)+self.players[player]

    def indices(self, batch):
        teams, players, complete = [], [], []
        for row in batch:
            task = TASKS.index(row['task'])
            sides = []
            for team in (row['team_a_id'], row['team_b_id']):
                roster = self.lineups[row['game_id']][team] if row['task']=='map_winner' else self.rosters.get(team, ())
                sides.append([self.player_index(task, p) for p in sorted(roster)])
            full = all(len(side)==5 for side in sides)
            complete.append(full)
            players.append(sides if full else [[0]*5, [0]*5])
            teams.append([self.team_index(task, row['team_a_id']), self.team_index(task, row['team_b_id'])])
        return torch.tensor(teams, device='cuda'), torch.tensor(players, device='cuda'), torch.tensor(complete, device='cuda')

    def deviation(self, ix, when, before_update=False):
        elapsed = torch.nan_to_num((when-self.last[ix])/86400, nan=0.).clamp_min(0)
        if before_update:
            elapsed = (elapsed-1).clamp_min(0)
        return (self.phi[ix]**2+self.sigma[ix]**2*elapsed).sqrt()

    def player_variance(self, ix, when):
        elapsed = torch.nan_to_num((when-self.plast[ix])/86400, nan=0.).clamp_min(0)
        return self.pv[ix]+(25/300)**2*elapsed

    def predict(self, batch, when):
        teams, players, complete = self.indices(batch)
        deviation = self.deviation(teams, when)
        g = (1+3*deviation**2/math.pi**2).rsqrt()
        difference = self.mu[teams[:, 0]]-self.mu[teams[:, 1]]
        gl = .5*((g[:, 1]*difference).sigmoid()+(g[:, 0]*difference).sigmoid())
        ep = ((self.elo[teams[:, 0]]-self.elo[teams[:, 1]])*math.log(10)/400).sigmoid()
        variance = self.player_variance(players, when)
        c = (variance.sum((1, 2))+10*(25/6)**2).sqrt()
        difference = self.pm[players[:, 0]].sum(1)-self.pm[players[:, 1]].sum(1)
        ts = torch.special.ndtr(difference/c)
        ts = torch.where(complete, ts, gl)
        return torch.stack((ep, gl, ts), 1).cpu().numpy(), players.cpu().numpy(), complete.cpu().numpy()

    def update(self, batch, labels, stored, when):
        teams, _, _ = self.indices(batch)
        target = torch.tensor(labels, device='cuda', dtype=torch.float64)
        flattened = teams.flatten()
        active = torch.unique(flattened)
        phi = self.deviation(active, when, before_update=True)
        local_phi = self.deviation(teams, when, before_update=True)
        g = (1+3*local_phi.flip(1)**2/math.pi**2).rsqrt()
        difference = self.mu[teams]-self.mu[teams.flip(1)]
        expectations = (g*difference).sigmoid()
        outcomes = torch.stack((target, 1-target), 1)
        hessian = torch.zeros_like(self.mu)
        gradient = torch.zeros_like(self.mu)
        hessian.index_add_(0, flattened, (g*g*expectations*(1-expectations)).flatten())
        gradient.index_add_(0, flattened, (g*(outcomes-expectations)).flatten())
        v = hessian[active].clamp_min(1e-14).reciprocal()
        delta = v*gradient[active]
        original = (self.sigma[active]**2).log()
        def f(x):
            ex = x.exp()
            return ex*(delta**2-phi**2-v-ex)/(2*(phi**2+v+ex)**2)-(x-original)/.5**2
        a = original.clone()
        condition = delta**2 > phi**2+v
        b = torch.where(condition, (delta**2-phi**2-v).clamp_min(1e-100).log(), original-.5)
        for _ in range(1000):
            missing = (~condition) & (f(b)<0)
            if not bool(missing.any()):
                break
            b = torch.where(missing, b-.5, b)
        else:
            raise RuntimeError('Glicko-2 volatility root could not be bracketed')
        fa, fb = f(a), f(b)
        for _ in range(100):
            running = (b-a).abs()>1e-6
            if not bool(running.any()):
                break
            denominator = fb-fa
            if bool((running & (denominator.abs()<1e-30)).any()):
                raise RuntimeError('Unresolved Glicko-2 volatility denominator')
            c = a+(a-b)*fa/torch.where(running, denominator, torch.ones_like(denominator))
            fc = f(c)
            cross = fc*fb<=0
            a = torch.where(running & cross, b, a)
            fa = torch.where(running, torch.where(cross, fb, fa/2), fa)
            b = torch.where(running, c, b)
            fb = torch.where(running, fc, fb)
        else:
            raise RuntimeError('Glicko-2 volatility did not converge')
        sigma = (a/2).exp()
        phi_star = (phi**2+sigma**2).sqrt()
        posterior_variance = (phi_star.pow(-2)+v.reciprocal()).reciprocal()
        self.mu[active] += posterior_variance*gradient[active]
        self.phi[active] = posterior_variance.sqrt()
        self.sigma[active] = sigma
        self.last[active] = when
        e = torch.tensor([r[0][0] for r in stored], device='cuda', dtype=torch.float64)
        increment = 20*(target-e)
        self.elo.index_add_(0, teams[:, 0], increment)
        self.elo.index_add_(0, teams[:, 1], -increment)
        full = np.array([r[2] for r in stored], dtype=bool)
        if full.any():
            selected = np.flatnonzero(full)
            player = torch.tensor(np.stack([stored[i][1] for i in selected]), device='cuda')
            pm = self.pm[player]
            pv = self.player_variance(player, when)
            sign = torch.tensor(2*np.asarray(labels)[selected]-1, device='cuda', dtype=torch.float64)
            c2 = pv.sum((1, 2))+10*(25/6)**2
            c = c2.sqrt()
            t = sign*(pm[:, 0].sum(1)-pm[:, 1].sum(1))/c
            vcorrection = (-.5*t*t-.5*math.log(2*math.pi)-torch.special.log_ndtr(t)).exp()
            wcorrection = (vcorrection*(vcorrection+t)).clamp(0, 1-1e-9)
            sidesign = torch.stack((sign, -sign), 1)
            postmean = pm+sidesign[:, :, None]*pv/c[:, None, None]*vcorrection[:, None, None]
            postvar = pv*(1-pv/c2[:, None, None]*wcorrection[:, None, None])
            affected = torch.unique(player)
            precision_delta = torch.zeros_like(self.pm)
            natural_delta = torch.zeros_like(self.pm)
            precision_delta.index_add_(0, player.flatten(), (postvar.reciprocal()-pv.reciprocal()).flatten())
            natural_delta.index_add_(0, player.flatten(), (postmean/postvar-pm/pv).flatten())
            oldvar = self.player_variance(affected, when)
            newvar = (oldvar.reciprocal()+precision_delta[affected]).reciprocal()
            self.pm[affected] = newvar*(self.pm[affected]/oldvar+natural_delta[affected])
            self.pv[affected] = newvar
            self.plast[affected] = when
        appeared = collections.defaultdict(set)
        for row in batch:
            if row['task']=='map_winner':
                for team, roster in self.lineups[row['game_id']].items():
                    appeared[team].add(tuple(sorted(roster)))
        for team, rosters in appeared.items():
            self.rosters[team] = next(iter(rosters)) if len(rosters)==1 else ()
        if not bool(torch.isfinite(self.mu[active]).all()) or not bool(torch.isfinite(self.pm).all()):
            raise RuntimeError('Nonfinite rating posterior')
        self.update_count += len(batch)


def sources(inputs):
    records = inputs.json(DYNAMIC/'online/sources.json')
    meta, labels = {}, {}
    for rec in records:
        keys = inputs.json(rec['keys'], rec['keys_sha256'])
        yy = np.load(inputs.bind(rec['labels'], rec['labels_sha256']))
        task = Path(rec['keys']).parent.name
        for k, y in zip(keys, yy):
            uid = task, k['game_id']
            if k['event_id']=='2766':
                raise RuntimeError('Protected event in source inventory')
            if uid in meta and any(meta[uid][f] != k[f] for f in ('match_id', 'event_id', 'anchor', 'prediction_cutoff', 'team_a_id', 'team_b_id')):
                raise RuntimeError('Conflicting chronological identity')
            meta.setdefault(uid, dict(k, task=task))
            labels.setdefault(uid, float(y))
    cfg = inputs.json(CORE/'config.json')
    ids = [g for t, g in meta if t=='map_winner']
    connection = duckdb.connect(cfg['source_database_path'], read_only=True)
    try:
        rows = connection.execute('select game_id,team_id,player_id from core.lineups where game_id in (select unnest(?))', [ids]).fetchall()
    finally:
        connection.close()
    lineups = collections.defaultdict(lambda: collections.defaultdict(set))
    for g, t, p in rows:
        lineups[str(g)][str(t)].add(str(p))
    if any(len(lineups[g][k['team_a_id']])!=5 or len(lineups[g][k['team_b_id']])!=5 for (t,g),k in meta.items() if t=='map_winner'):
        raise RuntimeError('Incomplete map lineup')
    identity = sorted((g, t, p) for g, teams in lineups.items() for t, players in teams.items() for p in players)
    inputs.hashes['lineup_projection_sha256'] = __import__('hashlib').sha256(json.dumps(identity).encode()).hexdigest()
    inputs.hashes['source_database_expected_sha256'] = cfg['source_database_sha256']
    teams = {t: i for i,t in enumerate(sorted({k[f] for k in meta.values() for f in ('team_a_id','team_b_id')}))}
    players = {p: i for i,p in enumerate(sorted({p for _,_,p in identity}))}
    return meta, labels, teams, players, lineups


def fit_stream(meta, labels, teams, players, lineups, delay, admitted, progress):
    rating = Ratings(teams, players, lineups)
    groups = collections.defaultdict(list)
    for uid, row in meta.items():
        groups[row['prediction_cutoff']].append(uid)
    pending, forecasts, stored = [], {}, {}
    started = time.time()
    for index, (cutoff, units) in enumerate(sorted(groups.items())):
        while pending and pending[0][0]<cutoff:
            when = pending[0][0]
            released = []
            while pending and pending[0][0]==when:
                _, uid = heapq.heappop(pending)
                if uid in admitted and meta[uid]['region']!='global' and np.isfinite(labels[uid]):
                    released.append(uid)
            if released:
                rating.update([meta[u] for u in released], [labels[u] for u in released], [stored[u] for u in released], when)
        batch = [meta[u] for u in units]
        probability, roster, full = rating.predict(batch, cutoff)
        for j, uid in enumerate(units):
            forecasts[uid] = probability[j]
            stored[uid] = probability[j], roster[j], full[j]
            heapq.heappush(pending, (meta[uid]['anchor']+delay, uid))
        if index%1000==0:
            save(progress, dict(batches=index, total=len(groups), seconds=time.time()-started, updates=rating.update_count))
            print(progress.parent.name, index, '/', len(groups), 'seconds', round(time.time()-started,1), flush=True)
    return forecasts, dict(seconds=time.time()-started, updates=rating.update_count, batches=len(groups), gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--methods', type=Path, required=True)
    parser.add_argument('--regime', choices=('online','frozen'), required=True)
    parser.add_argument('--delay-hours', type=int, choices=(2, 6))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA mandatory for numerical ratings')
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    delay = (args.delay_hours if args.delay_hours is not None else 6 if args.regime=='online' else 2)*3600
    identity = dict(code=sha(__file__), methods=sha(args.methods), regime=args.regime, delay_seconds=delay, torch=torch.__version__)
    marker = args.output/'identity.json'
    if marker.exists() and read(marker)!=identity:
        raise RuntimeError('Use a new version for changed fitting identity')
    save(marker, identity)
    shutil.copy2(__file__, args.output/'rating-code.py')
    shutil.copy2(args.methods, args.output/'methods.md')
    inputs = Inputs()
    meta, labels, teams, players, lineups = sources(inputs)
    if (args.output/'source-identity.json').exists() and read(args.output/'source-identity.json') != inputs.hashes:
        raise RuntimeError('Completed rating source identity changed; use a new version')
    save(args.output/'source-identity.json', inputs.hashes)
    cfg = read(CORE/'config.json')
    if args.regime=='online':
        jobs = [('online', meta, set(meta), delay)]
    else:
        jobs = []
        for fold in cfg['folds']:
            selected, admitted = {}, set()
            for task in TASKS:
                for phase in PHASES:
                    for k in read(CORE/'blocks'/fold['name']/'inputs'/task/(phase+'-keys.json')):
                        uid = task,k['game_id']
                        selected[uid] = dict(k, task=task)
                        if phase=='train':
                            admitted.add(uid)
            jobs.append((fold['name'], selected, admitted, delay))
    for name, local, admitted, delay in jobs:
        directory = args.output/name
        directory.mkdir(exist_ok=True)
        if (directory/'frozen.json').exists():
            record = read(directory/'frozen.json')
            if sha(directory/'predictions.npz')!=record['predictions']:
                raise RuntimeError('Saved rating unit changed')
            continue
        try:
            forecasts, stats = fit_stream(local, labels, teams, players, lineups, delay, admitted, directory/'progress.json')
            order = sorted(forecasts)
            path = directory/'predictions.npz'
            with path.with_suffix('.pending').open('wb') as stream:
                np.savez_compressed(stream, probability=np.stack([forecasts[u] for u in order]))
            path.with_suffix('.pending').replace(path)
            save(directory/'keys.json', order)
            save(directory/'frozen.json', dict(identity=identity, predictions=sha(path), keys=sha(directory/'keys.json'), telemetry=stats,
                                               columns=['elo_k20','glicko2','player_gaussian'], outcomes_used='only released admitted nonglobal rows; equal-cutoff predictions frozen', scoring=False))
        except Exception as error:
            save(directory/f'failure-{time.time_ns()}.json', dict(error=repr(error), identity=identity))
            raise
    save(args.output/'completion.json', dict(identity=identity, units=[name for name, *_ in jobs], scoring=False))
    print('RATING FORECASTS FROZEN', args.regime, flush=True)


if __name__=='__main__':
    main()
