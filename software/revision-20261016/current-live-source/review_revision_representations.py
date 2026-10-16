"""Controlled real CUDA player-pooling fits with recoverable seed/epoch units."""
import argparse
import gc
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from review_revision_diagnostics import CORE, PHASES, TASKS, read, save, sha
from review_revision_rating_report import calibrate, scale_probability

sys.path.insert(0,str(CORE/'code'))
from preprocess import arrays, swap

POOLS=('sum','mean','attention','summary')
REPS=('BN','BNC','BNL','BNLC')
SEEDS=(20261009,20261010,20261011)


class PlayerPool(torch.nn.Module):
    def __init__(self,width,player_width,pool):
        super().__init__()
        self.pool=pool
        self.encoder=torch.nn.Sequential(torch.nn.Linear(player_width,64),torch.nn.ReLU(),torch.nn.Dropout(.1),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Dropout(.1))
        self.query=torch.nn.Parameter(torch.zeros(64)) if pool=='attention' else None
        pooled_width=256 if pool=='summary' else 64
        self.head=torch.nn.Sequential(torch.nn.Linear(width+2*pooled_width,128),torch.nn.ReLU(),torch.nn.Dropout(.1),torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Dropout(.1),torch.nn.Linear(128,1))

    def forward(self,x,players):
        z=self.encoder(players)
        if self.pool=='sum':z=z.sum(2)
        elif self.pool=='mean':z=z.mean(2)
        elif self.pool=='attention':z=((z*self.query).sum(-1).softmax(2)[...,None]*z).sum(2)
        else:z=torch.cat((z.mean(2),z.std(2,unbiased=False),z.max(2).values,z.topk(2,dim=2).values.mean(2)),2)
        return self.head(torch.cat((x,z.flatten(1)),1)).flatten()


def tensor_inputs(block,task,rep):
    mode='full' if 'L' in rep else 'general'
    data,keys,proc=arrays(block,task,rep,player_mode=mode)
    if not data:return data,keys,proc
    meta=read(block/'inputs/prepared.json')['tasks'][task]
    categories=len(meta['player_agent_vocabulary'])+1
    for phase in PHASES:
        if mode=='full':
            data[phase]['players']=data[phase]['players'][...,:-categories]
        if 'C' in rep:
            with np.load(block/'inputs'/task/(phase+'-players.npz')) as z:
                agents=torch.tensor(z['agents'],device='cuda',dtype=torch.long)
            data[phase]['players']=torch.cat((data[phase]['players'],F.one_hot(agents,categories).float()),-1)
    proc['controlled_player_channels']=dict(linked='L' in rep,agent_categories='C' in rep)
    return data,keys,proc


def checkpoint(path,value):
    temporary=path.with_suffix('.pending')
    with temporary.open('wb') as stream:
        torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def checkpoint_link(path,target):
    temporary=path.with_suffix('.link-pending')
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target)
    temporary.replace(path)
    descriptor=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def predict(model,v,proc,batch):
    out=[]
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for start in range(0,len(v['x']),batch):
            x,p=v['x'][start:start+batch],v['players'][start:start+batch]
            out.append((model(x,p).float().sigmoid()+1-model(swap(x,proc),p.flip(1)).float().sigmoid())/2)
    return torch.cat(out).cpu().numpy() if out else np.empty(0)


def seed_fit(data,keys,proc,pool,seed,directory,identity):
    directory.mkdir(parents=True,exist_ok=True)
    marker=directory/'frozen.json'
    if marker.exists():
        old=read(marker)
        if old['identity']!=identity or sha(directory/'predictions.npz')!=old['predictions']:
            raise RuntimeError('Completed seed identity mismatch')
        return old
    train=data['train']
    x=torch.cat((train['x'],swap(train['x'],proc)))
    players=torch.cat((train['players'],train['players'].flip(1)))
    y=torch.cat((train['y'],1-train['y']))
    weight=train['weight'].repeat(2)
    stopping=data['stopping']
    sx=torch.cat((stopping['x'],swap(stopping['x'],proc)))
    sp=torch.cat((stopping['players'],stopping['players'].flip(1)))
    sy=torch.cat((stopping['y'],1-stopping['y']))
    sw=stopping['weight'].repeat(2)
    for batch in (1024,512,256,128):
        attempt=directory/f'batch-{batch}';attempt.mkdir(exist_ok=True)
        if (attempt/'oom.json').exists():continue
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);random.seed(seed);np.random.seed(seed)
        model=PlayerPool(x.shape[1],players.shape[-1],pool).cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        start_epoch,best,stale,best_epoch=0,float('inf'),0,None
        resume=attempt/'last.pt'
        if resume.exists():
            old=torch.load(resume,weights_only=False,map_location='cuda')
            model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer'])
            start_epoch,best,stale=old['epoch']+1,old['best'],old['stale']
            best_epoch=old['best_epoch']
            best_file=attempt/f'epoch-{best_epoch:03d}.pt'
            if not best_file.exists():raise RuntimeError('Durable best checkpoint is missing')
            checkpoint_link(attempt/'best.pt',best_file.name)
            torch.set_rng_state(old['cpu_rng'].cpu());torch.cuda.set_rng_state(old['cuda_rng'].cpu())
        started=time.time()
        try:
            for epoch in range(start_epoch,64):
                if stale>=8:break
                tick=time.time();model.train()
                generator=torch.Generator(device='cuda').manual_seed(seed+epoch)
                permutation=torch.randperm(len(y),device='cuda',generator=generator)
                train_loss=0.
                for offset in range(0,len(y),batch):
                    ix=permutation[offset:offset+batch];optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits=model(x[ix],players[ix]).float()
                        loss=(F.binary_cross_entropy_with_logits(logits,y[ix],reduction='none')*weight[ix]).mean()
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite player training objective')
                    loss.backward();optimizer.step();train_loss+=float(loss.detach())*len(ix)
                model.eval();total=0.
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    for offset in range(0,len(sy),batch):
                        sl=slice(offset,offset+batch)
                        logits=model(sx[sl],sp[sl]).float()
                        total+=float((F.binary_cross_entropy_with_logits(logits,sy[sl],reduction='none')*sw[sl]).sum())
                stopping_loss=total/len(sy)
                if not np.isfinite(stopping_loss):raise RuntimeError('Nonfinite stopping objective')
                improved=stopping_loss<best-1e-6
                best,stale=(stopping_loss,0) if improved else (best,stale+1)
                if improved:best_epoch=epoch+1
                state=dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),best=best,best_epoch=best_epoch,stale=stale,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
                epoch_path=attempt/f'epoch-{epoch+1:03d}.pt';checkpoint(epoch_path,state)
                checkpoint_link(resume,epoch_path.name)
                if improved:
                    checkpoint_link(attempt/'best.pt',epoch_path.name)
                telemetry=dict(epoch=epoch+1,batch=batch,seconds=time.time()-tick,examples_per_second=len(y)/(time.time()-tick),train_log_loss=train_loss/len(y),stopping_log_loss=stopping_loss,best=best,stale=stale,gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
                save(attempt/'progress.json',telemetry)
                with (attempt/'epochs.jsonl').open('a') as stream:stream.write(__import__('json').dumps(telemetry)+'\n')
            best_path=attempt/f'epoch-{best_epoch:03d}.pt'
            old=torch.load(best_path,weights_only=False,map_location='cuda');model.load_state_dict(old['model'])
            predictions={phase:predict(model,data[phase],proc,batch) for phase in PHASES}
            if any(len(predictions[p])!=len(keys[p]) or not np.isfinite(predictions[p]).all() for p in PHASES):raise RuntimeError('Invalid player forecast freeze')
            with (directory/'predictions.npz').open('wb') as stream:np.savez_compressed(stream,**predictions)
            record=dict(identity=identity,predictions=sha(directory/'predictions.npz'),batch=batch,best_epoch=old['epoch']+1,parameters=sum(p.numel() for p in model.parameters()),seconds=time.time()-started,best_checkpoint_sha256=sha(best_path))
            save(marker,record);return record
        except torch.OutOfMemoryError as error:
            save(attempt/'oom.json',dict(reason=str(error),batch=batch))
            model=optimizer=None;gc.collect();torch.cuda.empty_cache()
        except Exception as error:
            save(attempt/f'failure-{time.time_ns()}.json',dict(reason=repr(error),identity=identity));raise
    raise RuntimeError('Retained OOM attempts reached batch floor')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--corrected',type=Path,required=True)
    parser.add_argument('--methods',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory; no CPU model fallback')
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    if not (args.corrected/'completion.json').exists():raise RuntimeError('Complete corrected input projection first')
    args.output.mkdir(parents=True,exist_ok=True)
    identity=dict(code=sha(__file__),methods=sha(args.methods),corrected=sha(args.corrected/'completion.json'),core=sha(CORE/'release-identity.json'),preprocess_code=sha(CORE/'code/preprocess.py'),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0))
    if (args.output/'identity.json').exists() and read(args.output/'identity.json')!=identity:raise RuntimeError('Changed study identity; use a new root')
    save(args.output/'identity.json',identity);shutil.copy2(__file__,args.output/'code.py');shutil.copy2(args.methods,args.output/'methods.md')
    frozen=[]
    for fold in read(CORE/'config.json')['folds']:
        for task in TASKS:
            for projection,rep,pools in [('corrected',rep,POOLS) for rep in REPS]+[('original','BNLC',('sum',))]:
                source=args.corrected/fold['name'] if projection=='corrected' else CORE/'blocks'/fold['name']
                prepared=read(source/'inputs/prepared.json')
                for file,expected in prepared['files'].items():
                    if file.startswith(task+'/') and sha(source/'inputs'/file)!=expected:raise RuntimeError('Player input snapshot changed')
                for pool in pools:
                    directory=args.output/fold['name']/task/f'{projection}-{rep}-{pool}'
                    unit=identity|dict(block=fold['name'],task=task,projection=projection,rep=rep,pool=pool,prepared=sha(source/'inputs/prepared.json'))
                    if (directory/'frozen.json').exists():
                        receipt=read(directory/'frozen.json')
                        if receipt['identity']!=unit or sha(directory/'predictions.npz')!=receipt['predictions']:raise RuntimeError('Completed package changed')
                        frozen.append(str(directory/'frozen.json'));continue
                    directory.mkdir(parents=True,exist_ok=True)
                    data,keys,proc=tensor_inputs(source,task,rep)
                    eligible=bool(data) and len(np.unique(data['train']['y'].cpu().numpy()))==2 and all(len({k['event_id'] for k in keys[p]})>=5 and len(np.unique(data[p]['y'].cpu().numpy()))==2 for p in ('stopping','selection'))
                    pp=[]
                    if eligible:
                        for seed in SEEDS:
                            print(fold['name'],task,projection,rep,pool,seed,flush=True)
                            record=seed_fit(data,keys,proc,pool,seed,directory/str(seed),unit|dict(seed=seed))
                            pp.append({p:v.copy() for p,v in np.load(directory/str(seed)/'predictions.npz').items()})
                        raw={phase:np.mean([p[phase] for p in pp],axis=0) for phase in PHASES}
                    else:
                        names=prepared['tasks'][task]['feature_names'];column=names.index('elo_log_odds_difference')
                        raw={}
                        for phase in PHASES:
                            with np.load(source/'inputs'/task/(phase+'.npz')) as z:logit=np.nan_to_num(z['numeric'][:,column])
                            raw[phase]=1/(1+np.exp(-np.clip(logit,-30,30)))
                    cy=np.load(source/'inputs'/task/'calibration-labels.npy')
                    scale,calibration=calibrate(raw['calibration'],cy,keys['calibration'])
                    with (directory/'predictions.npz').open('wb') as stream:
                        np.savez_compressed(stream,**{p+'_raw':v for p,v in raw.items()},**{p+'_calibrated':scale_probability(v,scale) for p,v in raw.items()})
                    save(directory/'processor.json',proc)
                    save(directory/'frozen.json',dict(identity=unit,trained=eligible,calibration=calibration,scale=scale,predictions=sha(directory/'predictions.npz'),evaluation_labels_read=False))
                    frozen.append(str(directory/'frozen.json'))
                    del data;gc.collect();torch.cuda.empty_cache()
            save(args.output/'progress.json',dict(block=fold['name'],task=task,completed_packages=len(frozen)))
    save(args.output/'all-predictions-frozen.json',dict(identity=identity,packages={p:sha(p) for p in frozen},evaluation_labels_read=False))
    print('ALL CONTROLLED PLAYER FORECASTS FROZEN',flush=True)


if __name__=='__main__':main()
