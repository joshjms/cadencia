"""Train a whole-song intensity planner from the existing placement dataset."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import torch
from cadencia.interval import IntervalDataset
from cadencia.planner import BarPlanner, bar_inputs, bar_targets, planner_loss, PLANNER_FORMAT, PLAN_FIELDS


def prepare(dataset):
    songs=[]
    for i,entry in enumerate(dataset.entries):
        target,features=dataset._load(i)
        inputs,rows=bar_inputs(features,target['times'],target['beats'],target['bpm'],entry['difficulty'])
        songs.append((torch.from_numpy(inputs)[None],torch.from_numpy(bar_targets(target,rows))[None]))
    return songs


def epoch(model,songs,device,optimizer=None):
    model.train(optimizer is not None)
    loss_sum,bars=0.,0
    errors=torch.zeros(4,device=device)
    order=torch.randperm(len(songs)).tolist() if optimizer else range(len(songs))
    with torch.set_grad_enabled(optimizer is not None):
        for i in order:
            inputs,target=(v.to(device) for v in songs[i])
            raw=model(inputs)
            loss=planner_loss(raw,target)
            if not torch.isfinite(loss): raise ValueError('Nonfinite planner loss')
            if optimizer:
                optimizer.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
            predicted=torch.stack([raw[...,0].sigmoid(),torch.nn.functional.softplus(raw[...,1]),raw[...,2].sigmoid(),raw[...,3].sigmoid()],-1)
            errors+=(predicted.detach()-target).abs().sum((0,1))
            n=target.shape[1];bars+=n;loss_sum+=float(loss.detach())*n
    if not bars: raise ValueError('Empty planner split')
    return dict(loss=loss_sum/bars,bars=bars,mae=dict(zip(PLAN_FIELDS,(errors/bars).cpu().tolist())))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/placement_dataset_v1')
    p.add_argument('--output',default='data/bar_planner_v1.pt')
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.epochs<1:p.error('epochs must be positive')
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    device=torch.device(args.device)
    datasets=[IntervalDataset(args.data,s) for s in ['train','validation','test']]
    songs=[prepare(ds) for ds in datasets]
    model=BarPlanner(datasets[0].config.audio_dim,datasets[0].config.use_rhythm).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    best=float('inf');history=[];output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    for number in range(1,args.epochs+1):
        train=epoch(model,songs[0],device,optimizer);validation=epoch(model,songs[1],device)
        history.append(dict(epoch=number,train=train,validation=validation))
        if validation['loss']<best:
            best=validation['loss']
            checkpoint=dict(format=PLANNER_FORMAT,model=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()}),
                            plan_fields=PLAN_FIELDS,beats_per_bar=4,best_epoch=number,
                            use_rhythm=datasets[0].config.use_rhythm,audio_dim=datasets[0].config.audio_dim,mert=datasets[0].manifest['mert'],seed=args.seed,
                            dataset_sha256=hashlib.sha256(datasets[0].path.read_bytes()).hexdigest(),
                            splits={s:[e['chart_id'] for e in ds.entries] for s,ds in zip(['train','validation','test'],datasets)})
            temp=output.with_suffix('.pt.tmp');torch.save(checkpoint,temp);temp.replace(output)
        temp=output.with_suffix('.metrics.json.tmp');temp.write_text(json.dumps(dict(history=history),indent=2));temp.replace(output.with_suffix('.metrics.json'))
        print(f'epoch={number} train={train["loss"]:.4f} validation={validation["loss"]:.4f}',flush=True)
    model.load_state_dict(checkpoint['model'])
    metrics=dict(history=history,best_epoch=checkpoint['best_epoch'],test=epoch(model,songs[2],device))
    temp=output.with_suffix('.metrics.json.tmp');temp.write_text(json.dumps(metrics,indent=2));temp.replace(output.with_suffix('.metrics.json'))


if __name__=='__main__':main()
