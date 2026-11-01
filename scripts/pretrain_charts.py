"""Pretrain chart history on next-onset gap, joint pattern, and hold duration without audio."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from contextlib import nullcontext
import torch
from torch.utils.data import DataLoader
from cadencia.chart_pretraining import (ChartSequenceDataset,ChartSequenceModel,sequence_loss,
                                        FORMAT,split_identities,assert_disjoint)
from cadencia.interval import decode
from cadencia.placement import PlacementConfig


def run_epoch(model,loader,device,optimizer=None,amp=False):
    model.train(optimizer is not None)
    sums=dict(pattern=0.,duration=0.,gap=0.);counts=dict(pattern=0,duration=0,gap=0)
    confusion=torch.zeros(3,3,dtype=torch.long,device=device)
    gap_error,duration_error,rows=0.,0.,0
    with torch.set_grad_enabled(optimizer is not None):
        for number,batch in enumerate(loader):
            batch={k:v.to(device) for k,v in batch.items()}
            target=batch.pop('target');duration=batch.pop('duration_target');gap=batch.pop('gap_target')
            with torch.autocast('cuda',dtype=torch.bfloat16) if amp else nullcontext():
                output=model(**batch);loss,parts=sequence_loss(output,target,duration,gap)
            if not torch.isfinite(loss):raise ValueError('Nonfinite pretraining loss')
            if optimizer:
                optimizer.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            n=len(target);holds=target==2
            for key,support in dict(pattern=n,gap=n,duration=int(holds.sum())).items():
                sums[key]+=float(parts[key].detach())*support;counts[key]+=support
            predicted=decode(output['pattern'].argmax(-1)+1)
            confusion+=torch.bincount((target*3+predicted).ravel(),minlength=9).reshape(3,3)
            gap_error+=float((output['gap'].detach().float().clamp(max=12).expm1()-gap.expm1()).abs().sum())
            duration_error+=float((output['duration'][holds].detach().float().clamp(0,8).expm1()-duration[holds].expm1()).abs().sum())
            rows+=n
            if (number+1)%200==0:print(f'  batches={number+1} onset_rows={rows}',flush=True)
    if not rows:raise ValueError('Empty onset split')
    metrics={key:sums[key]/max(1,counts[key]) for key in sums};metrics['loss']=sum(metrics.values())
    metrics.update(rows=rows,gap_mae_beats=gap_error/rows,hold_duration_mae_beats=duration_error/max(1,counts['duration']))
    metrics['per_action']=[]
    confusion=confusion.cpu()
    for i,name in enumerate(['absent','tap','hold_start']):
        tp,pred,actual=int(confusion[i,i]),int(confusion[:,i].sum()),int(confusion[i].sum())
        metrics['per_action'].append(dict(action=name,precision=tp/max(1,pred),recall=tp/max(1,actual),f1=2*tp/max(1,pred+actual),support=actual))
    metrics['evaluation']='Teacher-forced next-onset prediction, not full-grid audio timing metrics'
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/placement_dataset_v1')
    p.add_argument('--output',default='data/chart_pretraining.pt')
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--amp',action='store_true')
    p.add_argument('--resume')
    args=p.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.workers<0:p.error('Invalid epochs, batch size, or worker count')
    if args.amp and (args.device!='cuda' or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):p.error('--amp requires CUDA bfloat16 support')
    torch.set_num_threads(2);torch.manual_seed(args.seed);device=torch.device(args.device)
    datasets=[ChartSequenceDataset(args.data,s) for s in ['train','validation','test']]
    assert_disjoint(datasets[0].manifest)
    loaders=[]
    for i,ds in enumerate(datasets):
        if not len(ds):p.error('All splits must have onset events')
        options=dict(num_workers=args.workers if i==0 else 0)
        if options['num_workers']:options.update(multiprocessing_context='spawn',persistent_workers=True)
        loaders.append(DataLoader(ds,batch_size=args.batch_size,shuffle=i==0,**options))
    model=ChartSequenceModel(datasets[0].config).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    digest=hashlib.sha256(datasets[0].path.read_bytes()).hexdigest();history=[];best=float('inf');best_epoch=0;best_state=None
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=True)
        if saved['format']!=FORMAT or saved['dataset_sha256']!=digest or asdict(PlacementConfig(**saved['config']))!=asdict(model.config):raise ValueError('Resume configuration/dataset mismatch')
        model.load_state_dict(saved['last_model']);optimizer.load_state_dict(saved['optimizer'])
        history,best,best_epoch,best_state=saved['history'],saved['best_loss'],saved['best_epoch'],saved['model']
        if len(history)>=args.epochs:p.error('--epochs must exceed completed epochs')
        torch.set_rng_state(saved['rng_state'])
        if args.device=='cuda':torch.cuda.set_rng_state_all(saved['cuda_rng_state'])
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    def save(path,value):
        tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)
    print(f'Chart-only pretraining device={device}; onsets train/validation/test={[len(ds) for ds in datasets]}',flush=True)
    for epoch in range(len(history)+1,args.epochs+1):
        start=time.perf_counter();train=run_epoch(model,loaders[0],device,optimizer,args.amp);validation=run_epoch(model,loaders[1],device,amp=args.amp)
        history.append(dict(epoch=epoch,train=train,validation=validation,seconds=time.perf_counter()-start))
        if validation['loss']<best:
            best,best_epoch=validation['loss'],epoch;best_state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
        checkpoint=dict(format=FORMAT,config=asdict(model.config),model=best_state,best_epoch=best_epoch,best_loss=best,
                        epochs_completed=len(history),dataset_sha256=digest,split_identities=split_identities(datasets[0].manifest),seed=args.seed,
                        objective='next_onset_pattern_ce+log_gap_smooth_l1+hold_duration_smooth_l1')
        save(out,checkpoint)
        save(out.with_suffix('.last.pt'),dict(**checkpoint,last_model=model.state_dict(),optimizer=optimizer.state_dict(),history=history,
                                             rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all() if args.device=='cuda' else []))
        metrics=dict(history=history,epochs_completed=len(history),best_epoch=best_epoch,dataset_sha256=digest)
        tmp=out.with_suffix('.metrics.json.tmp');tmp.write_text(json.dumps(metrics,indent=2));tmp.replace(out.with_suffix('.metrics.json'))
        print(f'epoch={epoch} train={train["loss"]:.4f} validation={validation["loss"]:.4f}',flush=True)
    model.load_state_dict(best_state);metrics['test']=run_epoch(model,loaders[2],device,amp=args.amp)
    tmp=out.with_suffix('.metrics.json.tmp');tmp.write_text(json.dumps(metrics,indent=2));tmp.replace(out.with_suffix('.metrics.json'))


if __name__=='__main__':main()
