"""Cadencia v1: audio-conditioned phrase training with joint transition supervision."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from cadencia.placement import PlacementConfig
from cadencia.planner import BarPlanner, PlannedTransformer, PlannedDataset, metadata, PLANNER_FORMAT
from cadencia.phrase_training import PhraseDataset, PhraseBatchSampler, collate_phrase
from cadencia.beat_alignment import load_prior
if __package__:
    from .train_intervals import run_epoch
else:
    from train_intervals import run_epoch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/placement_dataset_v1')
    p.add_argument('--planner', default='data/bar_planner_v1.pt')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--initialize', help='Best audio-conditioned decoder checkpoint; initializes weights, not optimizer')
    mode.add_argument('--resume', help='Phrase-training .last.pt including optimizer, scheduler, sampling and RNG')
    p.add_argument('--output', default='data/decoder_v1.pt')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--phrase-beats', type=float, default=4.)
    p.add_argument('--phrase-weight', type=float, default=.05)
    p.add_argument('--phrase-warmup-epochs', type=int, default=3)
    p.add_argument('--batch-rows', type=int, default=128)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--chunk-rows', type=int, default=128)
    p.add_argument('--empty-keep', type=float, default=.1)
    p.add_argument('--near-rows', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    import math
    if (args.epochs < 1 or args.workers < 0 or args.chunk_rows < 1 or args.batch_rows < 1 or args.phrase_warmup_epochs < 0 or args.seed < 0 or args.near_rows < 0 or
            not 0 < args.empty_keep <= 1 or (not math.isfinite(args.phrase_weight) or args.phrase_weight < 0) or any(not math.isfinite(x) or x <= 0 for x in (args.phrase_beats,args.lr))):
        p.error('Invalid phrase training configuration')
    if args.amp and (args.device != 'cuda' or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        p.error('--amp requires a CUDA device with bfloat16 support')
    output = Path(args.output)
    if not args.resume and (output.exists() or output.with_suffix('.last.pt').exists()):
        raise FileExistsError('Output exists; use --resume or another output')
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    planner_checkpoint = torch.load(args.planner,map_location='cpu',weights_only=True)
    if planner_checkpoint['format'] != PLANNER_FORMAT:
        raise ValueError('Invalid planner')
    planner = BarPlanner(planner_checkpoint.get('audio_dim',768),planner_checkpoint.get('use_rhythm',False))
    planner.load_state_dict(planner_checkpoint['model'],strict=True)
    print('Loading and verifying dataset splits...',flush=True)
    bases = [PlannedDataset(args.data,split) for split in ('train','validation','test')]
    dataset_hash, planner_hash = digest(bases[0].path), digest(args.planner)
    prior_hash = load_prior()[1] if bases[0].config.use_mel else None
    if planner_checkpoint['dataset_sha256'] != dataset_hash or planner_checkpoint['mert'] != bases[0].manifest['mert']:
        raise ValueError('Planner must match the dataset')
    print('Preparing frozen planner outputs...',flush=True)
    for base in bases:
        if not base.entries:
            raise ValueError('Every split must have charts')
        base.prepare_plans(planner)
    datasets = [PhraseDataset(base,args.phrase_beats,training=i==0,empty_keep=args.empty_keep,near_rows=args.near_rows,seed=args.seed) for i,base in enumerate(bases)]
    settings = dict(beats=args.phrase_beats,weight=args.phrase_weight,empty_keep=args.empty_keep,near_rows=args.near_rows,
                    seed=args.seed,learning_rate=args.lr,chunk_rows=args.chunk_rows,
                    recipe='mixed_phrases_v2',batch_rows=args.batch_rows,warmup_epochs=args.phrase_warmup_epochs,
                    timing_normalization='fixed original grid-row count',pattern_normalization='all reference onsets in mixed batch',
                    duration_normalization='all reference hold starts in mixed batch',transition_normalization='within-phrase transition-pair count',
                    checkpoint_selection='validation base_loss',
                    objective='event_bce + joint_pattern_ce + hold_duration_smooth_l1 + weighted_joint_transition_js',
                    history='causal teacher-forced reference history',gap_buckets_beats=[.5,1.],
                    windows='nonoverlapping annotated-beat windows, including partial intro/outro',
                    gradient_checkpointing=True)
    model = PlannedTransformer(bases[0].config).to(device)
    checkpoint = torch.load(args.resume or args.initialize,map_location='cpu',weights_only=True)
    if (checkpoint['dataset_sha256'] != dataset_hash or checkpoint.get('planner_sha256') != planner_hash
            or asdict(PlacementConfig(**checkpoint['config'])) != asdict(model.config)
            or checkpoint['mert'] != bases[0].manifest['mert']
            or checkpoint.get('alignment_prior_sha256') != prior_hash):
        raise ValueError('Source checkpoint dataset, model, planner or alignment prior mismatch')
    initialization = checkpoint.get('initialization') if args.resume else dict(path=str(Path(args.initialize).resolve()),sha256=digest(args.initialize),
                         source_version=checkpoint.get('architecture_version'),source_best_epoch=checkpoint.get('best_epoch'))
    model.load_state_dict(checkpoint['last_model'] if args.resume else checkpoint['model'],strict=True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=3,min_lr=1e-6)
    history, best, best_epoch, best_state = [], float('inf'), 0, None
    baseline_validation = None
    if args.resume:
        if checkpoint.get('phrase_training') != settings:
            raise ValueError('Resume phrase configuration mismatch')
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        baseline_validation = checkpoint['baseline_validation']
        history, best, best_epoch, best_state = checkpoint['history'], checkpoint['best_loss'], checkpoint['best_epoch'], checkpoint['model']
        torch.set_rng_state(checkpoint['rng_state'])
        if device.type == 'cuda': torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
        if len(history) >= args.epochs: p.error('--epochs must exceed completed epochs')
    sampler = PhraseBatchSampler(datasets[0],args.batch_rows,seed=args.seed)
    sampler.epoch = len(history)
    loaders = []
    for i,dataset in enumerate(datasets):
        options = dict(batch_sampler=sampler if i==0 else PhraseBatchSampler(dataset,args.batch_rows,shuffle=False),
                       num_workers=args.workers if i==0 else 0,collate_fn=collate_phrase,pin_memory=device.type=='cuda')
        if i==0 and args.workers:
            options.update(persistent_workers=True,multiprocessing_context='spawn')
        loaders.append(DataLoader(dataset,**options))
    for name,base,dataset in zip(('train','validation','test'),bases,datasets):
        print(f'{name}: charts={len(base.entries)} phrases={len(dataset)} full_grid_rows={len(base)}',flush=True)
    print(f'device={device} amp={args.amp} version=v1 initialized={initialization}',flush=True)
    print(f'phrase_training={settings}',flush=True)
    # Save initialization immediately; interruption before epoch 1 is recoverable.
    def state():
        return dict(**metadata(model),mert=bases[0].manifest['mert'],model=best_state,
                    dataset_sha256=dataset_hash,planner=planner_checkpoint,planner_sha256=planner_hash,
                    alignment_prior_sha256=prior_hash,chart_pretraining=checkpoint.get('chart_pretraining'),
                    initialization=initialization,phrase_training=settings,epochs_completed=len(history),
                    best_epoch=best_epoch,best_loss=best,baseline_validation=baseline_validation,
                    splits={name:[e['chart_id'] for e in base.entries] for name,base in zip(('train','validation','test'),bases)})
    def persist():
        value = state()
        value['architecture_version'] = 'v1'
        save(output,value)
        save(output.with_suffix('.last.pt'),dict(value,last_model={k:v.detach().cpu() for k,v in model.state_dict().items()},
              optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),history=history,
              rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all() if device.type=='cuda' else []))
    if best_state is None:
        print('Evaluating decoder initialization before any updates...',flush=True)
        baseline_validation = run_epoch(model,loaders[1],device,amp=args.amp,phrase_weight=args.phrase_weight,chunk_rows=args.chunk_rows,measure_phrase=True)
        best = baseline_validation['base_loss']
        print(f'baseline base_loss={best:.6f} phrase={baseline_validation["phrase_transition"]:.6f}',flush=True)
        best_state = copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
        persist()
    for epoch in range(len(history)+1,args.epochs+1):
        start = time.perf_counter()
        effective_weight = args.phrase_weight*min(1.,epoch/max(1,args.phrase_warmup_epochs))
        train = run_epoch(model,loaders[0],device,optimizer,args.amp,phrase_weight=effective_weight,chunk_rows=args.chunk_rows,measure_phrase=True)
        validation = run_epoch(model,loaders[1],device,amp=args.amp,phrase_weight=args.phrase_weight,chunk_rows=args.chunk_rows,measure_phrase=True)
        scheduler.step(validation['base_loss'])
        history.append(dict(epoch=epoch,train=train,validation=validation,seconds=time.perf_counter()-start,learning_rate=optimizer.param_groups[0]['lr'],effective_phrase_weight=effective_weight))
        if validation['base_loss'] < best:
            best,best_epoch = validation['base_loss'],epoch
            best_state = copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
        persist()
        report = dict(history=history,epochs_completed=len(history),best_epoch=best_epoch,phrase_training=settings,
                      initialization=initialization,dataset_sha256=dataset_hash,baseline_validation=baseline_validation)
        temporary = output.with_suffix('.metrics.json.tmp');temporary.write_text(json.dumps(report,indent=2));temporary.replace(output.with_suffix('.metrics.json'))
        print(f'epoch={epoch} train={train["loss"]:.4f} validation={validation["loss"]:.4f} base_validation={validation["base_loss"]:.4f} best_epoch={best_epoch} phrase_train={train["phrase_transition"]:.4f} phrase_validation={validation["phrase_transition"]:.4f} seconds={history[-1]["seconds"]:.1f}',flush=True)
    model.load_state_dict(best_state)
    report['test'] = run_epoch(model,loaders[2],device,amp=args.amp,phrase_weight=args.phrase_weight,chunk_rows=args.chunk_rows,measure_phrase=True)
    temporary = output.with_suffix('.metrics.json.tmp');temporary.write_text(json.dumps(report,indent=2));temporary.replace(output.with_suffix('.metrics.json'))


if __name__ == '__main__':
    main()
