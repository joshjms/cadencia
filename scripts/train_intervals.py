"""Train event timing, conditional joint onsets, and hold durations on cached MERT data."""
import argparse
import copy
from dataclasses import asdict
from cadencia.placement import PlacementConfig
import hashlib
import json
from pathlib import Path
import time
from contextlib import nullcontext
import torch
from torch.utils.data import DataLoader
from cadencia.interval import (IntervalTransformer, IntervalDataset, SampledIntervalDataset,
                               EventSampler, losses, metadata, decode, legal_pattern_logits, FORMAT)
from cadencia.placement_data import packed_collate


def run_epoch(model, loader, device, optimizer=None, amp=False, *, phrase_weight=0., chunk_rows=0, measure_phrase=False):
    model.train(optimizer is not None)
    totals = dict(timing=0., pattern=0., duration=0.)
    normalizers = dict(timing=0., pattern=0., duration=0.)
    phrase_mode = bool(phrase_weight or measure_phrase)
    phrase_windows = 0
    optimizer_steps = 0
    if phrase_mode:
        totals['phrase_transition'] = normalizers['phrase_transition'] = 0.
    count = 0
    confusion = torch.zeros(3, 3, dtype=torch.long, device=device)
    event_counts = torch.zeros(3, dtype=torch.long, device=device)
    duration_error = 0.
    duration_count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch_index, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            target = batch.pop('target')
            duration = batch.pop('duration_target')
            inclusion = batch.pop('inclusion', None)
            phrase_beats = batch.pop('phrase_beats', None)
            phrase_ids = batch.pop('phrase_ids', None)
            timing_normalizer = batch.pop('timing_normalizer', None)
            with torch.autocast('cuda', dtype=torch.bfloat16) if amp else nullcontext():
                if chunk_rows:
                    from cadencia.phrase_training import forward_rows
                    output = forward_rows(model,batch,chunk_rows,checkpointed=optimizer is not None)
                else:
                    output = model(**batch)
                loss, components = losses(output, target, duration, batch['condition'][:, 4:].bool(), inclusion,
                                          timing_normalizer=timing_normalizer)
                phrase_pairs = 0
                if phrase_mode:
                    from cadencia.phrase_training import grouped_transition_loss
                    if phrase_beats is None:
                        raise ValueError('Phrase objective requires one ordered phrase per batch')
                    phrase_loss, phrase_pairs, windows = grouped_transition_loss(output['pattern'],target,batch['condition'][:,4:].bool(),phrase_beats,phrase_ids)
                    phrase_windows += windows
                    loss = loss+phrase_weight*phrase_loss
                    components['phrase_transition'] = phrase_loss
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite interval loss')
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                optimizer_steps += 1
            n = len(target)
            supports = dict(timing=float(timing_normalizer) if timing_normalizer is not None else float(inclusion.reciprocal().sum()) if inclusion is not None else n,
                            pattern=int(target.any(-1).sum()), duration=int((target == 2).sum()))
            if phrase_mode:
                supports['phrase_transition'] = phrase_pairs
            for key, value in components.items():
                totals[key] += float(value.detach()) * supports[key]
                normalizers[key] += supports[key]
            count += n
            occupied = batch['condition'][:, 4:].bool()
            event = (output['event'] >= 0) & ~occupied.all(-1)
            actual_event = target.any(-1)
            event_counts += torch.stack([(event & actual_event).sum(), (event & ~actual_event).sum(), (~event & actual_event).sum()])
            predicted = torch.zeros_like(target)
            if event.any():
                predicted[event] = decode(legal_pattern_logits(output['pattern'][event], occupied[event]).argmax(-1) + 1)
            confusion += torch.bincount((target * 3 + predicted).ravel(), minlength=9).reshape(3, 3)
            holds = target == 2
            duration_error += float((output['duration'][holds].detach().float().clamp(0,8).expm1() - duration[holds].expm1()).abs().sum())
            duration_count += int(holds.sum())
            if (batch_index + 1) % 200 == 0:
                print(f"  {'train' if optimizer is not None else 'eval'} batches={batch_index+1} rows={count}", flush=True)
    if not count:
        raise ValueError('Empty split')
    result = {k: v/max(1,normalizers[k]) for k,v in totals.items()}
    result['base_loss'] = result['timing']+result['pattern']+result['duration']
    result['loss'] = result['base_loss']+phrase_weight*result.get('phrase_transition',0.)
    if phrase_mode:
        result['phrase_windows_with_transitions'] = phrase_windows
        result['phrase_transition_pairs'] = int(normalizers['phrase_transition'])
        result['optimizer_steps'] = optimizer_steps
        result['phrase_weight'] = phrase_weight
    result['rows'] = count
    result['per_action'] = []
    confusion = confusion.cpu()
    for i, name in enumerate(['empty', 'tap', 'hold_start']):
        tp, predicted, actual = int(confusion[i,i]), int(confusion[:,i].sum()), int(confusion[i].sum())
        result['per_action'].append(dict(action=name, precision=tp/max(1,predicted), recall=tp/max(1,actual), f1=2*tp/max(1,predicted+actual), support=actual))
    tp, fp, fn = event_counts.cpu().tolist()
    result['event'] = dict(precision=tp/max(1,tp+fp), recall=tp/max(1,tp+fn))
    result['hold_duration_mae_beats'] = duration_error/max(1,duration_count)
    result['evaluation'] = 'Teacher-forced full-grid validation; duration MAE at reference hold starts, not generated endpoint accuracy'
    if phrase_mode:
        result['evaluation'] += '; annotated beat windows with differentiable joint-pattern transition JS'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/placement_dataset_v1')
    parser.add_argument('--output', default='data/interval_placement_transformer.pt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--empty-keep', type=float, default=.1)
    parser.add_argument('--near-rows', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--resume')
    parser.add_argument('--planner', help='Trained bar planner checkpoint; enables planned v4 decoder')
    parser.add_argument('--chart-pretraining', help='Initialize shared decoder weights from chart-only pretraining')
    args = parser.parse_args()
    if args.resume and args.chart_pretraining:
        parser.error('--resume and --chart-pretraining are mutually exclusive')
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or not 0 < args.empty_keep <= 1 or args.near_rows < 0:
        parser.error('Invalid training or sampling configuration')
    if args.amp and (args.device != 'cuda' or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        parser.error('--amp requires CUDA bfloat16 support')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    planner_checkpoint = None
    alignment_prior_sha256 = None
    model_class, metadata_fn, expected_format = IntervalTransformer, metadata, FORMAT
    if args.planner:
        from cadencia.planner import (BarPlanner, PlannedTransformer, PlannedDataset, SampledPlannedDataset,
                                      metadata as planned_metadata, FORMAT as planned_format, HIGH_RES_FORMAT, PLANNER_FORMAT)
        planner_checkpoint = torch.load(args.planner,map_location='cpu',weights_only=True)
        if planner_checkpoint.get('format') != PLANNER_FORMAT:
            raise ValueError('Unsupported planner checkpoint')
        planner = BarPlanner(planner_checkpoint.get('audio_dim',768),planner_checkpoint.get('use_rhythm',False))
        planner.load_state_dict(planner_checkpoint['model'],strict=True)
        datasets = [SampledPlannedDataset(args.data,'train'), PlannedDataset(args.data,'validation'), PlannedDataset(args.data,'test')]
        digest = hashlib.sha256(datasets[0].path.read_bytes()).hexdigest()
        if planner_checkpoint.get('use_rhythm',False)!=datasets[0].config.use_rhythm or planner_checkpoint['dataset_sha256'] != digest or planner_checkpoint['mert'] != datasets[0].manifest['mert']:
            raise ValueError('Planner must be trained on the same dataset and splits')
        for dataset in datasets:
            dataset.prepare_plans(planner)
        model_class, metadata_fn, expected_format = PlannedTransformer, planned_metadata, (HIGH_RES_FORMAT if datasets[0].config.use_mel else planned_format)
        if args.output == 'data/interval_placement_transformer.pt':
            args.output = 'data/audio_decoder_v1.pt' if datasets[0].config.use_mel else 'data/planned_placement_transformer.pt'
    else:
        datasets = [SampledIntervalDataset(args.data,'train'), IntervalDataset(args.data,'validation'), IntervalDataset(args.data,'test')]
    if datasets[0].config.use_mel:
        from cadencia.beat_alignment import load_prior
        _,alignment_prior_sha256 = load_prior()
    sampler = EventSampler(datasets[0],args.empty_keep,args.near_rows,args.seed)
    print(f'device={device} amp={args.amp} batch_size={args.batch_size}', flush=True)
    for name, dataset in zip(['train','validation','test'], datasets):
        print(f'{name}: charts={len(dataset.entries)} full_grid_rows={len(dataset)}', flush=True)
    print(f'training sampled_rows_per_epoch≈{len(sampler)} empty_keep={args.empty_keep} near_rows={args.near_rows}', flush=True)
    datasets[0].sampling_always, datasets[0].empty_keep = sampler.always, args.empty_keep
    loaders = []
    for i, dataset in enumerate(datasets):
        if not len(dataset):
            parser.error('All three dataset splits must be nonempty')
        options = dict(num_workers=args.workers if i==0 else 0, pin_memory=device.type=='cuda', collate_fn=packed_collate)
        if options['num_workers']:
            options.update(persistent_workers=True,multiprocessing_context='spawn')
        loaders.append(DataLoader(dataset,batch_size=args.batch_size,sampler=sampler if i==0 else None,**options))
    model = model_class(datasets[0].config).to(device)
    pretraining_info = None
    if args.chart_pretraining:
        from cadencia.chart_pretraining import initialize_decoder
        chart_checkpoint = torch.load(args.chart_pretraining,map_location='cpu',weights_only=True)
        pretraining_info = initialize_decoder(model,chart_checkpoint,datasets[0].manifest)
        pretraining_info.update(path=str(Path(args.chart_pretraining).resolve()),sha256=hashlib.sha256(Path(args.chart_pretraining).read_bytes()).hexdigest())
        print(f'Initialized {len(pretraining_info["transferred_keys"])} tensors from chart pretraining; audio/event/planner parameters remain fresh',flush=True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4)
    if args.chart_pretraining and args.output in ('data/interval_placement_transformer.pt','data/planned_placement_transformer.pt'):
        args.output = 'data/pretrained_' + Path(args.output).name
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    dataset_hash = hashlib.sha256(datasets[0].path.read_bytes()).hexdigest()
    history, best, best_state, best_epoch = [], float('inf'), None, 0
    sampling = dict(empty_keep=args.empty_keep, near_rows=args.near_rows, seed=args.seed)
    if args.resume:
        saved = torch.load(args.resume,map_location='cpu',weights_only=True)
        if saved['format'] != expected_format or saved['dataset_sha256'] != dataset_hash or saved['sampling'] != sampling or asdict(PlacementConfig(**saved['config'])) != metadata_fn(model)['config']:
            raise ValueError('Resume dataset, sampling, or model configuration mismatch')
        if saved.get('planner_sha256') != (hashlib.sha256(Path(args.planner).read_bytes()).hexdigest() if args.planner else None):
            raise ValueError('Resume planner mismatch')
        if saved.get('alignment_prior_sha256') != alignment_prior_sha256:
            raise ValueError('Resume alignment prior mismatch')
        pretraining_info = saved.get('chart_pretraining')
        model.load_state_dict(saved['last_model'])
        optimizer.load_state_dict(saved['optimizer'])
        history, best, best_epoch, best_state = saved['history'], saved['best_loss'], saved['best_epoch'], saved['model']
        if len(history) >= args.epochs:
            parser.error('--epochs must exceed the completed epochs when resuming')
        sampler.epoch = len(history)
        torch.set_rng_state(saved['rng_state'])
        if args.device == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng_state'])
    def save(path, value):
        temp = path.with_suffix(path.suffix+'.tmp')
        torch.save(value,temp)
        temp.replace(path)
    for epoch in range(len(history)+1,args.epochs+1):
        start = time.perf_counter()
        train = run_epoch(model,loaders[0],device,optimizer,args.amp)
        validation = run_epoch(model,loaders[1],device,amp=args.amp)
        history.append(dict(epoch=epoch,train=train,validation=validation,seconds=time.perf_counter()-start))
        if validation['loss'] < best:
            best, best_epoch = validation['loss'], epoch
            best_state = copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
        checkpoint = dict(**metadata_fn(model),model=best_state,mert=datasets[0].manifest['mert'],dataset_sha256=dataset_hash,
                          planner=planner_checkpoint,planner_sha256=hashlib.sha256(Path(args.planner).read_bytes()).hexdigest() if args.planner else None,
                          alignment_prior_sha256=alignment_prior_sha256,chart_pretraining=pretraining_info,sampling=sampling,epochs_completed=len(history),best_epoch=best_epoch,best_loss=best,
                          splits={name:[e['chart_id'] for e in ds.entries] for name,ds in zip(['train','validation','test'],datasets)})
        save(output,checkpoint)
        save(output.with_suffix('.last.pt'),dict(**checkpoint,last_model=model.state_dict(),optimizer=optimizer.state_dict(),history=history,
                                               rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all() if args.device=='cuda' else []))
        metrics = dict(chart_pretraining=pretraining_info,history=history,epochs_completed=len(history),best_epoch=best_epoch,sampling=sampling,dataset_sha256=dataset_hash)
        temp = output.with_suffix('.metrics.json.tmp')
        temp.write_text(json.dumps(metrics,indent=2)); temp.replace(output.with_suffix('.metrics.json'))
        print(f'epoch={epoch} train={train["loss"]:.4f} validation={validation["loss"]:.4f} seconds={history[-1]["seconds"]:.1f}',flush=True)
    model.load_state_dict(best_state)
    metrics['test'] = run_epoch(model,loaders[2],device,amp=args.amp)
    temp = output.with_suffix('.metrics.json.tmp')
    temp.write_text(json.dumps(metrics,indent=2)); temp.replace(output.with_suffix('.metrics.json'))


if __name__ == '__main__':
    main()
