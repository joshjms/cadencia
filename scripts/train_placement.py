"""Train the four-lane event decoder on cached frozen MERT features."""

import argparse
import copy
from dataclasses import replace
import json
import hashlib
from contextlib import nullcontext
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cadencia.placement import ACTIONS, PATTERN_COUNT, PlacementTransformer, mask_logits, model_metadata
from cadencia.placement import encode_patterns, decode_patterns, checkpoint_config
from cadencia.placement_data import PackedPlacementDataset, BlockShuffleSampler, packed_collate


def run_epoch(model, loader, device, weights, optimizer=None, max_batches=None, *, amp=False, log_every=200):
    model.train(optimizer is not None)
    total, normalizer, count = 0., 0., 0
    confusion = torch.zeros(6, 6, dtype=torch.long, device=device)
    chord_confusion = torch.zeros(5, 5, dtype=torch.long, device=device)
    exact_patterns = torch.zeros((),dtype=torch.long,device=device)
    started = time.perf_counter()
    with torch.set_grad_enabled(optimizer is not None):
        for number, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            target = batch.pop('target')
            context = torch.autocast('cuda', dtype=torch.bfloat16) if amp else nullcontext()
            with context:
                logits = mask_logits(model(**batch), batch['condition'][:, 4:].bool())
                joint = logits.ndim == 2
                classes = encode_patterns(target) if joint else target.reshape(-1)
                if joint and weights is not None and weights.numel() != PATTERN_COUNT:
                    raise ValueError('Joint training requires one weight per complete pattern')
                flat_logits = logits if joint else logits.reshape(-1,6)
                weighted_sum = nn.functional.cross_entropy(flat_logits, classes, weight=weights, reduction='sum')
                denominator = weights[classes].sum() if weights is not None else torch.tensor(classes.numel(),device=device)
                loss = weighted_sum / denominator
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite placement loss')
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            total += weighted_sum.detach().float().item()
            normalizer += denominator.detach().float().item()
            count += len(target)
            predicted = decode_patterns(logits.argmax(-1)) if joint else logits.argmax(-1)
            exact_patterns += (predicted == target).all(-1).sum()
            actual_heads = ((target == 1) | (target == 2) | (target == 4) | (target == 5)).sum(-1)
            predicted_heads = ((predicted == 1) | (predicted == 2) | (predicted == 4) | (predicted == 5)).sum(-1)
            chord_confusion += torch.bincount(actual_heads * 5 + predicted_heads,minlength=25).reshape(5,5)
            pairs = (target * 6 + predicted).detach().ravel()
            confusion += torch.bincount(pairs, minlength=36).reshape(6, 6)
            if log_every and (number + 1) % log_every == 0:
                print(f"  {'train' if optimizer is not None else 'eval'} batches={number+1}/{len(loader)} rows={count} seconds={time.perf_counter()-started:.1f}", flush=True)
            if max_batches is not None and number + 1 >= max_batches:
                break
    if not count:
        raise ValueError('Empty data split')
    confusion = confusion.cpu()
    metrics = []
    for i, action in enumerate(ACTIONS):
        tp = int(confusion[i, i]); predicted = int(confusion[:, i].sum()); actual = int(confusion[i].sum())
        metrics.append(dict(action=action, precision=tp / max(1, predicted), recall=tp / max(1, actual),
                            f1=2 * tp / max(1, predicted + actual), support=actual))
    return dict(loss=total / normalizer, rows=count, per_action=metrics, confusion=confusion.tolist(),
                prediction_head='joint_patterns' if joint else 'independent_lanes',
                pattern_accuracy=float(exact_patterns.cpu()) / count,
                chord_size_confusion=chord_confusion.cpu().tolist(),
                target_chord_counts=chord_confusion.sum(1).cpu().tolist(),
                predicted_chord_counts=chord_confusion.sum(0).cpu().tolist(),
                evaluation='Teacher-forced history on fixed-grid queries; not free-generation quality')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/placement_dataset')
    parser.add_argument('--output', help='Defaults to a separate joint_placement_transformer.pt for new runs')
    parser.add_argument('--head', choices=['joint','independent'], help='New runs default to joint; resume preserves the saved head')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--workers', type=int, default=2, help='Background training data-loader processes')
    parser.add_argument('--amp', action='store_true', help='Use CUDA bfloat16 autocast (FP32 model weights)')
    parser.add_argument('--compile', action='store_true', help='Compile the GPU model with Torch Inductor')
    parser.add_argument('--resume', help='Resume an epoch checkpoint including optimizer and RNG state')
    parser.add_argument('--initialize-from', help='Start a new run from checkpoint weights, with fresh optimizer and dataset splits')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-batches', type=int, help='Limit each split per epoch for smoke tests')
    args = parser.parse_args()
    if args.resume and args.initialize_from:
        parser.error('--resume and --initialize-from are mutually exclusive')
    if args.epochs < 1 or args.batch_size < 1 or (args.max_batches is not None and args.max_batches < 1):
        parser.error('epochs, batch-size and max-batches must be positive')
    if args.workers < 0:
        parser.error('workers must be nonnegative')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    if args.amp and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        parser.error('--amp requires CUDA bfloat16 support')
    print(f'device={device} amp={args.amp} batch_size={args.batch_size}', flush=True)
    datasets = [PackedPlacementDataset(args.data, name) for name in ['train', 'validation', 'test']]
    if any(not len(dataset) for dataset in datasets):
        parser.error('Training requires at least three independent audio/set groups and nonempty splits')
    loaders = []
    for i, dataset in enumerate(datasets):
        workers = args.workers if i == 0 else 0
        options = dict(num_workers=workers, pin_memory=device.type == 'cuda', collate_fn=packed_collate)
        if workers:
            options.update(persistent_workers=True, multiprocessing_context='spawn', prefetch_factor=2)
        loaders.append(DataLoader(dataset, batch_size=args.batch_size,
                                  sampler=BlockShuffleSampler(dataset) if i == 0 else None, **options))
    for name, dataset in zip(['train','validation','test'], datasets):
        print(f'{name}: charts={len(dataset.entries)} rows={len(dataset)}', flush=True)
    initial = torch.load(args.initialize_from,map_location='cpu',weights_only=True) if args.initialize_from else None
    saved = torch.load(args.resume,map_location='cpu',weights_only=True) if args.resume else None
    if saved is not None and 'last_model' not in saved:
        raise ValueError('Resume requires a .last.pt checkpoint with optimizer state')
    saved_config = checkpoint_config(saved if saved is not None else initial) if saved is not None or initial is not None else None
    joint = saved_config.joint_patterns if saved_config is not None else args.head != 'independent'
    if args.head is not None and (args.head == 'joint') != joint:
        parser.error('Requested head does not match resume checkpoint')
    model_config = replace(datasets[0].config,joint_patterns=joint)
    if saved_config is not None and saved_config != model_config:
        raise ValueError('Resume model configuration does not match dataset preprocessing configuration')
    print(f"prediction_head={'joint_patterns' if joint else 'independent_lanes'}",flush=True)
    model = PlacementTransformer(model_config).to(device)
    raw_model = model
    if initial is not None:
        model.load_state_dict(initial['model'], strict=True)
        print(f'Initialized weights from {args.initialize_from}; starting new optimizer and epoch count', flush=True)
    if joint:
        # Preserve the empirical pattern/chord priors. Reweighting rare patterns
        # would encourage exactly the inflated chord frequencies we want to avoid.
        weights = torch.ones(PATTERN_COUNT,dtype=torch.float32,device=device)
    else:
        counts = np.array([entry['action_counts'] for entry in datasets[0].entries]).sum(axis=0)
        weights = 1 / np.sqrt(np.maximum(counts, 1))
        weights /= weights.mean()
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    best, best_state, best_epoch = float('inf'), None, 0
    history = []
    if args.output:
        output = Path(args.output)
    elif args.resume:
        previous = Path(args.resume)
        output = previous.with_name(previous.name.removesuffix('.last.pt')+'.pt')
    else:
        output = Path('data/joint_placement_transformer.pt' if joint else 'data/placement_transformer.pt')
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_hash = hashlib.sha256(datasets[0].path.read_bytes()).hexdigest()
    start_epoch = 1
    if args.resume:
        if saved['dataset_sha256'] != manifest_hash:
            raise ValueError('Resume dataset/config does not match')
        model.load_state_dict(saved['last_model'])
        optimizer.load_state_dict(saved['optimizer'])
        best, best_epoch, best_state = saved['best_loss'], saved['best_epoch'], saved['model']
        history = saved['history']
        start_epoch = saved['epochs_completed'] + 1
        torch.set_rng_state(saved['rng_state'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng_state'])
    def save_checkpoint(extra=None):
        checkpoint = dict(**model_metadata(raw_model), model=best_state, mert=datasets[0].manifest['mert'],
                          difficulty_units='osu!mania star rating', seed=args.seed,
                          initialized_from=str(Path(args.initialize_from).resolve()) if args.initialize_from else None,
                          epochs_completed=len(history), best_epoch=best_epoch, best_loss=best,
                          class_weights=weights.cpu(), dataset_sha256=manifest_hash,
                          device=str(device), gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
                          torch_version=str(torch.__version__), amp=args.amp, compiled=args.compile, batch_size=args.batch_size,
                          splits={name: [entry['chart_id'] for entry in dataset.entries]
                                  for name, dataset in zip(['train','validation','test'],datasets)})
        temporary = output.with_suffix('.pt.tmp')
        torch.save(checkpoint, temporary)
        temporary.replace(output)
        last = dict(**checkpoint, last_model={k:v.detach().cpu() for k,v in raw_model.state_dict().items()},
                    optimizer=optimizer.state_dict(), history=history, rng_state=torch.get_rng_state(),
                    cuda_rng_state=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
        temporary = output.with_suffix('.last.pt.tmp')
        torch.save(last,temporary)
        temporary.replace(output.with_suffix('.last.pt'))
        metrics = dict(history=history, max_batches=args.max_batches, dataset_sha256=manifest_hash,
                       device=str(device), gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
                          torch_version=str(torch.__version__), amp=args.amp, compiled=args.compile, batch_size=args.batch_size, epochs_completed=len(history),
                       requested_epochs=args.epochs, best_epoch=best_epoch, prediction_head='joint_patterns' if joint else 'independent_lanes')
        metrics.update(extra or {})
        temporary = output.with_suffix('.metrics.json.tmp')
        temporary.write_text(json.dumps(metrics,indent=2))
        temporary.replace(output.with_suffix('.metrics.json'))
    if args.compile:
        model = torch.compile(raw_model, dynamic=True)
    for epoch in range(start_epoch, args.epochs + 1):
        start = time.perf_counter()
        train = run_epoch(model, loaders[0], device, weights, optimizer, args.max_batches, amp=args.amp)
        validation = run_epoch(model, loaders[1], device, weights, max_batches=args.max_batches, amp=args.amp)
        history.append(dict(epoch=epoch, train=train, validation=validation, seconds=time.perf_counter() - start))
        print(f"epoch={epoch} train_loss={train['loss']:.4f} validation_loss={validation['loss']:.4f} seconds={history[-1]['seconds']:.1f}", flush=True)
        if validation['loss'] < best:
            best, best_epoch = validation['loss'], epoch
            best_state = copy.deepcopy({k: v.cpu() for k, v in raw_model.state_dict().items()})
        save_checkpoint()
    # Final evaluation uses the best validation checkpoint; retain last-epoch
    # optimizer/RNG state in .last.pt before switching the model for test.
    raw_model.load_state_dict(best_state)
    test = run_epoch(model, loaders[2], device, weights, max_batches=args.max_batches, amp=args.amp)
    metrics_path = output.with_suffix('.metrics.json')
    metrics = json.loads(metrics_path.read_text())
    metrics['test'] = test
    temporary = metrics_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(metrics,indent=2))
    temporary.replace(metrics_path)
    print(f'Saved {output} (best epoch {best_epoch})', flush=True)


if __name__ == '__main__':
    main()
