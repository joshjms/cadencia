"""Publish an immutable inference bundle with an explicit best/last weight selection."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import torch
from cadencia.model_versions import ROOT, SCHEMA, is_model_version


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inference_checkpoint(checkpoint, version, weights='best'):
    """Exclude resumable state and make generation load the explicitly selected weights."""
    if weights not in ('best', 'last'):
        raise ValueError('Weights must be best or last')
    if weights == 'last' and 'last_model' not in checkpoint:
        raise ValueError('Last weights require a resumable .last.pt checkpoint')
    selected = checkpoint.get('epochs_completed') if weights == 'last' else checkpoint['best_epoch']
    excluded = {'last_model', 'optimizer', 'scheduler', 'history', 'rng_state', 'cuda_rng_state',
                'initialization', 'baseline_validation'}
    result = {k: v for k, v in checkpoint.items() if k not in excluded}
    if weights == 'last':
        result['model'] = checkpoint['last_model']
    result.update(architecture_version=version, weights_epoch=selected, weights_selection=weights)
    if checkpoint.get('initialization'):
        result['initialization_sha256'] = checkpoint['initialization']['sha256']
    if result.get('chart_pretraining'):
        result['chart_pretraining'] = {k: v for k, v in result['chart_pretraining'].items() if k != 'path'}
    return result


def publish(version, decoder_path, planner_path, pretraining_path, *, weights='best'):
    if not is_model_version(version):
        raise ValueError('Invalid version tag')
    destination = ROOT / 'model_releases' / version
    registry = ROOT / 'model_releases' / f'{version}.json'
    if destination.exists() or registry.exists():
        raise FileExistsError('Release already exists; choose a new version')
    destination.parent.mkdir(parents=True, exist_ok=True)
    sources = dict(decoder=Path(decoder_path), planner=Path(planner_path), chart_pretraining=Path(pretraining_path))
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        staging = Path(directory)
        checkpoints, artifacts, source_hashes = {}, {}, {}
        # Training files are replaced atomically. Copy each once before reading it.
        for name, source in sources.items():
            target = staging / f'{name}.pt'
            shutil.copyfile(source, target)
            source_hashes[name] = digest(target)
            checkpoints[name] = torch.load(target, map_location='cpu', weights_only=True)
        decoder = checkpoints['decoder']
        if (decoder.get('planner_sha256') != source_hashes['planner'] or
                decoder.get('chart_pretraining', {}).get('sha256') != source_hashes['chart_pretraining']):
            raise ValueError('Decoder provenance does not match supplied planner/pretraining checkpoints')
        if decoder['mert'] != checkpoints['planner']['mert']:
            raise ValueError('Encoder configurations disagree')
        # Reserialize auxiliary checkpoints under canonical archive names rather
        # than carrying machine-specific training filenames in ZIP headers.
        for name in ('planner', 'chart_pretraining'):
            torch.save(checkpoints[name], staging / f'{name}.pt')
        inference = inference_checkpoint(decoder, version, weights)
        inference['planner_sha256'] = digest(staging / 'planner.pt')
        if inference.get('chart_pretraining'):
            inference['chart_pretraining']['sha256'] = digest(staging / 'chart_pretraining.pt')
        torch.save(inference, staging / 'decoder.pt')
        for name, checkpoint in checkpoints.items():
            target = staging / f'{name}.pt'
            artifacts[name] = dict(path=str((destination / target.name).relative_to(ROOT)),
                                   sha256=digest(target), format=checkpoint['format'],
                                   dataset_sha256=checkpoint['dataset_sha256'],
                                   source_checkpoint_sha256=source_hashes[name])
            if name == 'decoder':
                artifacts[name].update(weights=weights, weights_epoch=inference['weights_epoch'])
            else:
                artifacts[name]['best_epoch'] = checkpoint['best_epoch']
        if decoder.get('alignment_prior_sha256'):
            from cadencia.beat_alignment import DEFAULT_PRIOR
            prior = staging / 'alignment_prior.json'
            shutil.copyfile(DEFAULT_PRIOR, prior)
            if digest(prior) != decoder['alignment_prior_sha256']:
                raise ValueError('Alignment prior differs from decoder provenance')
            artifacts['alignment_prior'] = dict(path=str((destination / prior.name).relative_to(ROOT)), sha256=digest(prior))
        manifest = dict(schema=SCHEMA, status='released', version=version,
                        created_at=datetime.now(timezone.utc).isoformat(),
                        architecture='MERT-v2 + 10ms mel + tracked beats + bar planner + phrase-trained interval decoder',
                        encoder=decoder['mert'], rhythm_backend=decoder.get('rhythm_backend'),
                        phrase_training=decoder.get('phrase_training'), artifacts=artifacts,
                        notes='Immutable inference weights; resume training from a separate .last.pt checkpoint.')
        # Record the selected epoch's measurements without copying machine-specific paths.
        history = decoder.get('history', [])
        if history and inference['weights_epoch']:
            record = next((h for h in history if h['epoch'] == inference['weights_epoch']), None)
            if record:
                manifest['validation'] = record['validation']
        staging.rename(destination)
        try:
            with registry.open('x') as file:
                json.dump(manifest, file, indent=2)
                file.write('\n')
        except BaseException:
            shutil.rmtree(destination)
            raise
    return registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('version', help='Release name, e.g. v1 or v1.1.0')
    parser.add_argument('--decoder', type=Path, required=True)
    parser.add_argument('--planner', type=Path, required=True)
    parser.add_argument('--chart-pretraining', type=Path, required=True)
    parser.add_argument('--weights', choices=['best', 'last'], default='best',
                        help='Explicitly select protected best weights or latest completed training weights')
    args = parser.parse_args()
    print(publish(args.version, args.decoder, args.planner, args.chart_pretraining, weights=args.weights))


if __name__ == '__main__':
    main()
