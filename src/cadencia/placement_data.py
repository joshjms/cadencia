"""Four-lane action-grid targets and lazy, leakage-free history training samples."""

from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import logging
import os
import errno
from pathlib import Path
import tempfile

import numpy as np
from torch.utils.data import Dataset

from .dataset import _assign_splits
from .mert import FrozenMert
from .osumania.chart import Chart
from .osumania.events import Hold
from .osumania.loader import data_root
from .placement import ACTIONS, EMPTY, TAP, HOLD_START, HOLD_END, RELEASE_TAP, RELEASE_HOLD, PlacementConfig

SCHEMA = 'cadencia.placement_dataset.v1'
log = logging.getLogger(__name__)


def tempo_at(times, timing_map):
    if not timing_map:
        raise ValueError('Chart has no usable tempo map')
    starts = np.array([point['start_s'] for point in timing_map])
    indices = np.maximum(0, np.searchsorted(starts, times, side='right') - 1)
    bpms = np.array([point['bpm'] for point in timing_map])[indices]
    origin = np.array([point['beat_position'] for point in timing_map])[indices]
    beats = origin + (times - starts[indices]) * bpms / 60
    if not np.isfinite(bpms).all() or (bpms <= 0).any() or not np.isfinite(beats).all():
        raise ValueError('Invalid tempo map')
    return bpms.astype(np.float32), beats.astype(np.float32)


def advance_held(held, actions):
    result = np.array(held, dtype=bool, copy=True)
    for lane, action in enumerate(actions):
        if action == HOLD_START or action == RELEASE_HOLD:
            result[lane] = True
        elif action == HOLD_END or action == RELEASE_TAP:
            result[lane] = False
    return result


def chart_targets(chart, *, grid_hz=50):
    """Quantize heads/releases to the requested grid; reject collisions."""
    times = np.arange(int(np.ceil(chart.song.duration_s * grid_hz))) / grid_hz
    times = times[times < chart.song.duration_s]
    if len(times) < 2:
        raise ValueError('Song is too short')
    starts, releases = {}, set()
    for event in chart.events:
        if not event.analysis_eligible:
            continue
        start = min(len(times) - 1, max(0, int(np.floor(event.start_s * grid_hz + .5))))
        key = (start, event.lane)
        if key in starts:
            raise ValueError(f'Multiple heads in one grid lane cell: {key}')
        starts[key] = HOLD_START if isinstance(event, Hold) else TAP
        if isinstance(event, Hold):
            end = min(len(times) - 1, int(np.floor(event.end_s * grid_hz + .5)))
            if end <= start:
                raise ValueError('Hold is shorter than the grid resolution')
            if (end, event.lane) in releases:
                raise ValueError('Duplicate hold release')
            releases.add((end, event.lane))
    actions = np.zeros((len(times), 4), dtype=np.int64)
    for (index, lane), action in starts.items():
        actions[index, lane] = action
    for index, lane in releases:
        action = actions[index, lane]
        actions[index, lane] = {EMPTY: HOLD_END, TAP: RELEASE_TAP, HOLD_START: RELEASE_HOLD}[int(action)]
    states, held = [], np.zeros(4, dtype=bool)
    for row in actions:
        states.append(held.copy())  # State BEFORE this target, never after it.
        for lane, action in enumerate(row):
            if action in (TAP, HOLD_START) and held[lane]:
                raise ValueError('Overlapping notes on an active hold lane')
            if action in (HOLD_END, RELEASE_TAP, RELEASE_HOLD) and not held[lane]:
                raise ValueError('Release without active hold')
        held = advance_held(held, row)
    if held.any():
        raise ValueError('Unclosed hold')
    bpm, beats = tempo_at(times, chart.timing_map)
    groups = np.flatnonzero(actions.any(axis=1))
    return dict(times=times, actions=actions, held=np.asarray(states), bpm=bpm, beats=beats, groups=groups)


def make_sample(target_time, target_beat, bpm, difficulty, held, history_actions,
                history_times, history_beats, features, config):
    """Shared training/inference packing; history must strictly precede target."""
    if len(history_times) and np.any(np.asarray(history_times) >= target_time):
        raise ValueError('History includes the current or a future target')
    h = config.history_events
    count = min(h, len(history_times))
    history = np.zeros((h, 4), dtype=np.int64)
    timing = np.zeros((h, 4), dtype=np.float32)
    padding = np.ones(h, dtype=bool)
    if count:
        history[:count] = np.asarray(history_actions)[-count:]
        ages = np.asarray(history_times)[-count:] - target_time
        beat_gaps = np.asarray(history_beats)[-count:] - target_beat
        timing[:count] = np.column_stack([ages / 10, beat_gaps / 16,
                                         np.sin(beat_gaps * 2 * np.pi), np.cos(beat_gaps * 2 * np.pi)])
        padding[:count] = False
    memory_size = int(config.audio_window_s * config.audio_hz) + 2
    audio = np.zeros((memory_size, config.audio_dim), dtype=np.float32)
    audio_times = np.zeros(memory_size, dtype=np.float32)
    audio_padding = np.ones(memory_size, dtype=bool)
    stamps = features['times']
    lo, hi = np.searchsorted(stamps, [target_time - config.audio_window_s / 2,
                                     target_time + config.audio_window_s / 2])
    if lo == hi:
        lo = min(len(stamps) - 1, lo)
        hi = lo + 1
    n = hi - lo
    if n > memory_size:
        raise ValueError('Audio feature rate exceeds configured frame rate')
    audio[:n] = features['features'][lo:hi]
    audio_times[:n] = stamps[lo:hi] - target_time
    audio_padding[:n] = False
    condition = np.array([difficulty / 10, bpm / 240,
                          np.sin(target_beat * 2 * np.pi), np.cos(target_beat * 2 * np.pi),
                          *np.asarray(held, dtype=np.float32)], dtype=np.float32)
    extras = {}
    if config.use_mel:
        from .high_resolution import mel_windows
        extras = {k:v[0] for k,v in mel_windows(features['mel'],[target_time]).items()}
    if config.use_rhythm:
        from .rhythm import rhythm_at
        extras['rhythm']=rhythm_at(features['rhythm'],[target_time])[0]
    return dict(**extras,history=history, history_timing=timing, history_padding=padding,
                condition=condition, audio=audio, audio_times=audio_times, audio_padding=audio_padding)


def difficulty_for(chart, labels):
    value = labels.get(str(chart.id), chart.metadata.get('star_rating'))
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f'Chart {chart.id} needs an explicit positive star rating in --difficulty-labels')
    return float(value)


def build_placement_dataset(output, *, data_dir=None, difficulty_labels=None, extractor=None, config=None, seed=42):
    output, root = Path(output).resolve(), data_root(data_dir)
    if output.exists():
        raise FileExistsError(output)
    extractor = extractor or FrozenMert()
    config = config or PlacementConfig(audio_dim=extractor.config.feature_dim,audio_hz=extractor.config.frame_hz)
    if (config.audio_dim,config.audio_hz)!=(extractor.config.feature_dim,extractor.config.frame_hz):
        raise ValueError('Dataset model configuration disagrees with audio encoder')
    labels = difficulty_labels or {}
    rating_cache = root / 'star_rating_cache.json'
    rating_sources = json.loads(rating_cache.read_text()).get('ratings', {}) if rating_cache.exists() else {}
    traces = sorted((root / 'osumania/charts').glob('*/trace.json'))
    if not traces:
        raise FileNotFoundError('No cached charts')
    output.parent.mkdir(parents=True, exist_ok=True)
    entries, failures = [], []
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.placement-') as temporary:
        staging = Path(temporary)
        (staging / 'audio').mkdir()
        for trace in traces:
            try:
                chart = Chart.read(trace)
                difficulty = difficulty_for(chart, labels)
                targets = chart_targets(chart, grid_hz=config.grid_hz)
                features, cache = extractor.extract(chart.song.path, cache_dir=root / 'mert_cache')
                if abs(float(features['duration_s']) - chart.song.duration_s) > .05:
                    raise ValueError('Feature duration does not match chart')
                audio_path = staging / 'audio' / cache.name
                feature_path = audio_path.with_suffix('.npy')
                if not audio_path.exists():
                    # Metadata is tiny; the large matrix stays uncompressed and
                    # memory-mapped so random samples never decompress a song.
                    cached_matrix = cache.with_suffix('.npy')
                    if cached_matrix.is_file():
                        try:
                            os.link(cached_matrix, feature_path)
                        except OSError as error:
                            if error.errno != errno.EXDEV:
                                raise
                            np.save(feature_path, features['features'], allow_pickle=False)
                    else:
                        np.save(feature_path, features['features'], allow_pickle=False)
                    np.savez_compressed(audio_path, **{k: features[k] for k in ('times', 'duration_s', 'identity')})
                mel_entry = {}
                if config.use_mel:
                    from .high_resolution import extract_mel
                    _,mel_cache = extract_mel(chart.song.path,root/'mel_cache')
                    mel_path = staging/'audio'/mel_cache.name
                    if not mel_path.exists():
                        try: os.link(mel_cache,mel_path)
                        except OSError as error:
                            if error.errno!=errno.EXDEV: raise
                            import shutil
                            shutil.copyfile(mel_cache,mel_path)
                    mel_entry=dict(mel_file=f'audio/{mel_path.name}',mel_sha256=hashlib.sha256(mel_path.read_bytes()).hexdigest())
                if config.use_rhythm:
                    from .rhythm import analyze_rhythm
                    _,rhythm_cache=analyze_rhythm(chart.song.path,root/'rhythm_cache',device=extractor.device)
                    rhythm_path=staging/'audio'/rhythm_cache.name
                    if not rhythm_path.exists():
                        try:os.link(rhythm_cache,rhythm_path)
                        except OSError as error:
                            if error.errno!=errno.EXDEV:raise
                            import shutil
                            shutil.copyfile(rhythm_cache,rhythm_path)
                    mel_entry.update(rhythm_file=f'audio/{rhythm_path.name}',rhythm_sha256=hashlib.sha256(rhythm_path.read_bytes()).hexdigest())
                shard = staging / f'{chart.id}.npz'
                np.savez_compressed(shard, **targets)
                counts = np.bincount(targets['actions'].ravel(), minlength=6).tolist()
                entries.append(dict(**mel_entry,chart_id=chart.id, song_id=int(chart.metadata['beatmapset_id']),
                                    audio_sha256=chart.song.sha256, difficulty=difficulty,
                                    difficulty_source=rating_sources.get(str(chart.id), {}).get('source', chart.metadata.get('star_rating_source', 'explicit label')),
                                    file=shard.name, audio_file=f'audio/{cache.name}',
                                    feature_file=f'audio/{feature_path.name}', rows=len(targets['times']),
                                    action_counts=counts, trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                                    sha256=hashlib.sha256(shard.read_bytes()).hexdigest(),
                                    audio_file_sha256=hashlib.sha256(audio_path.read_bytes()).hexdigest(),
                                    feature_sha256=hashlib.sha256(feature_path.read_bytes()).hexdigest()))
                log.info('Prepared chart %s: %s grid positions', chart.id, len(targets['times']))
            except Exception as error:
                failures.append(dict(trace=str(trace), error=f'{type(error).__name__}: {error}'))
                log.warning('Skipping %s: %s', trace, error)
        if not entries:
            raise ValueError('No usable charts. Supply real star ratings via --difficulty-labels. '
                             + (failures[0]['error'] if failures else ''))
        _assign_splits(entries, seed)
        manifest = dict(schema=SCHEMA, config=asdict(config), mert=asdict(extractor.config), seed=seed,
                        actions=ACTIONS, charts=entries, failures=failures,
                        difficulty_units='osu!mania star rating (explicit labels, never OverallDifficulty)',
                        timing=f'{config.grid_hz} Hz regular grid, nearest {1000/config.grid_hz:g} ms, no target timestamps in candidates')
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        staging.rename(output)
    return manifest


class PlacementDataset(Dataset):
    def __init__(self, path, split):
        self.path = Path(path)
        if self.path.is_dir():
            self.path /= 'manifest.json'
        manifest = json.loads(self.path.read_text())
        if manifest.get('schema') != SCHEMA or manifest.get('actions') != ACTIONS:
            raise ValueError('Unsupported placement dataset')
        self.manifest = manifest
        self.config = PlacementConfig(**manifest['config'])
        self.entries = [entry for entry in manifest['charts'] if entry['split'] == split]
        self.ends = np.cumsum([entry['rows'] for entry in self.entries])
        # Verify each file once on load, including duplicated audio references.
        checked = set()
        for entry in self.entries:
            for key, checksum in [('file', 'sha256'), ('audio_file', 'audio_file_sha256'), ('feature_file', 'feature_sha256')] + ([('mel_file','mel_sha256')] if self.config.use_mel else []) + ([('rhythm_file','rhythm_sha256')] if self.config.use_rhythm else []):
                file = (self.path.parent / entry[key]).resolve()
                if not file.is_relative_to(self.path.parent.resolve()):
                    raise ValueError('Dataset file escapes its root')
                if file not in checked:
                    if hashlib.sha256(file.read_bytes()).hexdigest() != entry[checksum]:
                        raise ValueError(f'Dataset checksum mismatch: {file}')
                    checked.add(file)

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    @lru_cache(maxsize=32)
    def _load(self, index):
        entry = self.entries[index]
        with np.load(self.path.parent / entry['file'], allow_pickle=False) as arrays:
            target = {k: arrays[k] for k in arrays.files}
        with np.load(self.path.parent / entry['audio_file'], allow_pickle=False) as arrays:
            features = {k: arrays[k] for k in arrays.files}
        features['features'] = np.load(self.path.parent / entry['feature_file'], mmap_mode='r', allow_pickle=False)
        if features['features'].shape != (len(features['times']), self.config.audio_dim):
            raise ValueError('Invalid MERT feature matrix shape')
        if self.config.use_mel:
            features['mel']=np.load(self.path.parent/entry['mel_file'],mmap_mode='r',allow_pickle=False)
        if self.config.use_rhythm:
            from .rhythm import read_rhythm
            features['rhythm']=read_rhythm(self.path.parent/entry['rhythm_file'])
        return target, features

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        chart_index = int(np.searchsorted(self.ends, index, side='right'))
        row = index - (int(self.ends[chart_index - 1]) if chart_index else 0)
        targets, features = self._load(chart_index)
        groups = targets['groups']
        end = np.searchsorted(groups, row, side='left')
        history = groups[max(0, end - self.config.history_events):end]
        sample = make_sample(targets['times'][row], targets['beats'][row], targets['bpm'][row],
                             self.entries[chart_index]['difficulty'], targets['held'][row],
                             targets['actions'][history], targets['times'][history], targets['beats'][history],
                             features, self.config)
        return {**sample, 'target': targets['actions'][row].copy()}


class BlockShuffleSampler:
    """Visit every row once, shuffling short contiguous blocks for audio reuse."""

    def __init__(self, dataset, block_size=32):
        self.size, self.block_size = len(dataset), block_size

    def __len__(self):
        return self.size

    def __iter__(self):
        import torch
        blocks = (self.size + self.block_size - 1) // self.block_size
        for block in torch.randperm(blocks).tolist():
            yield from range(block * self.block_size, min(self.size, (block + 1) * self.block_size))


class PackedPlacementDataset(PlacementDataset):
    """Batch input packing with shared audio frames, mathematically identical targets.

    Nearby queries share most MERT frames. Transfer/project each unique frame
    only once per batch, then gather the local windows inside the model.
    """

    def __getitems__(self, indices):
        import torch

        indices = np.asarray(indices, dtype=np.int64)
        if (indices < 0).any() or (indices >= len(self)).any():
            raise IndexError('Batch index outside dataset')
        b, h = len(indices), self.config.history_events
        memory_size = int(self.config.audio_window_s * self.config.audio_hz) + 2
        result = dict(history=np.zeros((b,h,4),np.int64),
                      history_timing=np.zeros((b,h,4),np.float32),
                      history_padding=np.ones((b,h),bool),
                      condition=np.zeros((b,8),np.float32),
                      audio_times=np.zeros((b,memory_size),np.float32),
                      audio_padding=np.ones((b,memory_size),bool),
                      audio_indices=np.zeros((b,memory_size),np.int64),
                      target=np.zeros((b,4),np.int64))
        chart_indices = np.searchsorted(self.ends, indices, side='right')
        memories, offset = [], 0
        for chart_index in np.unique(chart_indices):
            selected = np.flatnonzero(chart_indices == chart_index)
            rows = indices[selected] - (self.ends[chart_index-1] if chart_index else 0)
            target, features = self._load(int(chart_index))
            times, beats = target['times'][rows], target['beats'][rows]
            groups = target['groups']
            if len(groups):
                end = np.searchsorted(groups, rows, side='left')
                counts = np.minimum(end, h)
                columns = np.arange(h)[None,:]
                valid = columns < counts[:,None]
                positions = np.clip(end[:,None] - counts[:,None] + columns, 0, len(groups)-1)
                history = groups[positions]
                result['history'][selected] = np.where(valid[...,None],target['actions'][history],0)
                ages = target['times'][history] - times[:,None]
                gaps = target['beats'][history] - beats[:,None]
                timing = np.stack([ages/10,gaps/16,np.sin(gaps*2*np.pi),np.cos(gaps*2*np.pi)],axis=-1)
                result['history_timing'][selected] = np.where(valid[...,None],timing,0)
                result['history_padding'][selected] = ~valid
            result['target'][selected] = target['actions'][rows]
            result['condition'][selected,0] = self.entries[int(chart_index)]['difficulty'] / 10
            result['condition'][selected,1] = target['bpm'][rows] / 240
            result['condition'][selected,2] = np.sin(beats * 2 * np.pi)
            result['condition'][selected,3] = np.cos(beats * 2 * np.pi)
            result['condition'][selected,4:] = target['held'][rows]
            stamps = features['times']
            lo = np.searchsorted(stamps,times-self.config.audio_window_s/2)
            hi = np.searchsorted(stamps,times+self.config.audio_window_s/2)
            empty = lo == hi
            lo[empty] = np.minimum(lo[empty],len(stamps)-1)
            hi[empty] = lo[empty]+1
            counts = hi-lo
            if (counts>memory_size).any():
                raise ValueError('Audio feature rate exceeds configured frame rate')
            valid = np.arange(memory_size)[None,:] < counts[:,None]
            raw = np.minimum(lo[:,None]+np.arange(memory_size)[None,:],len(stamps)-1)
            unique, inverse = np.unique(raw[valid],return_inverse=True)
            packed_indices = np.zeros_like(raw)
            packed_indices[valid] = inverse+offset
            result['audio_indices'][selected] = packed_indices
            result['audio_times'][selected] = np.where(valid,stamps[raw]-times[:,None],0)
            result['audio_padding'][selected] = ~valid
            memories.append(np.asarray(features['features'][unique]))
            offset += len(unique)
        if self.config.use_mel:
            from .high_resolution import mel_windows
            result['mel']=np.zeros((b,101,64),np.float32)
            result['mel_padding']=np.ones((b,101),bool)
            for chart_index in np.unique(chart_indices):
                selected=np.flatnonzero(chart_indices==chart_index)
                rows=indices[selected]-(self.ends[chart_index-1] if chart_index else 0)
                target,features=self._load(int(chart_index))
                for key,value in mel_windows(features['mel'],target['times'][rows]).items():result[key][selected]=value
        if self.config.use_rhythm:
            from .rhythm import rhythm_at
            result['rhythm']=np.zeros((b,6),np.float32)
            for chart_index in np.unique(chart_indices):
                selected=np.flatnonzero(chart_indices==chart_index)
                rows=indices[selected]-(self.ends[chart_index-1] if chart_index else 0)
                target,features=self._load(int(chart_index))
                result['rhythm'][selected]=rhythm_at(features['rhythm'],target['times'][rows])
        result['audio'] = np.concatenate(memories)
        return {key:torch.from_numpy(value) for key,value in result.items()}


def packed_collate(batch):
    """DataLoader's batched dataset fetch already produced the final tensors."""
    return batch
