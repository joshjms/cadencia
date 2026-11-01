"""Event timing, joint onset patterns, and explicitly supervised hold intervals."""
from dataclasses import asdict
import numpy as np
import torch
from torch import nn
from .placement import PlacementTransformer, PlacementConfig
from .placement_data import PackedPlacementDataset

FORMAT = 'cadencia.placement.intervals.v3'
# Base-three lanes: absent, tap, hold start. Empty is handled by the timing head.
PATTERNS = 3 ** 4


def merge_notes(notes):
    """Give hold intervals priority over same-lane taps; release boundaries are open."""
    holds = [[] for _ in range(4)]
    for note in sorted(notes, key=lambda n: n['time_s']):
        if note['kind'] == 'hold':
            lane = holds[note['lane']]
            if lane and note['time_s'] < lane[-1]['end_s']:
                raise ValueError('Overlapping holds on one lane')
            lane.append(note)
    result = []
    for note in sorted(notes, key=lambda n: n['time_s']):
        if note['kind'] == 'tap' and any(h['time_s'] <= note['time_s'] < h['end_s'] for h in holds[note['lane']]):
            continue
        result.append(note)
    return result


def decode(tokens):
    return tokens[..., None] // torch.tensor([1, 3, 9, 27], device=tokens.device) % 3


def encode(actions):
    return (actions * torch.tensor([1, 3, 9, 27], device=actions.device)).sum(-1)


class IntervalTransformer(PlacementTransformer):
    def __init__(self, config=None):
        super().__init__(config)
        del self.output
        self.event = nn.Linear(self.config.d_model, 1)
        self.pattern = nn.Linear(self.config.d_model, PATTERNS - 1)
        self.duration = nn.Linear(self.config.d_model, 4)

    def forward(self, **inputs):
        hidden = self.encode(**inputs)
        return dict(event=self.event(hidden).squeeze(-1), pattern=self.pattern(hidden),
                    duration=self.duration(hidden))


def metadata(model):
    return dict(format=FORMAT, config=asdict(model.config),
                pattern_encoding='base3: absent=0,tap=1,hold_start=2; IDs 1..80',
                duration_units='log1p(beats)', objective='event_bce+conditional_pattern_ce+hold_duration_smooth_l1')


def legal_pattern_logits(logits, held):
    patterns = decode(torch.arange(1, PATTERNS, device=logits.device))
    legal = ~((patterns[None] != 0) & held[:, None].bool()).any(-1)
    # No legal onset when all lanes are occupied. Caller must skip the event.
    return logits.masked_fill(~legal, float('-inf'))


def interval_targets(target):
    """Derive onsets and durations from validated action shards without rebuilding MERT."""
    actions = target['actions']
    onsets = np.zeros_like(actions)
    onsets[np.isin(actions, [1, 4])] = 1
    onsets[np.isin(actions, [2, 5])] = 2
    durations = np.zeros(actions.shape, np.float32)
    for lane in range(4):
        start = None
        for row, action in enumerate(actions[:, lane]):
            if action in (3, 4, 5):
                if start is None:
                    raise ValueError('Hold release without start')
                # Full timing-map beat positions account for tempo changes.
                beats = float(target['beats'][row] - target['beats'][start])
                if beats <= 0:
                    raise ValueError('Nonpositive hold duration')
                durations[start, lane] = np.log1p(beats)
                start = None
            if action in (2, 5):
                start = row
        if start is not None:
            raise ValueError('Unclosed hold')
    return onsets, durations


class IntervalDataset(PackedPlacementDataset):
    def _load(self, index):
        target, features = super()._load(index)
        if 'onsets' not in target:
            target['onsets'], target['durations'] = interval_targets(target)
        return target, features

    def __getitems__(self, indices):
        batch = super().__getitems__(indices)
        indices = np.asarray(indices)
        charts = np.searchsorted(self.ends, indices, side='right')
        onsets = np.zeros((len(indices), 4), np.int64)
        durations = np.zeros((len(indices), 4), np.float32)
        for chart in np.unique(charts):
            selected = np.flatnonzero(charts == chart)
            rows = indices[selected] - (self.ends[chart - 1] if chart else 0)
            targets, _ = self._load(int(chart))
            onsets[selected] = targets['onsets'][rows]
            durations[selected] = targets['durations'][rows]
        # Releases happen before new heads at the same timestamp.
        releases = torch.isin(batch['target'], torch.tensor([3, 4, 5]))
        batch['condition'][:, 4:][releases] = 0
        batch['target'] = torch.from_numpy(onsets)
        batch['duration_target'] = torch.from_numpy(durations)
        return batch


class EventSampler:
    """All onsets and nearby negatives, plus resampled distant negatives each epoch."""
    def __init__(self, dataset, empty_keep=.1, near_rows=3, seed=42):
        if not 0 < empty_keep <= 1 or near_rows < 0:
            raise ValueError('Invalid empty sampling configuration')
        self.dataset, self.empty_keep, self.seed, self.epoch = dataset, empty_keep, seed, 0
        self.always = np.zeros(len(dataset), bool)
        offset = 0
        for i, entry in enumerate(dataset.entries):
            target, _ = dataset._load(i)
            rows = np.flatnonzero(target['onsets'].any(-1))
            for delta in range(-near_rows, near_rows + 1):
                nearby = rows + delta
                nearby = nearby[(nearby >= 0) & (nearby < entry['rows'])]
                self.always[offset + nearby] = True
            offset += entry['rows']

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        indices = np.flatnonzero(self.always | (rng.random(len(self.always)) < self.empty_keep))
        # Shuffle short blocks, retaining nearby audio-frame reuse.
        blocks = [indices[i:i+32] for i in range(0, len(indices), 32)]
        for block in rng.permutation(len(blocks)):
            yield from blocks[block].tolist()

    def __len__(self):
        return int(self.always.sum() + (~self.always).sum() * self.empty_keep)


class SampledIntervalDataset(IntervalDataset):
    """Attach known inclusion probabilities for unbiased full-grid event loss."""
    def __getitems__(self, indices):
        batch = super().__getitems__(indices)
        batch['inclusion'] = torch.tensor(np.where(self.sampling_always[indices], 1., self.empty_keep), dtype=torch.float32)
        return batch


def losses(output, target, duration_target, held, inclusion=None, *, timing_normalizer=None):
    event = target.any(-1)
    bce = nn.functional.binary_cross_entropy_with_logits(output['event'], event.float(), reduction='none')
    # Horvitz-Thompson-style normalized weighting restores the empty/event prior.
    weights = torch.ones_like(bce) if inclusion is None else inclusion.reciprocal()
    # Phrase sampling knows the original grid size: use that fixed denominator
    # rather than a random sum of inverse sampling probabilities.
    timing = (bce * weights).sum() / (weights.sum() if timing_normalizer is None else timing_normalizer)
    if event.any():
        logits = legal_pattern_logits(output['pattern'][event], held[event])
        pattern = nn.functional.cross_entropy(logits, encode(target[event]) - 1)
    else:
        pattern = output['pattern'].sum() * 0
    hold = target == 2
    duration = nn.functional.smooth_l1_loss(output['duration'][hold], duration_target[hold]) if hold.any() else output['duration'].sum() * 0
    return timing + pattern + duration, dict(timing=timing, pattern=pattern, duration=duration)
