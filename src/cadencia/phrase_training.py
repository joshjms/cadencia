"""Contiguous beat windows and differentiable joint-pattern transition loss.

History remains causal and teacher-forced. This adds sequence supervision, not
backpropagation through sampled notes or the discrete phrase evaluator.
"""
import math
import numpy as np
import torch

from .interval import EventSampler, encode, legal_pattern_logits


class PhraseDataset:
    """One complete beat-relative phrase per item, with unbiased empty sampling."""
    def __init__(self, base, beats=4., *, training=False, empty_keep=.1, near_rows=3, seed=42):
        if not math.isfinite(beats) or beats <= 0:
            raise ValueError('Phrase beats must be positive')
        if not 0 < empty_keep <= 1 or type(near_rows) is not int or near_rows < 0:
            raise ValueError('Invalid sampling configuration')
        self.base, self.beats, self.training = base, beats, training
        self.empty_keep, self.seed = empty_keep, seed
        self.always = EventSampler(base, empty_keep, near_rows, seed).always if training else None
        self.windows = []
        self.chart_counts = []
        for chart, entry in enumerate(base.entries):
            target, _ = base._load(chart)
            positions = target['beats']
            if not np.isfinite(positions).all() or (np.diff(positions) <= 0).any():
                raise ValueError('Phrase training needs increasing beat positions')
            # Include partial intro/outro windows so no chart rows are dropped.
            labels = np.floor(positions/beats).astype(np.int64)
            boundaries = np.r_[0, np.flatnonzero(np.diff(labels))+1, len(labels)]
            offset = int(base.ends[chart-1]) if chart else 0
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                self.windows.append((chart, int(start), int(end), offset))
            self.chart_counts.append(len(boundaries)-1)

    def __len__(self):
        return len(self.windows)

    def selection(self, key):
        epoch, index = key if isinstance(key, tuple) else (0, key)
        chart, start, end, offset = self.windows[index]
        rows = np.arange(start, end)
        inclusion = np.ones(len(rows), np.float32)
        if self.training:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, index]))
            always = self.always[offset+rows]
            inclusion = np.where(always, 1., self.empty_keep).astype(np.float32)
            selected = always | (rng.random(len(rows)) < self.empty_keep)
            # Fully silent phrases can have zero sampled rows. Keep one uniformly
            # chosen row with its corrected inclusion probability in that case.
            if not selected.any():
                selected[rng.integers(len(rows))] = True
            # Each distant row can also be chosen by the uniform fallback.
            extra = (1-self.empty_keep)**len(rows)/len(rows) if not always.any() else 0.
            inclusion[~always] += extra
            rows, inclusion = rows[selected], inclusion[selected]
        return chart, rows, offset, inclusion, end-start

    def __getitem__(self, key):
        return self.__getitems__([key])

    def __getitems__(self, keys):
        selections = [self.selection(key) for key in keys]
        indices = np.concatenate([offset+rows for _,rows,offset,_,_ in selections])
        batch = self.base.__getitems__(indices.tolist())
        beats = []
        ids = []
        for phrase_id,(chart,rows,_,_,_) in enumerate(selections):
            target,_ = self.base._load(chart)
            beats.append(target['beats'][rows].astype(np.float32))
            ids.append(np.full(len(rows),phrase_id,np.int64))
        batch['phrase_beats'] = torch.from_numpy(np.concatenate(beats))
        batch['phrase_ids'] = torch.from_numpy(np.concatenate(ids))
        batch['timing_normalizer'] = torch.tensor(sum(n for *_,n in selections),dtype=torch.float32)
        if self.training:
            batch['inclusion'] = torch.from_numpy(np.concatenate([s[3] for s in selections]))
        return batch


class PhraseSampler:
    def __init__(self, size, seed=42):
        self.size, self.seed, self.epoch = size, seed, 0

    def __len__(self):
        return self.size

    def __iter__(self):
        epoch = self.epoch
        self.epoch += 1
        for index in np.random.default_rng(self.seed+epoch).permutation(self.size):
            yield epoch, int(index)


class PhraseBatchSampler:
    """Mix whole phrases until an update reaches the sampled-row budget.

    Sampling is computed identically here and in dataset workers. Oversized
    phrases stay intact; forward_rows bounds their activation memory.
    """
    def __init__(self, dataset, rows=128, *, shuffle=True, seed=42):
        if type(rows) is not int or rows < 1:
            raise ValueError('Batch row budget must be positive')
        self.dataset,self.rows,self.shuffle,self.seed = dataset,rows,shuffle,seed
        self.epoch = 0

    def __iter__(self):
        epoch = self.epoch
        self.epoch += int(self.shuffle)
        order = np.random.default_rng(self.seed+epoch).permutation(len(self.dataset)) if self.shuffle else range(len(self.dataset))
        batch,count = [],0
        for index in order:
            key = (epoch,int(index))
            count += len(self.dataset.selection(key)[1])
            batch.append(key)
            if count >= self.rows:
                yield batch
                batch,count = [],0
        if batch:
            yield batch


def collate_phrase(batch):
    return batch


def transition_loss(pattern_logits, target, held, beats):
    """JS divergence of expected vs human joint-pattern transition distributions.

    Eighty joint patterns retain taps, chords, hold starts and lane identities.
    Adjacent annotated onsets are grouped by their gap: <=half beat, <=one beat,
    or longer. Soft outer products connect predictions across the phrase; this
    objective is not merely a sum of per-note cross-entropies.
    """
    event = target.any(-1)
    rows = torch.nonzero(event, as_tuple=True)[0]
    if len(rows) < 2:
        return pattern_logits.sum()*0, 0
    probabilities = legal_pattern_logits(pattern_logits[rows].float(), held[rows]).softmax(-1)
    labels = encode(target[rows])-1
    gaps = beats[rows[1:]]-beats[rows[:-1]]
    if not torch.isfinite(gaps).all() or (gaps <= 0).any():
        raise ValueError('Phrase onset times must be strictly increasing')
    buckets = (gaps > .5).long()+(gaps > 1.).long()
    expected = []
    for bucket in range(3):
        selected = buckets == bucket
        # JS distributions are small enough to keep fully in float32 even when
        # the decoder runs in bfloat16 autocast.
        with torch.autocast(device_type=pattern_logits.device.type, enabled=False):
            expected.append(probabilities[:-1][selected].T @ probabilities[1:][selected])
    pairs = len(rows)-1
    expected = torch.stack(expected)/pairs
    indices = buckets*80*80+labels[:-1]*80+labels[1:]
    human = torch.bincount(indices, minlength=3*80*80).reshape(3,80,80).float()/pairs
    mean = (expected+human)/2
    eps = 1e-12
    js = .5*((expected*(expected.clamp_min(eps).log()-mean.clamp_min(eps).log())).sum()
             +(human*(human.clamp_min(eps).log()-mean.clamp_min(eps).log())).sum())/math.log(2)
    return js, pairs


def grouped_transition_loss(pattern_logits, target, held, beats, phrase_ids=None):
    """Pair-weighted phrase losses, never transitions across phrase/chart edges."""
    if phrase_ids is None:
        value,pairs = transition_loss(pattern_logits,target,held,beats)
        return value,pairs,int(pairs>0)
    total = pattern_logits.sum()*0
    count,windows = 0,0
    for phrase in torch.unique(phrase_ids):
        rows = phrase_ids == phrase
        value,pairs = transition_loss(pattern_logits[rows],target[rows],held[rows],beats[rows])
        total = total+value*pairs
        count += pairs
        windows += int(pairs>0)
    return total/max(1,count),count,windows


def forward_rows(model, inputs, chunk_size=128, *, checkpointed=False):
    """Retain complete phrase losses while limiting decoder activation memory."""
    from torch.utils.checkpoint import checkpoint
    n = len(inputs['history'])
    outputs = []
    for start in range(0, n, chunk_size):
        # Packed audio is shared; audio_indices maps each query into that memory.
        chunk = {key:value if key == 'audio' else value[start:start+chunk_size] for key,value in inputs.items()}
        if checkpointed:
            outputs.append(checkpoint(model, use_reentrant=False, **chunk))
        else:
            outputs.append(model(**chunk))
    return {key:torch.cat([output[key] for output in outputs]) for key in outputs[0]}
