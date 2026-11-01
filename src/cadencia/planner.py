"""Whole-song, four-beat-bar intensity planning from frozen MERT audio."""
from dataclasses import asdict
import numpy as np
import torch
from torch import nn
from .interval import IntervalTransformer, IntervalDataset, SampledIntervalDataset

FORMAT = 'cadencia.placement.planned.v4'
HIGH_RES_FORMAT = 'cadencia.placement.planned.v5'
PLANNER_FORMAT = 'cadencia.bar_planner.v1'
PLAN_FIELDS = ['rest_probability', 'log1p_heads_per_beat', 'chord_fraction', 'hold_occupancy']


def bar_inputs(features, times, beats, bpms, difficulty):
    """Audio-only inputs; four-beat bars aligned to the chart beat origin.

    Meter is explicitly assumed to be 4/4. Partial edge bars are retained.
    """
    times, beats = np.asarray(times), np.asarray(beats)
    if len(times) < 2 or not np.all(np.diff(beats) > 0):
        raise ValueError('Planner requires a monotonic beat timeline')
    rhythm=features.get('rhythm')
    positions=np.interp(times,rhythm['times'],rhythm['bar_positions']) if rhythm is not None else beats/4
    ids = np.floor(positions).astype(int)
    first, last = int(ids[0]), int(ids[-1])
    count = last - first + 1
    if count > 4096:
        raise ValueError('Song exceeds 4096-bar planner limit')
    row_bars = ids - first
    frame_beats = np.interp(features['times'], times, beats)
    frame_positions=np.interp(features['times'],rhythm['times'],rhythm['bar_positions']) if rhythm is not None else frame_beats/4
    frame_bars = np.clip(np.floor(frame_positions).astype(int)-first,0,count-1)
    dim = features['features'].shape[1]
    inputs = np.zeros((count,dim*2+3+(6 if rhythm is not None else 0)),np.float32)
    for bar in range(count):
        frames = np.asarray(features['features'][frame_bars == bar],dtype=np.float32)
        if len(frames):
            inputs[bar,:dim] = frames.mean(0)
            inputs[bar,dim:dim*2] = frames.std(0)
        rows = row_bars == bar
        inputs[bar,dim*2:dim*2+3] = [difficulty/10, float(np.asarray(bpms)[rows].mean())/240 if rows.any() else 0, bar/max(1,count-1)]
        if rhythm is not None:
            from .rhythm import rhythm_at
            inputs[bar,-6:]=rhythm_at(rhythm,times[rows]).mean(0) if rows.any() else 0
    return inputs, row_bars


def bar_targets(target, row_bars):
    from .interval import interval_targets
    onsets, _ = interval_targets(target)
    heads = (onsets != 0).sum(-1)
    result = np.zeros((int(row_bars.max())+1,4),np.float32)
    for bar in range(len(result)):
        rows = row_bars == bar
        beat_values = target['beats'][rows]
        if not len(beat_values):
            result[bar,0] = 1
            continue
        step = float(np.median(np.diff(target['beats'])))
        span = max(step,float(beat_values[-1]-beat_values[0])+step)
        events = heads[rows] > 0
        result[bar] = [not events.any(), np.log1p(heads[rows].sum()/span),
                       float((heads[rows]>1).sum())/max(1,int(events.sum())),
                       np.mean(target['held'][rows] | (onsets[rows] == 2))]
    return result


class BarPlanner(nn.Module):
    def __init__(self,audio_dim=768,use_rhythm=False):
        super().__init__()
        self.input = nn.Linear(audio_dim*2+3+(6 if use_rhythm else 0),128)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(128,4,512,dropout=.1,batch_first=True),2,
                                              enable_nested_tensor=False)
        self.output = nn.Linear(128,4)

    def forward(self, audio_bars):
        # Whole-song attention is noncausal: every bar can inspect every other bar.
        hidden = self.input(audio_bars)
        positions = torch.arange(hidden.shape[-2],device=hidden.device,dtype=hidden.dtype)
        frequencies = torch.exp(torch.arange(0,128,2,device=hidden.device,dtype=hidden.dtype)*(-np.log(10000.)/128))
        angle = positions[:,None]*frequencies
        position = torch.stack([angle.sin(),angle.cos()],-1).flatten(-2)
        return self.output(self.encoder(hidden + position))

    def predict(self, audio_bars):
        raw = self(audio_bars)
        return torch.stack([raw[...,0].sigmoid(),nn.functional.softplus(raw[...,1]),
                            raw[...,2].sigmoid(),raw[...,3].sigmoid()],-1)


def planner_loss(raw, target):
    return (nn.functional.binary_cross_entropy_with_logits(raw[...,0],target[...,0])
            + nn.functional.smooth_l1_loss(nn.functional.softplus(raw[...,1]),target[...,1])
            + nn.functional.mse_loss(raw[...,2:].sigmoid(),target[...,2:]))


class PlannedTransformer(IntervalTransformer):
    def __init__(self,config=None):
        super().__init__(config)
        self.plan_projection = nn.Sequential(nn.Linear(4,self.config.d_model),nn.GELU(),nn.Linear(self.config.d_model,self.config.d_model))

    def forward(self, plan, **inputs):
        hidden = self.encode(**inputs) + self.plan_projection(plan)
        # A rest plan lowers onset probability; training and generation share this rule.
        rest_prior = torch.log1p(-plan[:,0].clamp(max=1-1e-5))
        return dict(event=self.event(hidden).squeeze(-1)+rest_prior,
                    pattern=self.pattern(hidden),duration=self.duration(hidden))


def metadata(model):
    from .rhythm import SPEC
    return dict(rhythm_backend=SPEC if model.config.use_rhythm else None,format=HIGH_RES_FORMAT if model.config.use_mel else FORMAT,architecture_version='v1',config=asdict(model.config),plan_fields=PLAN_FIELDS,
                planner_format=PLANNER_FORMAT,beats_per_bar=4,
                duration_units='log1p(beats)',pattern_encoding='base3 onsets; IDs 1..80')


class PlannedDataset(IntervalDataset):
    """Decoder gets frozen audio-predicted plans, never reference plan labels."""
    def __getitems__(self,indices):
        batch = super().__getitems__(indices)
        indices = np.asarray(indices)
        charts = np.searchsorted(self.ends,indices,side='right')
        plan = np.zeros((len(indices),4),np.float32)
        for chart in np.unique(charts):
            selected = np.flatnonzero(charts==chart)
            rows = indices[selected]-(self.ends[chart-1] if chart else 0)
            plans,row_bars = self.plans[int(chart)]
            plan[selected] = plans[row_bars[rows]]
        batch['plan'] = torch.from_numpy(plan)
        return batch

    def prepare_plans(self,planner):
        self.plans = []
        planner.eval()
        with torch.inference_mode():
            for i,entry in enumerate(self.entries):
                target, features = self._load(i)
                inputs, rows = bar_inputs(features,target['times'],target['beats'],target['bpm'],entry['difficulty'])
                plan = planner.predict(torch.from_numpy(inputs)[None])[0].numpy()
                self.plans.append((plan,rows))


class SampledPlannedDataset(PlannedDataset):
    def __getitems__(self,indices):
        batch = super().__getitems__(indices)
        batch['inclusion'] = torch.tensor(np.where(self.sampling_always[indices],1.,self.empty_keep),dtype=torch.float32)
        return batch
