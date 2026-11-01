"""Chart-only next-onset pretraining and explicit transfer into audio decoders."""
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from .interval import IntervalTransformer, interval_targets, encode
from .placement import PlacementConfig
from .placement_data import SCHEMA, advance_held
from .placement import ACTIONS

FORMAT = 'cadencia.chart_sequence.v1'


class ChartSequenceModel(IntervalTransformer):
    """Same history/self-attention/FFN weights as interval and planned decoders.

    No audio encoder, cross-attention, event timing head, or planner is trained.
    The gap head predicts log1p(beats until the next onset) and is not transferred.
    """
    def __init__(self, config=None):
        super().__init__(config)
        del self.audio, self.audio_time, self.event
        if hasattr(self,'mel_encoder'):
            del self.mel_encoder,self.mel_time
        if hasattr(self,'rhythm_projection'):del self.rhythm_projection
        for layer in self.decoder.layers:
            del layer.multihead_attn, layer.norm2
        self.gap = nn.Linear(self.config.d_model,1)

    def forward(self, history, history_timing, history_padding, condition):
        hidden,padding,mask = self.history_tokens(history,history_timing,history_padding,condition)
        for layer in self.decoder.layers:
            attended = layer.self_attn(hidden,hidden,hidden,attn_mask=mask,
                                       key_padding_mask=padding,need_weights=False)[0]
            hidden = layer.norm1(hidden + layer.dropout1(attended))
            feedforward = layer.linear2(layer.dropout(layer.activation(layer.linear1(hidden))))
            hidden = layer.norm3(hidden + layer.dropout3(feedforward))
        hidden = hidden[:,-1]
        return dict(pattern=self.pattern(hidden),duration=self.duration(hidden),
                    gap=nn.functional.softplus(self.gap(hidden).squeeze(-1)))


def sequence_loss(output,target,duration,gap):
    pattern=nn.functional.cross_entropy(output['pattern'],encode(target)-1)
    hold=target==2
    duration_loss=nn.functional.smooth_l1_loss(output['duration'][hold],duration[hold]) if hold.any() else output['duration'].sum()*0
    gap_loss=nn.functional.smooth_l1_loss(output['gap'],gap)
    return pattern+duration_loss+gap_loss,dict(pattern=pattern,duration=duration_loss,gap=gap_loss)


class ChartSequenceDataset(Dataset):
    """Only action shards and metadata are read: no MERT files or current/future labels in inputs."""
    def __init__(self,path,split):
        self.path=Path(path)
        if self.path.is_dir(): self.path/='manifest.json'
        self.manifest=json.loads(self.path.read_text())
        if self.manifest.get('schema')!=SCHEMA or self.manifest.get('actions')!=ACTIONS:
            raise ValueError('Unsupported chart dataset')
        self.config=PlacementConfig(**self.manifest['config'])
        self.entries=[e for e in self.manifest['charts'] if e['split']==split]
        counts=[]
        for i in range(len(self.entries)):
            target=self._load(i)
            counts.append(len(target['onset_rows']))
        self.ends=np.cumsum(counts)

    @lru_cache(maxsize=None)
    def _load(self,index):
        entry=self.entries[index]
        path=(self.path.parent/entry['file']).resolve()
        if not path.is_relative_to(self.path.parent.resolve()): raise ValueError('Dataset shard escapes root')
        data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('Chart shard checksum mismatch')
        import io
        with np.load(io.BytesIO(data),allow_pickle=False) as arrays:
            target={k:arrays[k] for k in arrays.files}
        target['onsets'],target['durations']=interval_targets(target)
        target['onset_rows']=np.flatnonzero(target['onsets'].any(-1))
        return target

    def __len__(self):return int(self.ends[-1]) if len(self.ends) else 0

    def __getitem__(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        chart=int(np.searchsorted(self.ends,index,side='right'))
        position=index-(int(self.ends[chart-1]) if chart else 0)
        target=self._load(chart)
        row=target['onset_rows'][position]
        previous=target['onset_rows'][position-1] if position else None
        # Anchor at the previous onset, never the target timestamp. This keeps
        # the gap, future beat phase and intervening releases out of the input.
        anchor_time=float(target['times'][previous]) if previous is not None else 0.
        anchor_beat=float(target['beats'][previous]) if previous is not None else float(target['beats'][0])
        bpm=float(target['bpm'][previous]) if previous is not None else float(target['bpm'][0])
        groups=target['groups']
        history=groups[groups<=previous][-self.config.history_events:] if previous is not None else np.array([],int)
        held=advance_held(target['held'][previous],target['actions'][previous]) if previous is not None else np.zeros(4,bool)
        h=self.config.history_events
        actions=np.zeros((h,4),np.int64);timing=np.zeros((h,4),np.float32);padding=np.ones(h,bool)
        if len(history):
            n=len(history);actions[:n]=target['actions'][history]
            ages=target['times'][history]-anchor_time;gaps=target['beats'][history]-anchor_beat
            timing[:n]=np.column_stack([ages/10,gaps/16,np.sin(gaps*2*np.pi),np.cos(gaps*2*np.pi)])
            padding[:n]=False
        gap=float(target['beats'][row])-anchor_beat
        if gap<0:raise ValueError('Nonmonotonic beat timeline')
        return dict(history=actions,history_timing=timing,history_padding=padding,
                    condition=np.array([self.entries[chart]['difficulty']/10,bpm/240,np.sin(anchor_beat*2*np.pi),np.cos(anchor_beat*2*np.pi),*held],np.float32),
                    target=target['onsets'][row].copy(),duration_target=target['durations'][row].copy(),gap_target=np.float32(np.log1p(gap)))


def split_identities(manifest):
    return {s:{key:sorted({str(e[key]) for e in manifest['charts'] if e['split']==s and key in e})
               for key in ['chart_id','group_id','song_id','audio_sha256']}
            for s in ['train','validation','test']}


def assert_disjoint(manifest):
    identities=split_identities(manifest)
    for key in ['chart_id','group_id','song_id','audio_sha256']:
        values=[set(identities[s][key]) for s in identities]
        if any(values[i]&values[j] for i in range(3) for j in range(i)):
            raise ValueError(f'Pretraining split leakage: {key}')


def initialize_decoder(model,checkpoint,manifest):
    """Strict allowlist transfer; audio, cross-attention, event and planner stay fresh."""
    source_config = dict(checkpoint.get('config',{}))
    destination_config = asdict(model.config)
    for key in ['audio_dim','audio_hz','grid_hz','use_mel','use_rhythm']:
        source_config.pop(key,None);destination_config.pop(key,None)
    if checkpoint.get('format')!=FORMAT or source_config!=destination_config:
        raise ValueError('Incompatible chart pretraining checkpoint/configuration')
    assert_disjoint(manifest)
    current=split_identities(manifest)
    source=checkpoint['split_identities']
    # Never initialize from a pretraining run whose training charts/audio overlap
    # the downstream validation/test sets, even when manifests differ.
    for split in ['validation','test']:
        for key in ['chart_id','group_id','song_id','audio_sha256']:
            if set(source['train'][key])&set(current[split][key]):
                raise ValueError(f'Pretraining train data overlaps downstream {split}: {key}')
    for key in ['chart_id','group_id','song_id','audio_sha256']:
        if set(source['validation'][key])&set(current['test'][key]):
            raise ValueError(f'Pretraining validation data overlaps downstream test: {key}')
    reference=ChartSequenceModel(model.config).state_dict()
    state=checkpoint['model']
    if state.keys()!=reference.keys() or any(state[k].shape!=v.shape or not torch.isfinite(state[k]).all() for k,v in reference.items()):
        raise ValueError('Invalid chart pretraining state')
    destination=model.state_dict()
    transferred=[k for k in reference if not k.startswith('gap.')]
    if any(k not in destination or destination[k].shape!=state[k].shape for k in transferred):
        raise ValueError('Decoder layout does not support chart weight transfer')
    for key in transferred:destination[key]=state[key]
    model.load_state_dict(destination,strict=True)
    return dict(format=FORMAT,source_dataset_sha256=checkpoint['dataset_sha256'],
                best_epoch=checkpoint['best_epoch'],transferred_keys=transferred,
                fresh_keys=[k for k in destination if k not in transferred])
