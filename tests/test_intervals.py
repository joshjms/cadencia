"""Interval supervision, balanced sampling, duration gradients and inference compatibility."""
from dataclasses import asdict
import numpy as np
import pytest
import torch
from cadencia.interval import (IntervalTransformer, interval_targets, losses, EventSampler,
                               legal_pattern_logits, encode, decode, metadata)
from cadencia.mert import MertConfig
from cadencia.osumania import load_model
from tests.test_joint_placement import inputs, config


def targets():
    # Hold on lane 0, tap on lane 1 inside it; release/restart at its endpoint.
    actions = np.array([[2,0,0,0],[0,1,0,0],[5,0,0,0],[0,0,1,0],[3,0,0,0]])
    return dict(actions=actions, beats=np.array([0.,.5,1.,2.,3.]))


def test_interval_supervision_preserves_cross_lane_taps_and_restarts():
    onsets, durations = interval_targets(targets())
    np.testing.assert_array_equal(onsets[:,0],[2,0,2,0,0])
    assert onsets[1,1] == 1
    assert durations[0,0] == pytest.approx(np.log1p(1.))
    assert durations[2,0] == pytest.approx(np.log1p(2.))
    assert not durations[onsets != 2].any()
    bad = targets(); bad['actions'][0,0] = 0
    with pytest.raises(ValueError): interval_targets(bad)


def test_onset_vocabulary_and_occupancy():
    tokens=torch.arange(81)
    torch.testing.assert_close(encode(decode(tokens)),tokens)
    held=torch.tensor([[True,False,False,False],[True,True,True,True]])
    masked=legal_pattern_logits(torch.zeros(2,80),held)
    assert torch.isfinite(masked[0]).sum() == 26
    assert torch.isneginf(masked[1]).all()


def test_event_duration_heads_receive_gradients_and_empty_batches_are_finite():
    model=IntervalTransformer(config())
    output=model(**inputs())
    target=torch.tensor([[2,1,0,0]])
    duration=torch.tensor([[np.log1p(2.),0,0,0]],dtype=torch.float32)
    loss,_=losses(output,target,duration,torch.zeros(1,4,dtype=torch.bool))
    loss.backward()
    assert all(head.weight.grad.abs().sum()>0 for head in [model.event,model.pattern,model.duration])
    output=model(**inputs())
    empty,_=losses(output,torch.zeros_like(target),torch.zeros_like(duration),torch.ones(1,4,dtype=torch.bool))
    assert torch.isfinite(empty)
    empty.backward()


def test_importance_weights_restore_full_grid_event_loss():
    # Original: one event, nine empties. Sample one event + three empties.
    event=torch.tensor([1.,-1.,-1.,-1.])
    output=dict(event=event,pattern=torch.zeros(4,80),duration=torch.zeros(4,4))
    target=torch.tensor([[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    _,parts=losses(output,target,torch.zeros(4,4),torch.zeros(4,4,dtype=torch.bool),torch.tensor([1.,1/3,1/3,1/3]))
    expected=torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor([1.]+[-1.]*9),torch.tensor([1.]+[0.]*9))
    torch.testing.assert_close(parts['timing'],expected)


def test_sampler_keeps_events_nearby_empties_and_resamples_far_empties():
    class Dataset:
        entries=[dict(rows=100)]
        def __len__(self): return 100
        def _load(self,i):
            onsets=np.zeros((100,4),int);onsets[50,0]=2
            return dict(onsets=onsets),None
    sampler=EventSampler(Dataset(),empty_keep=.1,near_rows=3)
    first,second=list(sampler),list(sampler)
    assert set(range(47,54)) <= set(first)
    assert len(first)==len(set(first))
    assert first != second
    assert len(first)<100


def test_interval_checkpoint_generation_and_artifact(tmp_path,monkeypatch):
    import cadencia.osumania.placement_generation as generation
    model=IntervalTransformer(config())
    for p in model.parameters(): p.data.zero_()
    model.event.bias.data.fill_(20)
    model.pattern.bias.data.fill_(-20)
    # tap lane1 + hold lane0 = base3 token5, index4.
    model.pattern.bias.data[4]=20
    model.duration.bias.data.fill_(float(np.log1p(.4)))
    path=tmp_path/'interval.pt'
    torch.save(dict(**metadata(model),model=model.state_dict(),mert=asdict(MertConfig())),path)
    class Extractor:
        def __init__(self,**kwargs): pass
        def extract(self,*args,**kwargs):
            return dict(times=np.arange(75)/75,features=np.zeros((75,768),np.float32),duration_s=1.),None
    monkeypatch.setattr(generation,'FrozenMert',Extractor)
    loaded=load_model(path)
    result=loaded.predict(tmp_path/'song.mp3',difficulty=3,bpm=120,decoding='argmax')
    assert any(n['kind']=='hold' for n in result['notes'])
    holds=[n for n in result['notes'] if n['kind']=='hold']
    assert holds[0]['end_s']==pytest.approx(.2)
    for lane in range(4):
        notes=[n for n in result['notes'] if n['lane']==lane]
        for previous,current in zip(notes,notes[1:]):
            assert current['time_s'] >= previous.get('end_s',previous['time_s']+1e-9)
    assert all(n.get('end_s',0)<=1 for n in result['notes'])
    assert any(n['lane']==1 and 0<n['time_s']<.2 for n in result['notes'])
    from cadencia.osumania.artifact import validate_chart
    validate_chart(result,placement=True)


def test_merge_removes_only_same_lane_overlapping_taps():
    from cadencia.interval import merge_notes
    hold = dict(time_s=1.,end_s=2.,lane=0,kind='hold')
    def tap(time,lane): return dict(time_s=time,lane=lane,kind='tap')
    notes=[tap(1.5,0),tap(1.5,1),hold,tap(2.,0),tap(.5,0),tap(1.,0)]
    result=merge_notes(notes)
    assert result == [tap(.5,0),hold,tap(1.5,1),tap(2.,0)]
    with pytest.raises(ValueError):
        merge_notes([hold,dict(time_s=1.5,end_s=3.,lane=0,kind='hold')])
