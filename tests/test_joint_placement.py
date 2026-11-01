"""Joint-pattern correctness, hold constraints, training and checkpoint compatibility."""

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

torch=pytest.importorskip('torch')
from cadencia.mert import MertConfig
from cadencia.placement import (
    ACTIONS, CHECKPOINT_FORMAT, LEGACY_CHECKPOINT_FORMAT, PATTERN_COUNT,
    PlacementConfig, PlacementTransformer, encode_patterns, decode_patterns,
    legal_actions, legal_patterns, mask_logits, model_metadata, checkpoint_config,
    EMPTY, TAP, HOLD_START, HOLD_END,
)
from cadencia.placement_data import make_sample
from cadencia.osumania import load_model
from cadencia.osumania.artifact import read_artifact
from scripts.train_placement import run_epoch


def config():
    return PlacementConfig(d_model=32,layers=1,history_events=4,dropout=0.)


def inputs(held=None):
    features=dict(times=np.arange(300)/75,features=np.zeros((300,768),np.float32))
    sample=make_sample(1.,2.,120.,3.,np.zeros(4) if held is None else held,
                       [[TAP,EMPTY,EMPTY,EMPTY]],[.5],[1.],features,config())
    return {key:torch.from_numpy(value[None]) for key,value in sample.items()}


def test_joint_codec_roundtrips_every_pattern():
    tokens=torch.arange(PATTERN_COUNT)
    actions=decode_patterns(tokens)
    assert actions.shape==(1296,4)
    torch.testing.assert_close(encode_patterns(actions),tokens)
    assert encode_patterns(torch.tensor([TAP,EMPTY,TAP,EMPTY])).item()==37
    with pytest.raises(ValueError):encode_patterns(torch.tensor([6,0,0,0]))
    with pytest.raises(ValueError):decode_patterns(torch.tensor([1296]))
    with pytest.raises(ValueError):decode_patterns(torch.tensor([1.5]))


def test_joint_masks_match_all_lane_constraints_and_final_releases():
    held=(torch.arange(16)[:,None]//torch.tensor([1,2,4,8])%2).bool()
    patterns=decode_patterns(torch.arange(PATTERN_COUNT))
    lane_allowed=legal_actions(held)
    expected=lane_allowed[:,:,None,:].expand(-1,-1,PATTERN_COUNT,-1).gather(
        -1,patterns.T[None,:,:,None].expand(16,-1,-1,-1)).squeeze(-1).all(1)
    torch.testing.assert_close(legal_patterns(held),expected)
    for row,state in enumerate(held):
        active=int(state.sum())
        assert int(expected[row].sum())==3**(4-active)*4**active
    final=legal_patterns(held,final_step=True)
    for row,state in enumerate(held):
        allowed=patterns[final[row]]
        assert torch.all(allowed[:,state]==HOLD_END)
        assert torch.all((allowed[:,~state]==EMPTY)|(allowed[:,~state]==TAP))
    probabilities=mask_logits(torch.zeros(16,PATTERN_COUNT),held).softmax(-1)
    assert torch.all(probabilities[~expected]==0)
    torch.testing.assert_close(probabilities.sum(-1),torch.ones(16))


def test_one_joint_choice_preserves_single_notes_and_allows_real_chords():
    model=PlacementTransformer(config()).eval()
    for parameter in model.parameters():parameter.data.zero_()
    model.output.bias.data.fill_(-10)
    single1=int(encode_patterns(torch.tensor([TAP,EMPTY,EMPTY,EMPTY])))
    single2=int(encode_patterns(torch.tensor([EMPTY,TAP,EMPTY,EMPTY])))
    quad=int(encode_patterns(torch.tensor([TAP,TAP,TAP,TAP])))
    model.output.bias.data[single1]=3
    model.output.bias.data[single2]=2.9
    logits=mask_logits(model(**inputs()),torch.zeros(1,4,dtype=torch.bool))
    assert logits.shape==(1,PATTERN_COUNT)
    chosen=decode_patterns(logits.argmax(-1))
    assert int((chosen==TAP).sum())==1  # Ambiguity does not merge two separate patterns.
    model.output.bias.data[quad]=4
    logits=mask_logits(model(**inputs()),torch.zeros(1,4,dtype=torch.bool))
    assert torch.all(decode_patterns(logits.argmax(-1))==TAP)


def test_joint_loss_and_pattern_chord_metrics():
    from torch.utils.data import DataLoader
    model=PlacementTransformer(config())
    records=[]
    for actions in [[TAP,0,0,0],[0,TAP,0,0],[TAP,TAP,0,0],[0,0,0,0]]:
        row={key:value[0] for key,value in inputs().items()}
        row['target']=torch.tensor(actions)
        records.append(row)
    weights=torch.ones(PATTERN_COUNT)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    result=run_epoch(model,DataLoader(records,batch_size=4),torch.device('cpu'),weights,
                     optimizer,log_every=0)
    assert np.isfinite(result['loss'])
    assert result['prediction_head']=='joint_patterns'
    assert result['target_chord_counts']==[1,2,1,0,0]
    assert sum(result['predicted_chord_counts'])==4
    assert 0<=result['pattern_accuracy']<=1
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_checkpoint_formats_are_explicit_and_legacy_still_loads(tmp_path):
    joint=PlacementTransformer(config())
    metadata=model_metadata(joint)
    assert metadata['format']==CHECKPOINT_FORMAT
    assert checkpoint_config(metadata).joint_patterns
    bad=dict(metadata,pattern_encoding='reversed lanes')
    with pytest.raises(ValueError,match='vocabulary'):checkpoint_config(bad)
    legacy=PlacementTransformer(replace(config(),joint_patterns=False))
    old=model_metadata(legacy)
    assert old['format']==LEGACY_CHECKPOINT_FORMAT
    old['config'].pop('joint_patterns')  # The currently training v1 file has no flag.
    old.update(model=legacy.state_dict(),mert=asdict(MertConfig()))
    file=tmp_path/'legacy.pt';torch.save(old,file)
    loaded=load_model(file)
    assert not loaded.config.joint_patterns
    torch.testing.assert_close(loaded.model(**inputs()),legacy.eval()(**inputs()))


def test_joint_export_rollout_preserves_whole_patterns_and_holds(tmp_path,monkeypatch):
    import soundfile as sf
    from test_placement import FakeMert
    model=PlacementTransformer(config())
    for parameter in model.parameters():parameter.data.zero_()
    model.output.bias.data.fill_(-10)
    for actions,score in [([HOLD_START,0,TAP,0],10),([HOLD_END,0,0,0],9),([0,TAP,0,0],8),([0,0,0,0],0)]:
        model.output.bias.data[int(encode_patterns(torch.tensor(actions)))]=score
    checkpoint=tmp_path/'joint.pt'
    torch.save(dict(**model_metadata(model),model=model.state_dict(),mert=asdict(MertConfig())),checkpoint)
    loaded=load_model(checkpoint)
    monkeypatch.setattr(loaded,'extractor',FakeMert())
    source=tmp_path/'song.wav';sf.write(source,np.zeros(4000),8000)
    artifact=loaded.create_chart(source,out=tmp_path/'song.cadencia',difficulty=3.,bpm=120.)
    value,_=read_artifact(artifact)
    assert value['notes'] and any(note['kind']=='hold' for note in value['notes'])
    from collections import Counter
    assert max(Counter(note['time_s'] for note in value['notes']).values())<=2
    assert all(note['end_s']>note['time_s'] for note in value['notes'] if note['kind']=='hold')
    assert 'joint-pattern' in value['candidate_policy']


def test_seeded_joint_sampling_can_emit_nonmodal_patterns(tmp_path,monkeypatch):
    import soundfile as sf
    from test_placement import FakeMert
    model=PlacementTransformer(config())
    for parameter in model.parameters():parameter.data.zero_()
    model.output.bias.data.fill_(-50)
    single=int(encode_patterns(torch.tensor([TAP,0,0,0])))
    model.output.bias.data[0]=4.
    model.output.bias.data[single]=3.8
    checkpoint=tmp_path/'sampling.pt'
    torch.save(dict(**model_metadata(model),model=model.state_dict(),mert=asdict(MertConfig())),checkpoint)
    loaded=load_model(checkpoint);monkeypatch.setattr(loaded,'extractor',FakeMert())
    source=tmp_path/'song.wav';sf.write(source,np.zeros(8000),8000)
    greedy=loaded.predict(source,difficulty=3.,bpm=120.,decoding='argmax')
    assert greedy['notes']==[]
    sampled=loaded.predict(source,difficulty=3.,bpm=120.,decoding='sample',seed=42)
    assert sampled['notes'] and all(note['lane']==0 for note in sampled['notes'])
    assert sampled==loaded.predict(source,difficulty=3.,bpm=120.,seed=42)
    assert sampled['notes']!=loaded.predict(source,difficulty=3.,bpm=120.,seed=43)['notes']
    artifact=loaded.create_chart(source,out=tmp_path/'sample.cadencia',difficulty=3.,bpm=120.,seed=42)
    restored,_=read_artifact(artifact)
    assert restored['decoding']=='sample' and restored['seed']==42 and restored['temperature']==1.
    with pytest.raises(ValueError):loaded.predict(source,difficulty=3.,bpm=120.,temperature=0)
