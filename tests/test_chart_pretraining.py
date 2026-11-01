"""No future-target leakage, chart-only learning, and strict audio decoder initialization."""
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from cadencia.chart_pretraining import (ChartSequenceDataset,ChartSequenceModel,sequence_loss,
                                        initialize_decoder,FORMAT,split_identities)
from cadencia.placement import ACTIONS
from cadencia.placement_data import SCHEMA
from cadencia.planner import PlannedTransformer
from cadencia.interval import IntervalTransformer
from cadencia.mert import MertConfig
from tests.test_joint_placement import config,inputs


@pytest.fixture
def dataset(tmp_path):
    root=tmp_path/'data';root.mkdir()
    times=np.arange(300)/50;beats=times*2
    actions=np.zeros((300,4),np.int64)
    for i in range(25,275,25):actions[i,(i//25)%4]=1
    actions[50,2]=2;actions[75,2]=3
    held=np.zeros((300,4),bool);held[51:76,2]=True
    np.save(root/'features.npy',np.zeros((450,768),np.float32),allow_pickle=False)
    np.savez(root/'audio.npz',times=np.arange(450)/75,duration_s=np.array(6.))
    entries=[]
    for i,split in enumerate(['train','validation','test']):
        file=f'{i}.npz'
        np.savez(root/file,times=times,beats=beats,bpm=np.full(300,120.,np.float32),actions=actions,held=held,groups=np.flatnonzero(actions.any(-1)))
        entry=dict(chart_id=i,song_id=i,group_id=str(i),audio_sha256=f'audio{i}',split=split,rows=300,difficulty=3.,file=file,audio_file='audio.npz',feature_file='features.npy')
        for key,checksum in [('file','sha256'),('audio_file','audio_file_sha256'),('feature_file','feature_sha256')]:entry[checksum]=hashlib.sha256((root/entry[key]).read_bytes()).hexdigest()
        entries.append(entry)
    manifest=dict(schema=SCHEMA,actions=ACTIONS,config=asdict(config()),mert=asdict(MertConfig()),charts=entries)
    (root/'manifest.json').write_text(json.dumps(manifest))
    return root,manifest


def test_sequence_inputs_do_not_reveal_next_time_pattern_or_releases(dataset):
    root,_=dataset
    ds=ChartSequenceDataset(root,'train')
    before=ds[2]
    target=ds._load(0)
    row=target['onset_rows'][2]
    # Change the future onset timestamp, beat, pattern and tempo. Inputs stay equal.
    target['times'][row]+=.08;target['beats'][row]+=.16
    target['bpm'][row]=60
    target['onsets'][row]=[1,0,0,0]
    after=ds[2]
    for key in ['history','history_timing','history_padding','condition']:
        np.testing.assert_array_equal(before[key],after[key])
    assert before['gap_target'] != after['gap_target']
    assert not np.array_equal(before['target'],after['target'])
    # Neither MERT nor its metadata are opened by the chart-only dataset.
    (root/'features.npy').unlink();(root/'audio.npz').unlink()
    assert len(ChartSequenceDataset(root,'train'))>0
    assert ds[0]['history_padding'].all()


def checkpoint(model,manifest):
    return dict(format=FORMAT,config=asdict(model.config),model=model.state_dict(),best_epoch=1,
                dataset_sha256='fixture',split_identities=split_identities(manifest))


def test_chart_only_gradients_and_exact_allowlisted_transfer(dataset):
    _,manifest=dataset
    model=ChartSequenceModel(config())
    batch={k:v for k,v in inputs().items() if not k.startswith('audio')}
    output=model(**batch)
    loss,_=sequence_loss(output,torch.tensor([[2,1,0,0]]),torch.tensor([[.7,0,0,0]]),torch.tensor([.4]))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.lane_actions[0].weight.grad.abs().sum()>0
    assert model.decoder.layers[0].self_attn.in_proj_weight.grad.abs().sum()>0
    assert model.gap.weight.grad.abs().sum()>0
    assert not any('audio' in key or 'multihead_attn' in key for key in model.state_dict())
    for decoder in [IntervalTransformer(config()),PlannedTransformer(config())]:
        original=copy.deepcopy(decoder.state_dict())
        report=initialize_decoder(decoder,checkpoint(model,manifest),manifest)
        for key in report['transferred_keys']:torch.testing.assert_close(decoder.state_dict()[key],model.state_dict()[key])
        for key in report['fresh_keys']:torch.testing.assert_close(decoder.state_dict()[key],original[key])
        assert 'event.weight' in report['fresh_keys']
        assert 'audio.weight' in report['fresh_keys']
        assert 'decoder.layers.0.multihead_attn.in_proj_weight' in report['fresh_keys']
        assert 'pattern.weight' in report['transferred_keys']
        assert not any(k.startswith('gap.') for k in report['transferred_keys'])


def test_transfer_rejects_leakage_bad_layout_and_nonfinite_weights(dataset):
    _,manifest=dataset
    model=ChartSequenceModel(config());saved=checkpoint(model,manifest)
    bad=copy.deepcopy(saved);bad['split_identities']['train']['audio_sha256']=['audio2']
    with pytest.raises(ValueError,match='overlaps'):initialize_decoder(IntervalTransformer(config()),bad,manifest)
    bad=copy.deepcopy(saved);bad['config']['history_events']=999
    with pytest.raises(ValueError,match='configuration'):initialize_decoder(IntervalTransformer(config()),bad,manifest)
    bad=copy.deepcopy(saved);bad['model']['query'].fill_(float('nan'))
    with pytest.raises(ValueError,match='Invalid'):initialize_decoder(IntervalTransformer(config()),bad,manifest)


def test_pretrain_resume_initialize_and_audio_finetune(dataset,tmp_path):
    root,_=dataset;chart=tmp_path/'chart.pt';planner=tmp_path/'planner.pt';audio=tmp_path/'audio.pt'
    def run(*args):
        result=subprocess.run([sys.executable,*args],capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
    run('scripts/pretrain_charts.py','--data',str(root),'--output',str(chart),'--epochs','1','--batch-size','8')
    run('scripts/pretrain_charts.py','--data',str(root),'--output',str(chart),'--resume',str(chart.with_suffix('.last.pt')),'--epochs','2','--batch-size','8')
    run('scripts/train_bar_planner.py','--data',str(root),'--output',str(planner),'--epochs','1')
    run('scripts/train_intervals.py','--data',str(root),'--output',str(audio),'--planner',str(planner),'--chart-pretraining',str(chart),'--epochs','1','--workers','0','--batch-size','128')
    saved=torch.load(audio,weights_only=True)
    assert saved['chart_pretraining']['sha256']==hashlib.sha256(chart.read_bytes()).hexdigest()
    assert saved['chart_pretraining']['transferred_keys']
    from cadencia.osumania import load_model
    loaded=load_model(audio)
    assert loaded.planned
    run('scripts/train_intervals.py','--data',str(root),'--output',str(audio),'--planner',str(planner),'--resume',str(audio.with_suffix('.last.pt')),'--epochs','2','--workers','0','--batch-size','128')
    assert torch.load(audio,weights_only=True)['chart_pretraining']==saved['chart_pretraining']
