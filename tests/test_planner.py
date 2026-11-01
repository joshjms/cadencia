"""Bar supervision, global audio context, and planner-conditioned generation."""
from dataclasses import asdict
import numpy as np
import torch
import pytest
from cadencia.planner import (bar_inputs,bar_targets,BarPlanner,planner_loss,PlannedTransformer,
                              metadata,PLANNER_FORMAT,PLAN_FIELDS,PlannedDataset)
from cadencia.mert import MertConfig
from cadencia.osumania import load_model
from tests.test_joint_placement import config,inputs


def song():
    times=np.arange(600)/50
    beats=times*2
    features=dict(times=np.arange(900)/75,features=np.ones((900,768),np.float32))
    actions=np.zeros((600,4),int);actions[100,0]=2;actions[250,0]=3;actions[110,1]=1
    held=np.zeros((600,4),bool);held[101:251,0]=True
    return dict(times=times,beats=beats,bpm=np.full(600,120.),actions=actions,held=held),features


def test_bar_alignment_partial_bars_and_hold_occupancy():
    target,features=song()
    bars,rows=bar_inputs(features,target['times'],target['beats'],target['bpm'],3.)
    assert bars.shape==(6,1539)
    assert rows[99]==0 and rows[100]==1
    expected=bar_targets(target,rows)
    assert expected[0,0]==1 and expected[0,1]==0
    assert expected[1,0]==0 and expected[1,1]>0
    assert expected[2,3]>0 # A hold continuing across bars contributes occupancy.
    assert expected[2,0]==1 # No new heads in this bar.
    assert np.isfinite(expected).all()
    shifted=target['beats']-1
    _,partial=bar_inputs(features,target['times'],shifted,target['bpm'],3.)
    assert partial[0]==0 and partial[25]==1


def test_planner_global_context_gradient_and_output_ranges():
    torch.manual_seed(42)
    planner=BarPlanner().eval()
    bars=torch.randn(1,6,1539)
    first=planner.predict(bars)
    changed=bars.clone();changed[:,5]+=5
    assert not torch.allclose(first[:,0],planner.predict(changed)[:,0])
    assert ((first[...,[0,2,3]]>=0)&(first[...,[0,2,3]]<=1)).all()
    assert (first[...,1]>=0).all()
    loss=planner_loss(planner(bars),torch.zeros(1,6,4))
    loss.backward()
    assert planner.input.weight.grad.abs().sum()>0


def test_decoder_uses_plan_and_rest_prior():
    model=PlannedTransformer(config()).eval()
    for parameter in model.parameters():parameter.data.zero_()
    ordinary=torch.tensor([[0.,1.,.2,.1]])
    rest=torch.tensor([[.99,0.,0.,0.]])
    active=model(plan=ordinary,**inputs())['event'].sigmoid()
    quiet=model(plan=rest,**inputs())['event'].sigmoid()
    assert quiet.item()<active.item()/20
    model=PlannedTransformer(config())
    output=model(plan=ordinary,**inputs())
    output['event'].sum().backward()
    assert model.plan_projection[0].weight.grad.abs().sum()>0


def test_planned_checkpoint_loads_embedded_planner(tmp_path,monkeypatch):
    import cadencia.osumania.placement_generation as generation
    planner=BarPlanner()
    for p in planner.parameters():p.data.zero_()
    planner.output.bias.data[0]=30 # Near-certain rest from audio planner.
    model=PlannedTransformer(config())
    for p in model.parameters():p.data.zero_()
    saved_planner=dict(format=PLANNER_FORMAT,model=planner.state_dict(),mert=asdict(MertConfig()))
    path=tmp_path/'planned.pt'
    torch.save(dict(**metadata(model),model=model.state_dict(),planner=saved_planner,mert=asdict(MertConfig())),path)
    class Extractor:
        def __init__(self,**kwargs):pass
        def extract(self,*args,**kwargs):
            return dict(times=np.arange(150)/75,features=np.zeros((150,768),np.float32),duration_s=2.),None
    monkeypatch.setattr(generation,'FrozenMert',Extractor)
    loaded=load_model(path)
    result=loaded.predict(tmp_path/'song.mp3',difficulty=2,bpm=120,decoding='argmax')
    assert result['notes']==[]
    assert loaded.planned


def test_planned_dataset_uses_audio_predictions_not_labels(monkeypatch):
    from cadencia.interval import IntervalDataset
    ds=object.__new__(PlannedDataset)
    target,features=song()
    ds.entries=[dict(difficulty=3.)]
    ds.ends=np.array([600])
    ds._load=lambda i:(target,features)
    monkeypatch.setattr(IntervalDataset,'__getitems__',lambda self,indices:{})
    planner=BarPlanner().eval()
    ds.prepare_plans(planner)
    batch=ds.__getitems__([0,1])
    plans,rows=ds.plans[0]
    np.testing.assert_allclose(batch['plan'].numpy(),plans[rows[:2]])
    assert 'plan_target' not in batch


def test_planner_and_decoder_training_resume(tmp_path):
    import json,hashlib,subprocess,sys
    from cadencia.placement_data import SCHEMA
    from cadencia.placement import ACTIONS
    target,features=song()
    target['groups']=np.flatnonzero(target['actions'].any(-1))
    root=tmp_path/'dataset';root.mkdir()
    np.save(root/'features.npy',features['features'],allow_pickle=False)
    np.savez(root/'audio.npz',times=features['times'],duration_s=np.array(12.))
    np.savez(root/'targets.npz',**target)
    entries=[]
    for i,split in enumerate(['train','validation','test']):
        entry=dict(chart_id=i,split=split,rows=600,difficulty=3.,file='targets.npz',audio_file='audio.npz',feature_file='features.npy')
        for key,checksum in [('file','sha256'),('audio_file','audio_file_sha256'),('feature_file','feature_sha256')]:
            entry[checksum]=hashlib.sha256((root/entry[key]).read_bytes()).hexdigest()
        entries.append(entry)
    (root/'manifest.json').write_text(json.dumps(dict(schema=SCHEMA,actions=ACTIONS,config=asdict(config()),mert=asdict(MertConfig()),charts=entries)))
    planner=tmp_path/'planner.pt';decoder=tmp_path/'decoder.pt'
    def run(*args):
        result=subprocess.run([sys.executable,*args],capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
    run('scripts/train_bar_planner.py','--data',str(root),'--output',str(planner),'--epochs','2')
    run('scripts/train_intervals.py','--data',str(root),'--planner',str(planner),'--output',str(decoder),'--epochs','1','--batch-size','128','--workers','0')
    run('scripts/train_intervals.py','--data',str(root),'--planner',str(planner),'--output',str(decoder),'--resume',str(decoder.with_suffix('.last.pt')),'--epochs','2','--batch-size','128','--workers','0')
    loaded=load_model(decoder)
    assert loaded.planned
    metrics=json.loads(decoder.with_suffix('.metrics.json').read_text())
    assert metrics['epochs_completed']==2
    assert metrics['test']['rows']==600
