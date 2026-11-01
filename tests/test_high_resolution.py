from dataclasses import asdict
import copy
import hashlib
import json
import subprocess
import sys
import numpy as np
import pytest
import torch
from cadencia.high_resolution import timing_from_beats,time_at_beat,mel_windows,extract_mel
from cadencia.placement import PlacementConfig,ACTIONS
from cadencia.placement_data import make_sample,tempo_at,SCHEMA
from cadencia.planner import PlannedTransformer,metadata,HIGH_RES_FORMAT
from cadencia.mert import MertConfig
from cadencia.osumania.artifact import validate_chart
from scripts.convert_cadencia import osu_text


def config():return PlacementConfig(d_model=32,layers=1,history_events=4,dropout=0.,audio_dim=1024,audio_hz=25,grid_hz=100,use_mel=True)


def test_variable_timing_and_hold_duration_crossing_tempo_change():
    timing=timing_from_beats([0.,.5,1.,1.4,1.8])
    bpm,beats=tempo_at(np.array([.75,1.2,1.6]),timing)
    np.testing.assert_allclose(bpm,[120,150,150])
    np.testing.assert_allclose(beats,[1.5,2.5,3.5])
    assert time_at_beat(3.,timing)==pytest.approx(1.4)
    chart=dict(duration_s=3.,notes=[],difficulty=2.,bpm=120.,timing_map=timing,threshold=0.,candidates=1200)
    restored=validate_chart(chart,placement=True)
    assert restored['timing_map']==timing
    exported=osu_text(restored)
    assert '1000.000000,400.000000000' in exported
    bad=copy.deepcopy(chart);bad['timing_map'][1]['start_s']=0
    with pytest.raises(ValueError):validate_chart(bad,placement=True)


def test_real_mel_frame_timing_cache_and_padding(tmp_path):
    import soundfile as sf
    path=tmp_path/'click.wav'
    samples=np.zeros(24000,np.float32);samples[12000]=1.
    sf.write(path,samples,24000,subtype='FLOAT')
    matrix,cache=extract_mel(path,tmp_path/'cache')
    assert abs(int(np.argmax(matrix.sum(1)))-50)<=1
    same,other=extract_mel(path,tmp_path/'cache')
    assert cache==other;np.testing.assert_array_equal(matrix,same)
    batch=mel_windows(matrix,[0.,.5])
    assert batch['mel'].shape==(2,101,64)
    assert batch['mel_padding'][0,:50].all()
    assert not batch['mel_padding'][1].any()


def test_mel_branch_gradients_and_padding_invariance():
    c=config()
    features=dict(times=np.arange(100)/25,features=np.zeros((100,1024),np.float32),mel=np.random.default_rng(0).random((400,64)).astype(np.float32))
    sample=make_sample(0.,0.,120.,2.,np.zeros(4),[],[],[],features,c)
    inputs={k:torch.from_numpy(v[None]) for k,v in sample.items()}
    model=PlannedTransformer(c).eval();plan=torch.tensor([[.1,1.,.1,.1]])
    output=model(plan=plan,**inputs)
    (output['event'].sum()+output['pattern'].sum()+output['duration'].sum()).backward()
    assert model.mel_encoder[0].weight.grad.abs().sum()>0
    assert torch.isfinite(model.mel_encoder[0].weight.grad).all()
    original=output['event'].detach()
    inputs['mel'][inputs['mel_padding']]=10000
    torch.testing.assert_close(model(plan=plan,**inputs)['event'],original)
    assert metadata(model)['format']==HIGH_RES_FORMAT


def test_highres_training_checkpoint_and_generation(tmp_path,monkeypatch):
    import soundfile as sf
    root=tmp_path/'dataset';root.mkdir()
    times=np.arange(300)/100;actions=np.zeros((300,4),np.int64)
    actions[51,0]=1;actions[100,1]=2;actions[175,1]=3;actions[201,2]=1
    held=np.zeros((300,4),bool);held[101:176,1]=True
    target=dict(times=times,beats=times*2,bpm=np.full(300,120.),actions=actions,held=held,groups=np.flatnonzero(actions.any(-1)))
    np.save(root/'features.npy',np.zeros((75,1024),np.float16))
    np.save(root/'mel.npy',np.zeros((301,64),np.float16))
    np.savez(root/'audio.npz',times=np.arange(75)/25,duration_s=np.array(3.))
    entries=[]
    for i,split in enumerate(['train','validation','test']):
        file=f'{i}.npz';np.savez(root/file,**target)
        entry=dict(chart_id=i,song_id=i,group_id=str(i),audio_sha256=str(i),split=split,rows=300,difficulty=2.,file=file,audio_file='audio.npz',feature_file='features.npy',mel_file='mel.npy')
        for key,checksum in [('file','sha256'),('audio_file','audio_file_sha256'),('feature_file','feature_sha256'),('mel_file','mel_sha256')]:entry[checksum]=hashlib.sha256((root/entry[key]).read_bytes()).hexdigest()
        entries.append(entry)
    (root/'manifest.json').write_text(json.dumps(dict(schema=SCHEMA,actions=ACTIONS,config=asdict(config()),mert=asdict(MertConfig.v2()),charts=entries)))
    from cadencia.planner import PlannedDataset
    from cadencia.interval import IntervalDataset
    ds=IntervalDataset(root,'train')
    packed=ds.__getitems__([51,100])
    single=ds[51]
    np.testing.assert_array_equal(packed['mel'][0],single['mel'])
    assert packed['target'][0,0]==1
    planner=tmp_path/'planner.pt';decoder=tmp_path/'decoder.pt'
    for script,args in [('scripts/train_bar_planner.py',['--output',str(planner)]),('scripts/train_intervals.py',['--planner',str(planner),'--output',str(decoder),'--batch-size','64','--workers','0'])]:
        result=subprocess.run([sys.executable,script,'--data',str(root),'--epochs','1',*args],text=True,capture_output=True)
        assert result.returncode==0,result.stdout+result.stderr
    import cadencia.osumania.placement_generation as generation
    class Extractor:
        def __init__(self,**kw):pass
        def extract(self,*args,**kw):return dict(times=np.arange(75)/25,features=np.zeros((75,1024),np.float32),duration_s=3.),None
    monkeypatch.setattr(generation,'FrozenMert',Extractor)
    import cadencia.high_resolution as high
    monkeypatch.setattr(high,'tracked_timing',lambda *args,**kw:timing_from_beats([0.,.5,1.,1.4,1.8,2.2,2.6]))
    wav=tmp_path/'song.wav';sf.write(wav,np.zeros(72000,np.float32),24000)
    from cadencia.osumania import load_model
    model=load_model(decoder)
    result=model.predict(wav,difficulty=2,decoding='argmax')
    assert result['candidates']==300*4
    assert len(result['timing_map'])==7
    validate_chart(result,placement=True)
