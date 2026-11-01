from dataclasses import asdict,replace
import hashlib,json,subprocess,sys
import numpy as np
import pytest
import torch
from cadencia.rhythm import from_predictions,bar_positions,rhythm_at,SPEC
from cadencia.planner import BarPlanner,PlannedTransformer,bar_inputs,metadata
from cadencia.placement_data import make_sample
from tests.test_high_resolution import config
from tests.test_chart_pretraining import dataset


def analysis():
    beat=np.zeros(150);beat[::25]=.99
    down=np.zeros(150);down[[0,75]]=.95
    return from_predictions(beat,down,np.arange(0,3,.5),np.array([0.,1.5]),np.ones(300),np.arange(300)/100,3.)


def test_detected_bars_follow_downbeats_not_fixed_four_beats():
    a=analysis()
    positions,detected=bar_positions([0.,.75,1.5,2.25],a['beat_times'],a['downbeat_times'])
    np.testing.assert_allclose(positions,[0,.5,1,1.5]);assert detected
    _,detected=bar_positions([0.,.5],a['beat_times'],[]);assert not detected
    assert rhythm_at(a,[0.,.75]).shape==(2,6)
    assert a['features'][0,0]==pytest.approx(.99)
    assert a['features'][0,5]>0
    bad=np.zeros(150);bad[0]=np.nan
    with pytest.raises(ValueError):from_predictions(bad,bad,[0,.5],[],[1,1],[0,1],1)


def test_planner_and_decoder_receive_rhythm_gradients():
    a=analysis();c=replace(config(),use_rhythm=True)
    features=dict(times=np.arange(75)/25,features=np.zeros((75,1024),np.float32),mel=np.zeros((301,64),np.float32),rhythm=a)
    bars,rows=bar_inputs(features,np.arange(300)/100,np.arange(300)/50,np.full(300,120),2.)
    assert bars.shape[1]==1024*2+3+6
    assert rows[149]==0 and rows[150]==1
    planner=BarPlanner(1024,use_rhythm=True).eval()
    assert planner.predict(torch.from_numpy(bars)[None]).shape==(1,2,4)
    sample=make_sample(.75,1.5,120,2,np.zeros(4),[],[],[],features,c)
    model=PlannedTransformer(c)
    output=model(plan=torch.zeros(1,4),**{k:torch.from_numpy(v[None]) for k,v in sample.items()})
    output['event'].sum().backward()
    assert model.rhythm_projection[0].weight.grad.abs().sum()>0
    assert metadata(model)['rhythm_backend']==SPEC
    from cadencia.chart_pretraining import ChartSequenceModel,initialize_decoder,split_identities,FORMAT
    source=ChartSequenceModel(replace(c,use_rhythm=False,use_mel=False,grid_hz=50,audio_dim=768,audio_hz=75))
    manifest=dict(charts=[dict(chart_id=i,split=split,group_id=i,song_id=i,audio_sha256=str(i)) for i,split in enumerate(['train','validation','test'])])
    checkpoint=dict(format=FORMAT,config=asdict(source.config),model=source.state_dict(),best_epoch=1,dataset_sha256='fixture',split_identities=split_identities(manifest))
    before=model.rhythm_projection[0].weight.detach().clone()
    transferred=initialize_decoder(model,checkpoint,manifest)
    assert 'rhythm_projection.0.weight' in transferred['fresh_keys']
    torch.testing.assert_close(model.rhythm_projection[0].weight,before)



def test_low_beat_confidence_disables_grid_snapping():
    from cadencia.beat_alignment import align_chart
    a=analysis();a['features'][:,0]=0
    c=dict(notes=[dict(time_s=.512,lane=0,kind='tap',probability=.8)],duration_s=3.,timing_map=a['timing_map'],candidates=1,threshold=0.)
    r=align_chart(c,rhythm_evidence=a)
    assert r['notes'][0]['time_s']==.512


def test_rhythm_dataset_training_and_checkpoint_loading(dataset,tmp_path):
    root,m=dataset;c=replace(config(),use_rhythm=True);a=analysis()
    m['config']=asdict(c)
    from cadencia.mert import MertConfig
    m['mert']=asdict(MertConfig.v2())
    np.save(root/'features.npy',np.zeros((75,1024),np.float16))
    np.save(root/'mel.npy',np.zeros((301,64),np.float16))
    np.savez(root/'audio.npz',times=np.arange(75)/25,duration_s=np.array(3.))
    meta=dict(spec=SPEC,timing_map=a['timing_map'],bar_boundaries_detected=True)
    np.savez(root/'rhythm.npz',**{k:a[k] for k in ['times','features','bar_positions','beat_times','downbeat_times']},metadata=np.array(json.dumps(meta)))
    for e in m['charts']:
        with np.load(root/e['file']) as arrays:t={k:arrays[k] for k in arrays.files}
        t['times']=np.arange(300)/100;t['beats']=t['times']*2
        np.savez(root/e['file'],**t)
        e.update(mel_file='mel.npy',rhythm_file='rhythm.npz')
        for key,checksum in [('file','sha256'),('audio_file','audio_file_sha256'),('feature_file','feature_sha256'),('mel_file','mel_sha256'),('rhythm_file','rhythm_sha256')]:e[checksum]=hashlib.sha256((root/e[key]).read_bytes()).hexdigest()
    (root/'manifest.json').write_text(json.dumps(m))
    from cadencia.interval import IntervalDataset
    ds=IntervalDataset(root,'train');batch=ds.__getitems__([0,50,100])
    np.testing.assert_allclose(batch['rhythm'][1],ds[50]['rhythm'])
    planner=tmp_path/'planner.pt';decoder=tmp_path/'decoder.pt'
    for script,args in [('scripts/train_bar_planner.py',['--output',str(planner)]),('scripts/train_intervals.py',['--planner',str(planner),'--output',str(decoder),'--batch-size','64','--workers','0'])]:
        result=subprocess.run([sys.executable,script,'--data',str(root),'--epochs','1',*args],capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
    from cadencia.osumania import load_model
    loaded=load_model(decoder)
    assert loaded.config.use_rhythm
    assert loaded.planner.input.in_features==2057
