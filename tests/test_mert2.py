"""MERT-v2 interface compatibility; real checkpoint execution needs matched torchaudio."""
from dataclasses import asdict,replace
import numpy as np
import torch
from cadencia.mert import MertConfig,FrozenMert,read_features
from cadencia.placement import PlacementConfig
from cadencia.placement_data import make_sample
from cadencia.planner import BarPlanner,bar_inputs,PlannedTransformer
from cadencia.chart_pretraining import ChartSequenceModel,initialize_decoder,FORMAT,split_identities


def test_v2_chunk_timestamps_mask_and_cache(tmp_path,monkeypatch):
    import cadencia.mert as module
    from types import SimpleNamespace
    samples=np.zeros((int(29.2*24000),1),np.float32)
    monkeypatch.setattr(module,'read_audio',lambda path:(samples,24000,{}))
    extractor=FrozenMert(config=MertConfig.v2())
    class Model:
        def __call__(self,input_values,**kwargs):
            n=input_values.shape[-1]//960
            hidden=torch.zeros(1,n,1024)
            return SimpleNamespace(hidden_states=(hidden,)*24,feature_attention_mask=torch.ones(1,n,dtype=torch.bool))
    extractor.model=Model()
    extractor.processor=lambda segment,**kwargs:dict(input_values=torch.tensor(segment[None]))
    song=tmp_path/'input.wav';song.write_bytes(b'fixture')
    features,cache=extractor.extract(song,cache_dir=tmp_path/'cache')
    assert features['features'].shape[1]==1024
    assert features['times'][0]==.02
    np.testing.assert_allclose(np.diff(features['times']),.04,atol=1e-10)
    assert features['times'][-1]<29.2
    restored=read_features(cache)
    np.testing.assert_array_equal(restored['features'],features['features'])
    assert cache.name!=MertConfig().revision


def test_v2_planner_decoder_and_v1_chart_weight_transfer():
    features=dict(times=np.arange(100)/25,features=np.zeros((100,1024),np.float32))
    config=PlacementConfig(d_model=32,layers=1,history_events=4,dropout=0.,audio_dim=1024,audio_hz=25)
    sample=make_sample(1.,2.,120.,3.,np.zeros(4,bool),[],[],[],features,config)
    assert sample['audio'].shape==(52,1024)
    inputs={k:torch.from_numpy(v[None]) for k,v in sample.items()}
    bars,rows=bar_inputs(features,np.arange(200)/50,np.arange(200)/25,np.full(200,120),3.)
    plan=BarPlanner(1024).eval().predict(torch.from_numpy(bars)[None])[:,0]
    model=PlannedTransformer(config)
    assert model(plan=plan,**inputs)['pattern'].shape==(1,80)
    chart=ChartSequenceModel(replace(config,audio_dim=768,audio_hz=75))
    manifest=dict(charts=[dict(chart_id=i,split=s,group_id=i,song_id=i,audio_sha256=str(i)) for i,s in enumerate(['train','validation','test'])])
    saved=dict(format=FORMAT,config=asdict(chart.config),model=chart.state_dict(),best_epoch=1,dataset_sha256='fixture',split_identities=split_identities(manifest))
    report=initialize_decoder(model,saved,manifest)
    assert 'pattern.weight' in report['transferred_keys']
    assert model.audio.weight.shape[1]==1024
