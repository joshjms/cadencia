from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from cadencia.mert import FrozenMert, MertConfig
from cadencia.placement import ACTIONS, PlacementConfig, PlacementTransformer, legal_actions, mask_logits, model_metadata
from cadencia.placement import EMPTY, TAP, HOLD_START, HOLD_END, RELEASE_TAP, RELEASE_HOLD
from cadencia.placement_data import chart_targets, make_sample, build_placement_dataset, PlacementDataset, tempo_at
from cadencia.osumania import load_model
from cadencia.osumania.artifact import read_artifact, validate_chart


class FakeMert:
    config = MertConfig()
    def extract(self, path, *, cache_dir):
        import soundfile as sf
        duration = sf.info(path).duration
        cache = Path(cache_dir) / 'fixture.npz'
        cache.parent.mkdir(parents=True, exist_ok=True)
        value = dict(features=np.ones((int(duration * 75), 768), np.float32),
                     times=np.arange(int(duration * 75)) / 75,
                     duration_s=np.array(duration), identity=np.array('{}'))
        np.savez_compressed(cache, **value)
        return value, cache


def small_config():
    return PlacementConfig(d_model=32, layers=1, history_events=4, dropout=0., joint_patterns=False)


def sample(config=None, held=None):
    features = dict(features=np.ones((300, 768), np.float32), times=np.arange(300) / 75)
    return make_sample(1., 2., 120., 3., np.zeros(4) if held is None else held,
                       [[TAP, EMPTY, EMPTY, EMPTY]], [.5], [1.], features, config or small_config())


def tensors(value):
    return {k: torch.from_numpy(v[None]) for k, v in value.items()}


def test_targets_preserve_chords_and_hold_state(downloaded_chart):
    chart, _, _ = downloaded_chart
    target = chart_targets(chart)
    np.testing.assert_array_equal(target['actions'][25], [TAP, TAP, EMPTY, EMPTY])
    assert target['actions'][50, 2] == HOLD_START
    assert not target['held'][50, 2]
    assert target['held'][51, 2]
    assert target['actions'][75, 2] == HOLD_END
    assert target['held'][75, 2] and not target['held'][76, 2]
    assert len(target['times']) == 200  # Includes negatives; no oracle candidate selection.


def test_release_and_rehit_in_same_cell(downloaded_chart):
    chart, _, _ = downloaded_chart
    lanes = {k: list(v) for k, v in chart.lanes.items()}
    lanes[2].append(replace(lanes[0][0], lane=2, start_s=1.5))
    target = chart_targets(replace(chart, lanes=lanes))
    assert target['actions'][75, 2] == RELEASE_TAP
    lanes[2][-1] = replace(lanes[2][0], start_s=1.5, end_s=2.)
    target = chart_targets(replace(chart, lanes=lanes))
    assert target['actions'][75, 2] == RELEASE_HOLD
    assert target['held'][76, 2]


def test_grid_collision_is_reported(downloaded_chart):
    chart, _, _ = downloaded_chart
    lanes = {k: list(v) for k, v in chart.lanes.items()}
    lanes[0].append(replace(lanes[0][0], start_s=.501))
    with pytest.raises(ValueError, match='Multiple heads'):
        chart_targets(replace(chart, lanes=lanes))


def test_tempo_changes_and_beat_phase():
    bpm, beats = tempo_at(np.array([0., .5, 1., 1.5]), [
        dict(start_s=0., beat_position=0., bpm=120.), dict(start_s=1., beat_position=2., bpm=60.)])
    np.testing.assert_allclose(bpm, [120, 120, 60, 60])
    np.testing.assert_allclose(beats, [0, 1, 2, 2.5])


def test_history_padding_conditioning_and_finite_backward():
    c = small_config()
    model = PlacementTransformer(c)
    for value in [sample(c), make_sample(0., 0., 120., 3., np.zeros(4), [], [], [],
                   dict(features=np.zeros((1,768),np.float32), times=np.array([.008])), c)]:
        inputs = tensors(value)
        logits = model(**inputs)
        assert logits.shape == (1, 4, 6) and torch.isfinite(logits).all()
        logits.sum().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    # Padded history and audio frames must not influence the prediction.
    value = sample(c)
    reference = model(**tensors(value))
    value['history'][value['history_padding']] = 5
    value['history_timing'][value['history_padding']] = 1e5
    value['audio'][value['audio_padding']] = 1e5
    torch.testing.assert_close(reference, model(**tensors(value)))
    baseline = sample(c)
    changed = sample(c); changed['condition'][0] += 1
    assert not torch.allclose(model(**tensors(baseline)), model(**tensors(changed)))
    changed = sample(c); changed['condition'][1] += 1
    assert not torch.allclose(model(**tensors(baseline)), model(**tensors(changed)))


def test_no_target_in_history():
    with pytest.raises(ValueError, match='future target'):
        make_sample(1., 2., 120., 3., np.zeros(4), [[TAP]*4], [1.], [2.],
                    dict(features=np.zeros((1,768),np.float32), times=np.array([.008])), small_config())


def test_legal_hold_transition_masks():
    held = torch.tensor([[False, True, False, True]])
    legal = legal_actions(held)
    assert legal[0, 0].tolist() == [True, True, True, False, False, False]
    assert legal[0, 1].tolist() == [True, False, False, True, True, True]
    probabilities = mask_logits(torch.zeros(1,4,6), held).softmax(-1)
    assert probabilities[0,0,HOLD_END] == 0
    assert probabilities[0,1,TAP] == 0


def test_dataset_offline_integrity_and_history(downloaded_chart, tmp_path):
    chart, root, _ = downloaded_chart
    out = tmp_path / 'placement'
    result = build_placement_dataset(out, data_dir=root, difficulty_labels={str(chart.id):3.5},
                                     extractor=FakeMert(), config=small_config())
    assert result['difficulty_units'].startswith('osu!mania star')
    dataset = PlacementDataset(out, 'train')
    assert len(dataset) == 200
    assert dataset[25]['history_padding'].all()  # Target chord is NOT in history.
    assert dataset[50]['history'][0].tolist() == [TAP,TAP,EMPTY,EMPTY]
    assert dataset[60]['condition'][6] == 1  # Hold state outlives the target window.
    assert dataset[50]['condition'][0] == pytest.approx(.35)
    assert result['charts'][0]['action_counts'][HOLD_START] == 1
    with pytest.raises(FileExistsError):
        build_placement_dataset(out, data_dir=root)
    shard = out / result['charts'][0]['file']
    shard.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        PlacementDataset(out, 'train')


def test_missing_real_difficulty_is_not_substituted(downloaded_chart, tmp_path):
    _, root, _ = downloaded_chart
    with pytest.raises(ValueError, match='star ratings'):
        build_placement_dataset(tmp_path / 'placement', data_dir=root, extractor=FakeMert())
    assert not (tmp_path / 'placement').exists()


def test_four_lane_checkpoint_export_with_frozen_stub(tmp_path, monkeypatch):
    import soundfile as sf
    from cadencia.osumania import placement_generation
    c = small_config()
    transformer = PlacementTransformer(c)
    # Force hold start on every inactive lane; active lanes then release/rehold.
    for p in transformer.parameters(): p.data.zero_()
    transformer.output.bias.data.reshape(4,6)[:, HOLD_START] = 10
    transformer.output.bias.data.reshape(4,6)[:, HOLD_END] = 9
    checkpoint = tmp_path / 'model.pt'
    torch.save(dict(**model_metadata(transformer), model=transformer.state_dict(), mert=asdict(MertConfig())), checkpoint)
    model = load_model(checkpoint)
    monkeypatch.setattr(model, 'extractor', FakeMert())
    audio = tmp_path / 'song.wav'
    sf.write(audio, np.zeros(4000), 8000)
    output = model.create_chart(audio, out=tmp_path/'chart.cadencia', difficulty=3., bpm=120.)
    value, _ = read_artifact(output)
    assert set(n['lane'] for n in value['notes']) == {0,1,2,3}
    assert any(n['kind']=='hold' for n in value['notes'])
    assert all(n['end_s']>n['time_s'] for n in value['notes'] if n['kind']=='hold')
    assert value['difficulty'] == 3. and value['bpm'] == 120.
    assert value['notes'] == model.predict(audio, difficulty=3., bpm=120.)['notes']
    with pytest.raises(ValueError): model.predict(audio, difficulty=-1.)


def test_artifact_rejects_hold_overlap():
    chart = dict(duration_s=3., threshold=0., candidates=8, difficulty=3., notes=[
        dict(time_s=1., lane=0, kind='hold',end_s=2.,probability=.9),
        dict(time_s=1.5,lane=0,kind='tap',probability=.8)])
    with pytest.raises(ValueError, match='Overlapping'):
        validate_chart(chart, placement=True)


def test_mert_cache_reuse_invalidation_and_frame_alignment(tmp_path):
    import soundfile as sf
    from types import SimpleNamespace
    calls = []
    class FakeEncoder(torch.nn.Module):
        def forward(self, input_values, **kwargs):
            calls.append(len(input_values[0]))
            count = (input_values.shape[-1] - 400) // 320 + 1
            hidden = torch.ones(1,count,768)
            return SimpleNamespace(hidden_states=[hidden]*13)
    path = tmp_path/'audio.wav'
    sf.write(path,np.zeros(24000*8),24000)
    encoder=FrozenMert()
    encoder.model=FakeEncoder().eval()
    encoder.processor=lambda values, **kwargs: {'input_values':torch.from_numpy(values[None])}
    value,cache=encoder.extract(path,cache_dir=tmp_path/'cache')
    assert len(calls)==2 and value['features'].shape==(599,768)
    np.testing.assert_allclose(np.diff(value['times']),1/75,atol=1e-12)
    again,again_path=encoder.extract(path,cache_dir=tmp_path/'cache')
    assert len(calls)==2 and cache==again_path
    np.testing.assert_array_equal(value['features'],again['features'])
    sf.write(path,np.ones(24000*8)*.1,24000)
    _,different=encoder.extract(path,cache_dir=tmp_path/'cache')
    assert different!=cache and len(calls)==4


def test_fetch_missing_star_labels_without_downloads(downloaded_chart, monkeypatch):
    from scripts import make_placement_dataset
    chart, root, _ = downloaded_chart
    calls=[]
    def fetch(set_id):
        calls.append(set_id)
        return {chart.id:3.75}
    monkeypatch.setattr(make_placement_dataset,'_fetch_set_ratings',fetch)
    def unavailable(*_): raise FileNotFoundError('fixture forces metadata fallback')
    monkeypatch.setattr(make_placement_dataset,'_calculate_local_rating',unavailable)
    labels=make_placement_dataset.fetch_difficulty_labels(root,{})
    assert labels=={str(chart.id):3.75}
    assert calls==[chart.metadata['beatmapset_id']]
    assert make_placement_dataset.fetch_difficulty_labels(root,{})==labels
    assert len(calls)==1
    # Changed chart bytes invalidate a cached rating.
    path=root/'osumania/charts'/str(chart.id)/'trace.json'
    value=json.loads(path.read_text());value['metadata']['chart_sha256']='changed'
    path.write_text(json.dumps(value))
    make_placement_dataset.fetch_difficulty_labels(root,{})
    assert len(calls)==2


def test_official_rating_parser_and_identity(monkeypatch):
    import io
    from scripts import make_placement_dataset as prepare
    payload={'id':123,'beatmaps':[{'id':456,'difficulty_rating':4.2}]}
    page=f"<script type='application/json' id='json-beatmapset'>{json.dumps(payload)}</script>"
    monkeypatch.setattr(prepare.urllib.request,'urlopen',lambda *a,**kw:io.BytesIO(page.encode()))
    assert prepare._fetch_set_ratings(123)=={456:4.2}
    monkeypatch.setattr(prepare.time,'sleep',lambda *_:None)
    with pytest.raises(RuntimeError,match='identity mismatch'):
        prepare._fetch_set_ratings(999)


def test_packed_batches_match_individual_queries_and_gradients(downloaded_chart, tmp_path):
    from cadencia.placement_data import PackedPlacementDataset, BlockShuffleSampler
    chart, root, _=downloaded_chart
    out=tmp_path/'packed'
    build_placement_dataset(out,data_dir=root,difficulty_labels={str(chart.id):3.5},
                            extractor=FakeMert(),config=small_config())
    dataset=PackedPlacementDataset(out,'train')
    indices=[0,25,50,60,75,199]
    packed=dataset.__getitems__(indices)
    samples=[dataset[index] for index in indices]
    standard={key:torch.from_numpy(np.stack([sample[key] for sample in samples])) for key in samples[0]}
    torch.testing.assert_close(packed['target'],standard['target'])
    for key in ['history','history_timing','history_padding','condition','audio_times','audio_padding']:
        torch.testing.assert_close(packed[key],standard[key])
    standard.pop('target');packed.pop('target')
    a=PlacementTransformer(small_config()).eval()
    b=PlacementTransformer(small_config()).eval();b.load_state_dict(a.state_dict())
    expected=a(**standard); actual=b(**packed)
    torch.testing.assert_close(expected,actual,atol=2e-6,rtol=2e-5)
    expected.sum().backward();actual.sum().backward()
    for pa,pb in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(pa.grad,pb.grad,atol=2e-5,rtol=2e-4)
    sampler=BlockShuffleSampler(dataset)
    assert sorted(sampler)==list(range(len(dataset)))


def test_offline_ratings_are_cached_and_versioned(downloaded_chart, monkeypatch):
    pytest.importorskip('rosu_pp_py')
    from scripts import make_placement_dataset as prepare
    chart,root,_=downloaded_chart
    def unexpected(*_): pytest.fail('Complete cached charts should not need network metadata')
    monkeypatch.setattr(prepare,'_fetch_set_ratings',unexpected)
    labels=prepare.fetch_difficulty_labels(root,{})
    assert labels[str(chart.id)]>0
    cache=json.loads((root/'star_rating_cache.json').read_text())
    record=cache['ratings'][str(chart.id)]
    assert record['source'].startswith('rosu-pp-py/')
    assert record['chart_sha256']==chart.metadata['chart_sha256']
    assert prepare.fetch_difficulty_labels(root,{})==labels


def test_weighted_epoch_loss_is_independent_of_batch_boundaries():
    from torch.utils.data import DataLoader
    from scripts.train_placement import run_epoch
    class Model(torch.nn.Module):
        def forward(self,condition):
            score=condition[:,0]
            logits=torch.stack([score,-score,torch.zeros_like(score),torch.zeros_like(score),
                                torch.zeros_like(score),torch.zeros_like(score)],dim=-1)
            return logits[:,None].expand(-1,4,-1)
    records=[]
    for i in range(7):
        condition=torch.zeros(8);condition[0]=i/3
        records.append({'condition':condition,'target':torch.full((4,),i%2,dtype=torch.long)})
    weights=torch.tensor([.1,4.,1.,1.,1.,1.])
    a=run_epoch(Model(),DataLoader(records,batch_size=1),torch.device('cpu'),weights,log_every=0)
    b=run_epoch(Model(),DataLoader(records,batch_size=4),torch.device('cpu'),weights,log_every=0)
    assert a['loss']==pytest.approx(b['loss'],rel=1e-6)
    assert a['confusion']==b['confusion']


def test_half_precision_matrix_is_shared_with_dataset(downloaded_chart,tmp_path):
    from cadencia.placement_data import PackedPlacementDataset
    chart,root,_=downloaded_chart
    class HalfMert(FakeMert):
        def extract(self,path,*,cache_dir):
            features,cache=super().extract(path,cache_dir=cache_dir)
            features['features']=features['features'].astype(np.float16)
            np.save(cache.with_suffix('.npy'),features['features'])
            return features,cache
    out=tmp_path/'half'
    manifest=build_placement_dataset(out,data_dir=root,difficulty_labels={str(chart.id):3.},
                                     extractor=HalfMert(),config=small_config())
    file=out/manifest['charts'][0]['feature_file']
    cached=root/'mert_cache'/file.name
    assert file.stat().st_ino==cached.stat().st_ino
    assert np.load(file,mmap_mode='r').dtype==np.float16
    batch=PackedPlacementDataset(out,'train').__getitems__([0,25,50])
    batch.pop('target')
    assert torch.isfinite(PlacementTransformer(small_config())(**batch)).all()
