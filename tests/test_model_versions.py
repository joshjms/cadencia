import hashlib,json
import pytest
import cadencia.model_versions as versions


def test_release_verifies_artifacts_and_rejects_modified_files(tmp_path,monkeypatch):
    monkeypatch.setattr(versions,'ROOT',tmp_path)
    (tmp_path/'model_releases').mkdir()
    path=tmp_path/'decoder.pt';path.write_bytes(b'weights')
    manifest=dict(schema=versions.SCHEMA,version='v1',artifacts=dict(decoder=dict(path='decoder.pt',sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    (tmp_path/'model_releases/v1.json').write_text(json.dumps(manifest))
    assert versions.resolve_model_version('v1')==path
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):versions.resolve_model_version('v1')
    with pytest.raises(ValueError):versions.resolve_model_version('../other')


def test_major_release_tags_and_rejected_paths():
    assert versions.DEFAULT_MODEL_VERSION == 'v1'
    assert versions.is_model_version('v1')
    assert versions.is_model_version('v1.2.3')
    for value in ('../v1', 'v1/decoder.pt', 'v1.2', '1', ''):
        assert not versions.is_model_version(value)


def test_explicit_last_selection_and_immutable_publication(tmp_path, monkeypatch):
    import torch
    from scripts import register_model_release as release
    monkeypatch.setattr(release, 'ROOT', tmp_path)
    monkeypatch.setattr(versions, 'ROOT', tmp_path)
    planner = tmp_path/'planner.pt'
    pretraining = tmp_path/'pretraining.pt'
    decoder = tmp_path/'decoder.last.pt'
    common = dict(format='fixture', best_epoch=1, dataset_sha256='dataset', model={'weight': torch.ones(1)})
    torch.save(dict(common, mert={}), planner)
    torch.save(common, pretraining)
    source = dict(common, mert={}, epochs_completed=8,
                  last_model={'weight': torch.full((1,), 8.)}, optimizer={'unused': True},
                  planner_sha256=release.digest(planner),
                  chart_pretraining=dict(sha256=release.digest(pretraining), path='machine-specific'),
                  history=[dict(epoch=8, validation={'base_loss': 2.})])
    torch.save(source, decoder)
    manifest_path = release.publish('v1', decoder, planner, pretraining, weights='last')
    manifest = json.loads(manifest_path.read_text())
    published = versions.resolve_model_version()
    saved = torch.load(published, weights_only=True)
    assert saved['model']['weight'].item() == 8
    assert saved['weights_epoch'] == 8 and saved['architecture_version'] == 'v1'
    assert 'last_model' not in saved and 'optimizer' not in saved
    assert 'path' not in saved['chart_pretraining']
    assert manifest['validation'] == {'base_loss': 2.}
    # Later training updates cannot mutate the released inference weights.
    torch.save(dict(source, last_model={'weight': torch.full((1,), 9.)}), decoder)
    assert torch.load(versions.resolve_model_version(), weights_only=True)['model']['weight'].item() == 8
    with pytest.raises(FileExistsError):
        release.publish('v1', decoder, planner, pretraining, weights='last')
    assert release.inference_checkpoint(source, 'v2')['model']['weight'].item() == 1
    with pytest.raises(ValueError, match='resumable'):
        release.inference_checkpoint(common, 'v2', 'last')


def test_bundled_release_loads_without_fetching_audio_models():
    import torch
    from cadencia.osumania import load_model
    from cadencia.beat_alignment import load_prior
    path = versions.resolve_model_version()
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = load_model('v1')
    assert model.planned and model.config.grid_hz == 100
    assert model.config.use_mel and model.config.use_rhythm
    assert checkpoint['weights_selection'] == 'last' and checkpoint['weights_epoch'] > 0
    for key, value in model.model.state_dict().items():
        torch.testing.assert_close(value, checkpoint['model'][key])
    assert load_prior(model.alignment_prior_path)[1] == checkpoint['alignment_prior_sha256']
