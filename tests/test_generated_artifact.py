import hashlib
import io
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

import numpy as np
import pytest
import soundfile as sf

from cadencia.osumania import Simulator, SimulatorConfig, load_model
from cadencia.osumania.artifact import read_artifact, write_artifact


def test_model_exports_portable_chart_and_simulator_plays_without_model(downloaded_chart, tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from cadencia.dataset import FEATURES
    from cadencia.model import HitTransformer

    chart, root, _ = downloaded_chart
    transformer = HitTransformer()
    for parameter in transformer.parameters():
        parameter.data.zero_()
    transformer.output.bias.data.fill_(10)
    checkpoint = root / 'model.pt'
    torch.save(dict(model=transformer.state_dict(), mean=torch.zeros(20), scale=torch.ones(20),
                    threshold=.5, features=FEATURES), checkpoint)
    original_audio = chart.song.path.read_bytes()
    model = load_model(checkpoint)
    def no_reload(*args, **kwargs):
        pytest.fail('Model was reloaded during generation or saved-chart playback')
    monkeypatch.setattr(torch, 'load', no_reload)
    output = tmp_path / 'exports' / 'song.cadencia'
    assert model.create_chart(chart.song.path, out=output) == output
    assert model.create_chart(chart.song.path, out=tmp_path / 'second.cadencia').is_file()
    with pytest.raises(FileExistsError):
        model.create_chart(chart.song.path, out=output)
    assert chart.song.path.read_bytes() == original_audio
    saved, audio = read_artifact(output)
    assert saved['notes'] and all(note['lane'] == 0 for note in saved['notes'])
    assert saved['model_sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert sf.info(io.BytesIO(audio)).duration == chart.song.duration_s
    # Move just the artifact, then remove original audio and checkpoint.
    moved = tmp_path / 'moved.cadencia'
    shutil.move(output, moved)
    chart.song.path.unlink()
    checkpoint.unlink()
    app = Simulator(moved, port=0, config=SimulatorConfig(tile_speed=500), block=False, open_browser=False)
    try:
        with urllib.request.urlopen(app.url + '/api/status') as response:
            state = json.load(response)
        assert state['status'] == 'ready'
        assert state['notes'] == saved['notes']
        with urllib.request.urlopen(app.url + '/audio.wav') as response:
            assert response.read() == audio
        with urllib.request.urlopen(app.url + '/api/config') as response:
            assert json.load(response)['tile_speed'] == 500
    finally:
        app.close()
    assert moved.is_file()


@pytest.fixture
def artifact(tmp_path):
    audio = tmp_path / 'source.wav'
    sf.write(audio, np.zeros(3000), 1000)
    result = dict(notes=[dict(time_s=2.5, lane=0, probability=.9)], duration_s=3,
                  threshold=.5, candidates=1)
    output = tmp_path / 'song.cadencia'
    write_artifact(output, result, audio)
    return output


@pytest.mark.parametrize('corruption', ['checksum', 'lane', 'time', 'duration', 'version', 'traversal'])
def test_artifact_rejects_invalid_content(artifact, tmp_path, corruption):
    with zipfile.ZipFile(artifact) as source:
        manifest = json.loads(source.read('chart.json'))
        audio = source.read('audio.wav')
    if corruption == 'checksum': manifest['audio_sha256'] = 'incorrect'
    elif corruption == 'lane': manifest['chart']['notes'][0]['lane'] = 2
    elif corruption == 'time': manifest['chart']['notes'][0]['time_s'] = float('nan')
    elif corruption == 'duration': manifest['chart']['duration_s'] = 4
    elif corruption == 'version': manifest['schema_version'] = 'unsupported'
    output = tmp_path / 'bad.cadencia'
    with zipfile.ZipFile(output, 'w') as target:
        target.writestr('chart.json', json.dumps(manifest))
        target.writestr('audio.wav', audio)
        if corruption == 'traversal': target.writestr('../outside', 'bad')
    with pytest.raises(ValueError): read_artifact(output)
    with pytest.raises(ValueError): Simulator(output, port=0, open_browser=False, block=False)
    assert not (tmp_path / 'outside').exists()


def test_existing_chart_is_preserved(artifact):
    original = artifact.read_bytes()
    with pytest.raises(FileExistsError):
        write_artifact(artifact, dict(notes=[], duration_s=3, threshold=.5, candidates=0), artifact)
    assert artifact.read_bytes() == original
