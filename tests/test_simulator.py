import json
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from cadencia.osumania import Simulator, SimulatorConfig
from cadencia.osumania import simulator


@pytest.mark.parametrize('kwargs', [dict(keybinds=('KeyD',)*4), dict(keybinds=('KeyA',)),
                                   dict(tile_speed=0), dict(tile_speed=float('nan'))])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        SimulatorConfig(**kwargs)


def test_audio_only_candidates(downloaded_chart):
    chart, _, _ = downloaded_chart
    analysis = dict(song=chart.song, estimated_beats=[1, 2, 3, 4], candidate_onsets=chart.candidate_onsets)
    times = simulator.candidate_times(analysis)
    assert 2.5 in times
    assert (times >= 2).all() and (times < chart.song.duration_s).all()
    assert np.all(np.diff(times) > 0)


def test_server_upload_lifecycle_and_errors(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'model.pt'
    checkpoint.touch()
    def generate(path, checkpoint_path):
        assert path.read_bytes() == b'test audio'
        path.with_name('playback.wav').write_bytes(b'RIFF')
        return dict(notes=[dict(time_s=2.5, lane=0)], duration_s=4, candidates=2)
    monkeypatch.setattr(simulator, 'generate_chart', generate)
    app = Simulator(port=0, checkpoint=checkpoint, block=False, open_browser=False)
    def get(route):
        with urllib.request.urlopen(app.url + route) as response:
            return response.read()
    try:
        assert b'Feel the prediction' in get('/')
        config = json.loads(get('/api/config'))
        assert config['keybinds'] == ['KeyD','KeyF','KeyJ','KeyK']
        request = urllib.request.Request(app.url+'/api/song', data=b'test audio', method='POST')
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 403
        request.add_header('X-Cadencia-Token', config['token'])
        with urllib.request.urlopen(request) as response:
            assert response.status == 202
        for _ in range(100):
            state = json.loads(get('/api/status'))
            if state['status'] != 'processing': break
            time.sleep(.01)
        assert state['status'] == 'ready'
        assert state['notes'][0]['lane'] == 0
        assert get('/audio.wav') == b'RIFF'
        def fail(*_): raise ValueError('Bad audio')
        monkeypatch.setattr(simulator, 'generate_chart', fail)
        with urllib.request.urlopen(request): pass
        for _ in range(100):
            state = json.loads(get('/api/status'))
            if state['status'] != 'processing': break
            time.sleep(.01)
        assert state['status'] == 'error' and 'Bad audio' in state['error']
        assert b'Cadencia' in get('/')
    finally:
        app.close()
    assert not app._root.exists()


def test_real_checkpoint_inference(downloaded_chart, monkeypatch):
    torch = pytest.importorskip('torch')
    from cadencia.dataset import FEATURES
    from cadencia.model import HitTransformer
    chart, root, _ = downloaded_chart
    model = HitTransformer()
    for parameter in model.parameters():
        parameter.data.zero_()
    model.output.bias.data.fill_(10)
    checkpoint = root / 'test.pt'
    torch.save(dict(model=model.state_dict(), mean=torch.zeros(20), scale=torch.ones(20),
                    features=FEATURES, threshold=.5), checkpoint)
    import cadencia.model
    histories = []
    class RecordingTransformer(HitTransformer):
        def forward(self, x):
            histories.append(x[0, :, -1].clone())
            return super().forward(x)
    monkeypatch.setattr(cadencia.model, 'HitTransformer', RecordingTransformer)
    original_audio = chart.song.path.read_bytes()
    result = simulator.generate_chart(chart.song.path, checkpoint)
    assert torch.count_nonzero(histories[0]) == 0
    assert any(torch.count_nonzero(history) > 0 for history in histories[1:])
    assert chart.song.path.read_bytes() == original_audio
    assert result['notes']
    assert all(note['lane'] == 0 and 2 <= note['time_s'] < chart.song.duration_s for note in result['notes'])
    assert np.all(np.diff([note['time_s'] for note in result['notes']]) >= .08)
    assert chart.song.path.with_name('playback.wav').is_file()
