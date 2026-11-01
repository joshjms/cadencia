import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import soundfile as sf

from scripts import build_chart
from cadencia.osumania.artifact import read_artifact, write_artifact
from scripts.convert_cadencia import osu_text


def spec_file(tmp_path):
    value = dict(audio_file='song.wav', title='Song', artist='Artist', author='Author',
                 version='1.5 target', difficulty=1.5, checkpoint='model.pt',
                 filename='chart', output_dir='output')
    path = tmp_path/'chart.json'
    path.write_text(json.dumps(value))
    return path, value


def chart():
    return dict(duration_s=1., notes=[dict(time_s=.2, lane=0, probability=.8, kind='tap'),
                                    dict(time_s=.5, lane=1, probability=.9, kind='hold', end_s=.8)],
                threshold=0., candidates=400, difficulty=1.5, bpm=90.,
                timing_map=[dict(start_s=0., beat_position=0., bpm=90.),
                            dict(start_s=.5, beat_position=.75, bpm=750.)])


def test_json_pipeline_preserves_notes_audio_and_metadata(tmp_path, monkeypatch):
    if not build_chart.shutil.which('ffmpeg'):
        pytest.skip('ffmpeg unavailable')
    path, value = spec_file(tmp_path)
    sf.write(tmp_path/'song.wav', np.zeros(24000), 24000)
    (tmp_path/'model.pt').write_bytes(b'fixture')
    real_run = build_chart.run
    commands = []
    def run(*args):
        commands.append([str(arg) for arg in args])
        if str(args[1]).endswith('scripts/run_simulator.py'):
            destination = Path(args[args.index('--chart-out')+1])
            write_artifact(destination, chart(), tmp_path/'song.wav')
        else:
            real_run(*args)
    monkeypatch.setattr(build_chart, 'run', run)
    paths = build_chart.build(path)
    assert all(p.is_file() for p in paths.values())
    result, audio = read_artifact(paths['cadencia'])
    assert result['notes'] == chart()['notes']
    assert {k:result[k] for k in ('title','artist','author','version')} == {k:value[k] for k in ('title','artist','author','version')}
    with ZipFile(paths['osz']) as archive:
        assert archive.read('audio.wav') == audio
        text = archive.read('chart.osu').decode()
        assert 'Creator:Author' in text and 'Version:1.5 target' in text
        assert text.split('[HitObjects]\n')[1] == osu_text(result).split('[HitObjects]\n')[1]
        points = text.split('[TimingPoints]\n')[1].split('[HitObjects]')[0].strip().splitlines()
        assert len(points) == 1 and float(points[0].split(',')[1]) == pytest.approx(60000/90)
    assert all('--target-stars' not in command for command in commands)
    with pytest.raises(FileExistsError): build_chart.build(path)


def test_export_stage_needs_no_audio_download_or_checkpoint(tmp_path):
    path, _ = spec_file(tmp_path)
    output = tmp_path/'output'; output.mkdir()
    sf.write(tmp_path/'song.wav', np.zeros(24000), 24000)
    write_artifact(output/'chart.cadencia', chart(), tmp_path/'song.wav')
    paths = build_chart.build(path, stage='export')
    assert paths['osz'].exists() and not paths['mp3'].exists()


@pytest.mark.parametrize('change', [dict(difficulty=-1), dict(difficulty=True), dict(filename='../oops'),
                                    dict(repeat_variation=2), dict(constant_scroll='true'),
                                    dict(model_version='v1'), dict(unknown=True),
                                    dict(author='bad\nmetadata')])
def test_invalid_specs_fail_before_execution(tmp_path, change):
    path, value = spec_file(tmp_path)
    value.update(change); path.write_text(json.dumps(value))
    with pytest.raises(ValueError): build_chart.load_spec(path)


def test_youtube_download_is_single_video_and_publishes_mp3(tmp_path, monkeypatch):
    path, value = spec_file(tmp_path)
    value.pop('audio_file'); value['youtube_url']='https://www.youtube.com/watch?v=abc&list=playlist'
    path.write_text(json.dumps(value))
    commands=[]
    monkeypatch.setattr(build_chart.shutil, 'which', lambda _: '/usr/bin/ffmpeg')
    monkeypatch.setattr(build_chart.importlib.util, 'find_spec', lambda _: object())
    def run(*args):
        commands.append(args)
        template = str(args[args.index('-o')+1])
        Path(template.replace('%(ext)s','mp3')).write_bytes(b'fake mp3')
    monkeypatch.setattr(build_chart, 'run', run)
    paths = build_chart.build(path, stage='audio')
    assert paths['mp3'].read_bytes() == b'fake mp3'
    assert '--no-playlist' in commands[0] and '--ignore-config' in commands[0]
    assert not paths['cadencia'].exists()


def test_constant_scroll_only_changes_timing_points():
    value = chart()
    before = osu_text(value)
    after = osu_text(value, constant_scroll=True)
    assert before.split('[HitObjects]\n')[1] == after.split('[HitObjects]\n')[1]
    assert '750' not in after.split('[TimingPoints]\n')[1].split('[HitObjects]')[0]


def test_missing_checkpoint_is_reported_before_download(tmp_path, monkeypatch):
    path, _ = spec_file(tmp_path)
    monkeypatch.setattr(build_chart, 'make_audio', lambda *args: pytest.fail('Downloaded before checkpoint validation'))
    with pytest.raises(FileNotFoundError, match='Checkpoint not found:.*Paths are relative to the JSON'):
        build_chart.build(path)
    assert not (tmp_path/'output').exists()
