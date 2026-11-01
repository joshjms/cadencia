"""Build MP3, .cadencia and .osz artifacts from one JSON chart specification."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlparse

SCRIPTS = Path(__file__).resolve().parent
FIELDS = {'youtube_url', 'audio_file', 'title', 'artist', 'author', 'version',
          'difficulty', 'checkpoint', 'model_version', 'output_dir', 'filename',
          'device', 'seed', 'temperature', 'decoding', 'bpm', 'beat_alignment',
          'alignment_max_shift_ms', 'repeat_sections', 'repeat_variation', 'constant_scroll'}


def load_spec(path):
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    if not isinstance(spec, dict):
        raise ValueError('Expected a JSON object')
    if set(spec)-FIELDS:
        raise ValueError(f'Unknown chart fields: {sorted(set(spec)-FIELDS)}')
    for key in ('title', 'artist', 'author', 'version'):
        if not isinstance(spec.get(key), str) or not spec[key].strip() or any(c in spec[key] for c in '\r\n\x00'):
            raise ValueError(f'{key} must be a nonempty single-line string')
    if ('youtube_url' in spec) == ('audio_file' in spec):
        raise ValueError('Provide exactly one of youtube_url or audio_file')
    if 'youtube_url' in spec:
        if not isinstance(spec['youtube_url'], str):
            raise ValueError('youtube_url must be a string')
        url = urlparse(spec['youtube_url'])
        if url.scheme != 'https' or url.hostname not in ('youtube.com', 'www.youtube.com', 'm.youtube.com', 'youtu.be'):
            raise ValueError('youtube_url must be an HTTPS YouTube link')
    if 'checkpoint' in spec and 'model_version' in spec:
        raise ValueError('Use checkpoint or model_version, not both')
    if 'model_version' in spec and (not isinstance(spec['model_version'], str) or not spec['model_version']):
        raise ValueError('model_version must be a nonempty string')
    for key, default, lower, upper in [('difficulty', None, 0, None), ('temperature', 1., 0, None),
                                      ('alignment_max_shift_ms', 25., 0, 50), ('repeat_variation', .2, -1e-9, 1)]:
        value = spec.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= lower or (upper is not None and value > upper):
            raise ValueError(f'Invalid {key}')
    if 'bpm' in spec and (isinstance(spec['bpm'], bool) or not isinstance(spec['bpm'], (int,float)) or not math.isfinite(spec['bpm']) or spec['bpm'] <= 0):
        raise ValueError('Invalid bpm')
    if type(spec.get('seed', 42)) is not int or not 0 <= spec.get('seed', 42) < 2**63:
        raise ValueError('Invalid seed')
    if spec.get('device', 'cpu') not in ('cpu', 'cuda') or spec.get('decoding', 'sample') not in ('sample', 'argmax'):
        raise ValueError('Invalid device or decoding')
    for key in ('beat_alignment', 'repeat_sections', 'constant_scroll'):
        if key in spec and type(spec[key]) is not bool:
            raise ValueError(f'{key} must be a boolean')
    for key in ('checkpoint', 'audio_file', 'output_dir'):
        if key in spec:
            if not isinstance(spec[key], str) or not spec[key]:
                raise ValueError(f'Invalid {key}')
            spec[key] = str((path.parent / spec[key]).resolve())
    spec.setdefault('output_dir', str(path.parent / 'output'))
    name = spec.setdefault('filename', f"{spec['artist']} - {spec['title']} ({spec['difficulty']:g} target)")
    if not isinstance(name, str) or not name.strip() or name in ('.','..') or any(c in name for c in '/\\\x00\r\n'):
        raise ValueError('filename must be a plain filename stem, without directory separators')
    return spec


def run(*command):
    subprocess.run([str(arg) for arg in command], check=True)


def make_audio(spec, destination):
    if destination.exists():
        raise FileExistsError(destination)
    if not shutil.which('ffmpeg'):
        raise RuntimeError('ffmpeg must be installed to produce MP3 audio')
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.audio-') as directory:
        temporary = Path(directory)
        if 'youtube_url' in spec:
            if importlib.util.find_spec('yt_dlp') is None:
                raise RuntimeError('Install downloads with: uv sync --extra placement --extra download')
            run(sys.executable, '-m', 'yt_dlp', '--ignore-config', '--no-playlist', '-f', 'bestaudio',
                '--extract-audio', '--audio-format', 'mp3', '--audio-quality', '0',
                '-o', temporary/'audio.%(ext)s', '--', spec['youtube_url'])
        else:
            source = Path(spec['audio_file'])
            if not source.is_file():
                raise FileNotFoundError(source)
            run('ffmpeg', '-nostdin', '-v', 'error', '-i', source, '-vn', '-codec:a', 'libmp3lame', '-q:a', '0', temporary/'audio.mp3')
        mp3 = temporary/'audio.mp3'
        if not mp3.is_file() or not mp3.stat().st_size:
            raise RuntimeError('Audio conversion did not produce an MP3')
        os.link(mp3, destination)


def resolve_checkpoint(spec):
    from cadencia.model_versions import resolve_model_version, DEFAULT_MODEL_VERSION
    checkpoint = Path(spec['checkpoint']) if 'checkpoint' in spec else resolve_model_version(spec.get('model_version', DEFAULT_MODEL_VERSION))
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}. Paths are relative to the JSON file; check its checkpoint field.')
    return checkpoint


def make_chart(spec, audio, destination):
    from cadencia.osumania.artifact import read_artifact, write_artifact
    checkpoint = resolve_checkpoint(spec)
    if not audio.is_file():
        raise FileNotFoundError(f'Run the audio stage first: {audio}')
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.chart-') as directory:
        temporary = Path(directory)
        raw = temporary/'generated.cadencia'
        command = [sys.executable, SCRIPTS/'run_simulator.py', '--song', audio, '--chart-out', raw,
                   '--checkpoint', checkpoint, '--generate-only', '--difficulty', spec['difficulty'],
                   '--device', spec.get('device', 'cpu'), '--seed', spec.get('seed', 42),
                   '--temperature', spec.get('temperature', 1.), '--decoding', spec.get('decoding', 'sample'),
                   '--alignment-max-shift-ms', spec.get('alignment_max_shift_ms', 25.)]
        if 'beat_alignment' in spec:
            command.append('--beat-alignment' if spec['beat_alignment'] else '--no-beat-alignment')
        if spec.get('repeat_sections', False):
            command += ['--repeat-sections', '--repeat-variation', spec.get('repeat_variation', .2)]
        if 'bpm' in spec:
            command += ['--bpm', spec['bpm']]
        run(*command)
        chart, wav = read_artifact(raw)
        chart.update({key: spec[key] for key in ('title', 'artist', 'author', 'version')})
        if 'youtube_url' in spec:
            chart['source_url'] = spec['youtube_url']
        playback = temporary/'audio.wav'
        playback.write_bytes(wav)
        write_artifact(destination, chart, playback)


def build(spec_path, *, stage='all'):
    spec = load_spec(spec_path)
    if stage not in ('all', 'audio', 'chart', 'export'):
        raise ValueError('Unknown stage')
    directory = Path(spec['output_dir'])
    paths = {extension: directory / (spec['filename']+'.'+extension) for extension in ('mp3','cadencia','osz')}
    selected = ('audio','chart','export') if stage == 'all' else (stage,)
    if 'chart' in selected:
        # Catch invalid model paths before spending time downloading audio.
        resolve_checkpoint(spec)
    for step in selected:
        target = paths[dict(audio='mp3', chart='cadencia', export='osz')[step]]
        if target.exists():
            raise FileExistsError(f'{target} already exists; use a separate stage or a different filename')
    directory.mkdir(parents=True, exist_ok=True)
    for step in selected:
        print(f'Stage: {step}', flush=True)
        if step == 'audio':
            make_audio(spec, paths['mp3'])
        elif step == 'chart':
            make_chart(spec, paths['mp3'], paths['cadencia'])
        else:
            # No --target-stars: difficulty is conditioning, never note deletion.
            run(sys.executable, SCRIPTS/'convert_cadencia.py', paths['cadencia'], '--out', paths['osz'],
                '--title', spec['title'], '--artist', spec['artist'], '--creator', spec['author'],
                '--version', spec['version'], '--constant-scroll' if spec.get('constant_scroll', True) else '--no-constant-scroll')
        print(f'Saved {paths[dict(audio="mp3", chart="cadencia", export="osz")[step]]}', flush=True)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec', type=Path, help='JSON chart specification; paths are relative to this file')
    parser.add_argument('--stage', choices=['all','audio','chart','export'], default='all')
    args = parser.parse_args()
    try:
        build(args.spec, stage=args.stage)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
