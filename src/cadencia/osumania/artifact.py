"""Portable simulator charts: a versioned JSON manifest plus a WAV in one ZIP."""

import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile

SCHEMA_VERSION = "cadencia.simulator.v1"
PLACEMENT_SCHEMA = "cadencia.simulator.v2"
MAX_AUDIO_BYTES = 512 * 1024 * 1024


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_chart(value, *, placement=False):
    """Validate ordered lane actions; v1 remains restricted to first-lane taps."""
    if not isinstance(value, dict):
        raise ValueError("Chart must be a JSON object")
    duration = value.get("duration_s")
    if not _number(duration) or duration <= 0:
        raise ValueError("Invalid chart duration")
    notes = value.get("notes")
    if not isinstance(notes, list):
        raise ValueError("Chart notes must be a list")
    previous = -1
    lane_ends = [-1.] * 4
    clean_notes = []
    for note in notes:
        if not isinstance(note, dict):
            raise ValueError("Invalid note")
        time_s, lane, probability = note.get("time_s"), note.get("lane"), note.get("probability")
        kind = note.get("kind", "tap")
        if (not _number(time_s) or not 0 <= time_s < duration
                or time_s < previous or (not placement and time_s == previous)
                or type(lane) is not int or lane not in range(4) or (not placement and lane != 0)
                or not _number(probability) or not 0 <= probability <= 1
                or kind not in ("tap", "hold") or (not placement and kind != "tap")):
            raise ValueError("Invalid chart note order, lane, kind or probability")
        # A release and a new head may share a timestamp. Two heads may not.
        if time_s < lane_ends[lane]:
            raise ValueError("Overlapping notes on one lane")
        previous = time_s
        clean = dict(time_s=time_s, lane=lane, probability=probability)
        if placement:
            clean['kind'] = kind
        if kind == 'hold':
            end = note.get('end_s')
            if not _number(end) or not time_s < end <= duration:
                raise ValueError('Invalid hold endpoint')
            clean['end_s'] = end
            lane_ends[lane] = end
        else:
            lane_ends[lane] = time_s + 1e-9
        clean_notes.append(clean)
    threshold, candidates = value.get("threshold"), value.get("candidates")
    if not _number(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid chart threshold")
    if type(candidates) is not int or candidates < len(notes):
        raise ValueError("Invalid candidate count")
    result = dict(notes=clean_notes, duration_s=duration, threshold=threshold, candidates=candidates)
    for key in ("title", "artist", "author", "version", "source_url", "candidate_policy", "model_sha256", "decoding"):
        if key in value:
            if not isinstance(value[key], str):
                raise ValueError(f"Invalid {key}")
            result[key] = value[key]
    for key in ('difficulty', 'bpm', 'beat_offset_s', 'temperature'):
        if key in value:
            if not _number(value[key]) or (key != 'beat_offset_s' and value[key] <= 0):
                raise ValueError(f'Invalid {key}')
            result[key] = value[key]
    if 'timing_map' in value:
        timing=value['timing_map']
        if not isinstance(timing,list) or not timing or len(timing)>100000:
            raise ValueError('Invalid tracked timing map')
        previous_time=-1.;previous_beat=float('-inf')
        for point in timing:
            if not isinstance(point,dict) or any(not _number(point.get(k)) for k in ['start_s','beat_position','bpm']):
                raise ValueError('Invalid tracked timing point')
            if point['start_s']<=previous_time or point['beat_position']<=previous_beat or point['bpm']<=0:
                raise ValueError('Nonmonotonic tracked timing map')
            previous_time,previous_beat=point['start_s'],point['beat_position']
        result['timing_map']=[dict(point) for point in timing]
    if 'rhythm' in value:
        rhythm=value['rhythm']
        if not isinstance(rhythm,dict) or not isinstance(rhythm.get('spec'),dict) or type(rhythm.get('bar_boundaries_detected')) is not bool:
            raise ValueError('Invalid rhythm provenance')
        downbeats=rhythm.get('downbeat_times')
        if not isinstance(downbeats,list) or any(not _number(t) or not 0<=t<duration for t in downbeats) or any(b<=a for a,b in zip(downbeats,downbeats[1:])):
            raise ValueError('Invalid detected downbeats')
        result['rhythm']=dict(rhythm)
    if 'alignment' in value:
        report=value['alignment']
        if not isinstance(report,dict) or report.get('schema')!='cadencia.beat_alignment.v1':
            raise ValueError('Invalid alignment report')
        if not isinstance(report.get('prior_sha256'),str) or len(report['prior_sha256'])!=64:
            raise ValueError('Invalid alignment prior hash')
        for key in ['onset_groups','moved_groups']:
            if type(report.get(key)) is not int or report[key]<0:raise ValueError('Invalid alignment count')
        for key in ['max_shift_ms','mean_absolute_shift_ms']:
            if not _number(report.get(key)) or report[key]<0:raise ValueError('Invalid alignment shift')
        result['alignment']=dict(report)
    if 'repetition' in value:
        report = value['repetition']
        if not isinstance(report, dict) or report.get('schema') != 'cadencia.repetition.v1':
            raise ValueError('Invalid repetition report')
        for key in ('notes_before', 'notes_after', 'detected_matches'):
            if type(report.get(key)) is not int or report[key] < 0:
                raise ValueError('Invalid repetition count')
        if report['notes_after'] != len(clean_notes) or not isinstance(report.get('matches'), list):
            raise ValueError('Invalid repetition matches or note count')
        for key in ('variation', 'similarity_threshold'):
            if not _number(report.get(key)) or not 0 <= report[key] <= 1:
                raise ValueError('Invalid repetition setting')
        result['repetition'] = dict(report)
    if 'seed' in value:
        if type(value['seed']) is not int or not 0 <= value['seed'] < 2**63:
            raise ValueError('Invalid sampling seed')
        result['seed'] = value['seed']
    return result


def write_artifact(out, chart, audio):
    """Publish a complete artifact atomically, refusing to replace any existing file."""
    out, audio = Path(out), Path(audio)
    if out.exists():
        raise FileExistsError(f"Chart already exists: {out}")
    if audio.stat().st_size > MAX_AUDIO_BYTES:
        raise ValueError("Chart audio exceeds supported size (512 MiB)")
    placement = any('kind' in note for note in chart.get('notes', [])) or 'difficulty' in chart
    chart = validate_chart(chart, placement=placement)
    manifest = dict(schema_version=PLACEMENT_SCHEMA if placement else SCHEMA_VERSION, chart=chart, audio="audio.wav",
                    audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest())
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{out.name}-", dir=out.parent) as temporary:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("chart.json", json.dumps(manifest, allow_nan=False))
            archive.write(audio, "audio.wav")
        # A hard link publishes the completed file without an overwrite race.
        os.link(temporary.name, out)


def read_artifact(path):
    """Read and verify a chart without extracting archive-supplied filesystem paths."""
    import soundfile as sf

    try:
        with zipfile.ZipFile(path) as archive:
            if sorted(archive.namelist()) != ["audio.wav", "chart.json"]:
                raise ValueError("Chart artifact must contain exactly chart.json and audio.wav")
            if (archive.getinfo("chart.json").file_size > 8 * 1024 * 1024
                    or archive.getinfo("audio.wav").file_size > MAX_AUDIO_BYTES):
                raise ValueError("Chart artifact exceeds supported size")
            manifest = json.loads(archive.read("chart.json"))
            if (not isinstance(manifest, dict) or manifest.get("schema_version") not in (SCHEMA_VERSION, PLACEMENT_SCHEMA)
                    or manifest.get("audio") != "audio.wav"):
                raise ValueError("Unsupported chart artifact format")
            chart = validate_chart(manifest.get("chart"), placement=manifest["schema_version"] == PLACEMENT_SCHEMA)
            audio = archive.read("audio.wav")
        if hashlib.sha256(audio).hexdigest() != manifest.get("audio_sha256"):
            raise ValueError("Chart audio checksum mismatch")
        info = sf.info(io.BytesIO(audio))
        if info.format not in ("WAV", "WAVEX") or abs(info.duration - chart["duration_s"]) > .05:
            raise ValueError("Chart duration does not match its WAV audio")
        return chart, audio
    except (zipfile.BadZipFile, json.JSONDecodeError, sf.LibsndfileError) as error:
        raise ValueError(f"Invalid chart artifact: {error}") from error
