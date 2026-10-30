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
MAX_AUDIO_BYTES = 512 * 1024 * 1024


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_chart(value):
    """Validate playback data and return only the public chart fields."""
    if not isinstance(value, dict):
        raise ValueError("Chart must be a JSON object")
    duration = value.get("duration_s")
    if not _number(duration) or duration <= 0:
        raise ValueError("Invalid chart duration")
    notes = value.get("notes")
    if not isinstance(notes, list):
        raise ValueError("Chart notes must be a list")
    previous = -1
    clean_notes = []
    for note in notes:
        if not isinstance(note, dict):
            raise ValueError("Invalid note")
        time_s, lane, probability = note.get("time_s"), note.get("lane"), note.get("probability")
        if (not _number(time_s) or not 0 <= time_s < duration or time_s <= previous
                or type(lane) is not int or lane != 0
                or not _number(probability) or not 0 <= probability <= 1):
            raise ValueError("Notes must be ordered first-lane taps within the song with valid probabilities")
        previous = time_s
        clean_notes.append(dict(time_s=time_s, lane=lane, probability=probability))
    threshold, candidates = value.get("threshold"), value.get("candidates")
    if not _number(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid chart threshold")
    if type(candidates) is not int or candidates < len(notes):
        raise ValueError("Invalid candidate count")
    result = dict(notes=clean_notes, duration_s=duration, threshold=threshold, candidates=candidates)
    for key in ("title", "candidate_policy", "model_sha256"):
        if key in value:
            if not isinstance(value[key], str):
                raise ValueError(f"Invalid {key}")
            result[key] = value[key]
    return result


def write_artifact(out, chart, audio):
    """Publish a complete artifact atomically, refusing to replace any existing file."""
    out, audio = Path(out), Path(audio)
    if out.exists():
        raise FileExistsError(f"Chart already exists: {out}")
    if audio.stat().st_size > MAX_AUDIO_BYTES:
        raise ValueError("Chart audio exceeds supported size (512 MiB)")
    chart = validate_chart(chart)
    manifest = dict(schema_version=SCHEMA_VERSION, chart=chart, audio="audio.wav",
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
            if (not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION
                    or manifest.get("audio") != "audio.wav"):
                raise ValueError("Unsupported chart artifact format")
            chart = validate_chart(manifest.get("chart"))
            audio = archive.read("audio.wav")
        if hashlib.sha256(audio).hexdigest() != manifest.get("audio_sha256"):
            raise ValueError("Chart audio checksum mismatch")
        info = sf.info(io.BytesIO(audio))
        if info.format not in ("WAV", "WAVEX") or abs(info.duration - chart["duration_s"]) > .05:
            raise ValueError("Chart duration does not match its WAV audio")
        return chart, audio
    except (zipfile.BadZipFile, json.JSONDecodeError, sf.LibsndfileError) as error:
        raise ValueError(f"Invalid chart artifact: {error}") from error
