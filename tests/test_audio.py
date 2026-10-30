import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import cadencia
from cadencia import audio
from cadencia.analysis import analyze_song


def test_native_audio_is_read_without_ffmpeg(tmp_path, monkeypatch):
    path = tmp_path / "native.wav"
    expected = np.zeros((100, 2), dtype="float32")
    sf.write(path, expected, 48000, subtype="FLOAT")
    monkeypatch.setattr(audio.shutil, "which", lambda name: pytest.fail("Native decoder should suffice"))
    samples, rate, decoder = audio.read_audio(path)
    np.testing.assert_array_equal(samples, expected)
    assert rate == 48000
    assert decoder == "soundfile"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is optional")
def test_ffmpeg_fallback_preserves_channels_rate_and_source(tmp_path, monkeypatch):
    path = tmp_path / "stereo.wav"
    expected = np.column_stack([np.arange(4800) / 4800, -np.arange(4800) / 4800]).astype("float32")
    sf.write(path, expected, 48000, subtype="FLOAT")
    original = path.read_bytes()
    read = sf.read

    def fail_native(source, **kwargs):
        if Path(source) == path:
            raise sf.LibsndfileError(29)
        return read(source, **kwargs)

    monkeypatch.setattr(audio.sf, "read", fail_native)
    samples, rate, decoder = audio.read_audio(path)
    np.testing.assert_array_equal(samples, expected)
    assert samples.dtype == np.float32
    assert rate == 48000
    assert decoder == "ffmpeg"
    assert path.read_bytes() == original


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is optional")
def test_analysis_and_waveform_use_same_fallback(downloaded_chart, monkeypatch):
    chart, _, _ = downloaded_chart
    read = sf.read
    expected, _ = read(chart.song.path, dtype="float32")

    def fail_native(source, **kwargs):
        if Path(source) == chart.song.path:
            raise sf.LibsndfileError(29)
        return read(source, **kwargs)

    monkeypatch.setattr(audio.sf, "read", fail_native)
    features = analyze_song(chart.song.path, chart.song.title, chart.song.artist)
    assert features["analysis_metadata"]["audio_decoder"] == "ffmpeg"
    assert features["song"] == chart.song
    np.testing.assert_array_equal(cadencia.get_waveform(chart), expected)


def test_missing_ffmpeg_reports_actionable_decoder_error(tmp_path, monkeypatch):
    path = tmp_path / "broken.mp3"
    path.write_bytes(b"invalid audio")
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="Install FFmpeg") as error:
        audio.read_audio(path)
    assert str(path) in str(error.value)
    assert isinstance(error.value.__cause__, sf.LibsndfileError)


def test_failed_fallback_reports_both_decoders_and_cleans_up(tmp_path, monkeypatch):
    path = tmp_path / "broken.mp3"
    path.write_bytes(b"invalid audio")
    monkeypatch.setattr(audio.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    outputs = []

    def fail_decode(command, **kwargs):
        outputs.append(Path(command[-1]))
        return subprocess.CompletedProcess(command, 1, "", "Invalid data found")

    monkeypatch.setattr(audio.subprocess, "run", fail_decode)
    with pytest.raises(RuntimeError, match="Neither SoundFile nor FFmpeg") as error:
        audio.read_audio(path)
    assert "Invalid data found" in str(error.value)
    assert str(path) in str(error.value)
    assert not outputs[0].parent.exists()
