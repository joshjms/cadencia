import json
import urllib.request
import zipfile

import numpy as np
import pytest

import cadencia
from cadencia.osumania import loader
from conftest import LINK, beatmap, ranked_set


def test_download_analyze_serialize_and_reload(downloaded_chart, monkeypatch):
    chart, root, calls = downloaded_chart
    assert isinstance(chart, cadencia.Chart)
    assert chart.id == 4855704
    assert len(chart.events) == 4
    assert isinstance(chart.lanes[2][0], cadencia.osumania.Hold)
    assert chart.lanes[2][0].end_s == 1.5
    assert chart.song.path.is_file()
    assert chart.song.duration_s == 4
    folder = root / "osumania" / "charts" / "4855704"
    assert set(path.name for path in folder.iterdir()) == {"archive.osz", "chart.osu", "song.wav", "download.json", "trace.json"}
    trace = json.loads((folder / "trace.json").read_text())
    assert trace["song"]["path"] == "song.wav"
    assert trace["lanes"]["2"][0]["kind"] == "hold"
    assert set(trace["lanes"]) == {"0", "1", "2", "3"}
    assert len(chart.waveform.times) == len(chart.waveform.minimum) == len(chart.waveform.maximum) == 200
    for name in ["onset_strength", "rms_energy", "spectral_centroid", "harmonic_energy_fraction", "chroma"]:
        series = getattr(chart, name)
        assert len(series.times) == len(series.values)
        assert np.isfinite(np.array(series.values, dtype=float)).all()
    assert np.asarray(chart.chroma.values).shape == (201, 12)
    assert chart.chroma.labels[9] == "A"
    assert np.argmax(chart.chroma.values[35]) == 9
    assert chart.rms_energy.values[35] == pytest.approx(0.1 / np.sqrt(2), rel=0.02)
    assert chart.spectral_centroid.values[35] == pytest.approx(440, rel=0.03)
    assert chart.harmonic_energy_fraction.values[35] > 0.95
    for attack in [0.5, 1, 2, 3]:
        assert any(abs(time - attack) < 0.08 for time in chart.candidate_onsets.times)
    assert all(0 <= time < 4 for time in chart.candidate_onsets.times)
    assert all(0 <= value <= 1 for value in chart.harmonic_energy_fraction.values)
    assert all(abs(time - index * 0.02) < 1e-10 for index, time in enumerate(chart.onset_strength.times))
    assert chart.metadata["analysis"]["hop_length"] == 441
    assert chart.metadata["provider"] == "local test fixture"

    def unexpected(*args, **kwargs):
        pytest.fail("Cache access attempted a download or audio analysis")

    monkeypatch.setattr(urllib.request, "urlopen", unexpected)
    monkeypatch.setattr(loader, "analyze_song", unexpected)
    for params in [{"url": LINK}, {"name": "test recording"}, {"name": "Test artist - Test recording [Hyper]"}]:
        cached = cadencia.osumania.load_chart(**params, data_dir=root)
        assert cached.to_dict() == chart.to_dict()
    assert calls == [2043161]


def test_missing_local_chart_does_not_create_directories_or_call_network(tmp_path, monkeypatch):
    def unexpected(*args):
        pytest.fail("Network access attempted")

    monkeypatch.setattr(urllib.request, "urlopen", unexpected)
    root = tmp_path / "missing"
    for params in [{"url": LINK}, {"name": "missing"}]:
        with pytest.raises(FileNotFoundError, match="local"):
            cadencia.osumania.load_chart(**params, data_dir=root)
    assert not root.exists()


def test_only_client_rebuilds_existing_archive_offline(downloaded_chart, monkeypatch):
    chart, root, calls = downloaded_chart
    chart.trace_path.unlink()

    def unexpected(*args):
        pytest.fail("Existing archive should be reused")

    monkeypatch.setattr(urllib.request, "urlopen", unexpected)
    with pytest.raises(FileNotFoundError, match="preprocessed"):
        cadencia.osumania.load_chart(url=LINK, data_dir=root)
    assert not chart.trace_path.exists()
    rebuilt = cadencia.osumania.HinamizawaClient(data_dir=root).download_charts({"count": 1})[0]
    assert rebuilt.to_dict() == chart.to_dict()
    assert calls == [2043161]


def test_failed_import_leaves_no_partial_cache(tmp_path, package, monkeypatch):
    with zipfile.ZipFile(package, "a") as archive:
        archive.writestr("duplicate.osu", beatmap("64,192,500,1,0"))

    def download(self, set_id, destination):
        destination.write_bytes(package.read_bytes())
        return {}

    client = cadencia.osumania.HinamizawaClient
    monkeypatch.setattr(client, "_download", download)
    monkeypatch.setattr(client, "get_ranked_mania", lambda self, params: {"beatmapsets": [ranked_set(4855704)]})
    root = tmp_path / "data"
    with pytest.raises(ValueError, match="exactly one"):
        client(data_dir=root).download_charts({"count": 1})
    assert not list((root / "osumania" / "charts").iterdir())


@pytest.mark.parametrize("filename", ["virtual", "Virtual", "missing.wav"])
def test_missing_song_has_specific_error_and_no_partial_cache(tmp_path, package, monkeypatch, filename):
    archive_path = tmp_path / "missing-song.osz"
    with zipfile.ZipFile(package) as source, zipfile.ZipFile(archive_path, "w") as archive:
        for name in source.namelist():
            payload = source.read(name)
            if name == "hyper.osu":
                payload = payload.replace(b"AudioFilename: audio.wav", f"AudioFilename: {filename}".encode())
            archive.writestr(name, payload)

    client = cadencia.osumania.HinamizawaClient(data_dir=tmp_path / "data")

    def download(set_id, destination):
        destination.write_bytes(archive_path.read_bytes())
        return {}

    monkeypatch.setattr(client, "_download", download)
    monkeypatch.setattr(loader, "analyze_song", lambda *args: pytest.fail("Missing recording reached analysis"))
    with pytest.raises(cadencia.osumania.MissingChartAudioError) as error:
        client.download_charts({"urls": [LINK]})
    assert isinstance(error.value, FileNotFoundError)
    assert filename in str(error.value)
    assert "4855704" in str(error.value) and "2043161" in str(error.value)
    assert LINK in str(error.value)
    assert list((client.data_dir / "osumania/charts").iterdir()) == []


@pytest.mark.parametrize("name", ["../outside.wav", "/absolute.wav", "C:/outside.wav", "..\\outside.wav"])
def test_unsafe_archives_are_rejected(tmp_path, name):
    path = tmp_path / "unsafe.osz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(name, b"test")
    with zipfile.ZipFile(path) as archive, pytest.raises(ValueError, match="Unsafe"):
        loader._archive_files(archive)


def test_cache_requires_matching_identity_and_present_audio(downloaded_chart):
    chart, root, _ = downloaded_chart
    trace = chart.to_dict()
    trace["metadata"]["beatmapset_id"] = 99
    chart.trace_path.write_text(json.dumps(trace))
    with pytest.raises(ValueError, match="identity"):
        cadencia.osumania.load_chart(url=LINK, data_dir=root)
    chart.save()
    chart.song.path.unlink()
    with pytest.raises(FileNotFoundError, match="song"):
        cadencia.osumania.load_chart(url=LINK, data_dir=root)


@pytest.mark.parametrize("params", [{}, {"name": "x", "url": LINK}, {"name": " "}])
def test_local_search_requires_one_nonempty_parameter(tmp_path, params):
    with pytest.raises(ValueError):
        cadencia.osumania.load_chart(**params, data_dir=tmp_path)


def test_ambiguous_local_name_requires_disambiguation(downloaded_chart):
    chart, root, _ = downloaded_chart
    trace = chart.to_dict()
    trace["id"] = 1
    trace["song"]["path"] = chart.song.path.name
    trace["metadata"]["difficulty"] = "Another difficulty"
    folder = chart.trace_path.parent.parent / "1"
    folder.mkdir()
    (folder / "trace.json").write_text(json.dumps(trace))
    with pytest.raises(ValueError, match="Multiple"):
        cadencia.osumania.load_chart(name="Test recording", data_dir=root)
    assert cadencia.osumania.load_chart(name="Test artist - Test recording [Hyper]", data_dir=root).id == chart.id


def test_link_formats_and_data_directory(tmp_path, monkeypatch):
    assert loader.chart_ids(LINK) == (2043161, 4855704)
    assert loader.chart_ids(f"[{LINK}]({LINK})") == (2043161, 4855704)
    for invalid in ["https://example.com/beatmapsets/2043161#mania/4855704", "https://osu.ppy.sh/beatmaps/4855704", "https://osu.ppy.sh/beatmapsets/2043161#osu/4855704"]:
        with pytest.raises(ValueError):
            loader.chart_ids(invalid)
    monkeypatch.setenv("CADENCIA_DATA_DIR", str(tmp_path))
    assert loader.data_root() == tmp_path
