import io
import shutil
import zipfile

import numpy as np
import pytest
import soundfile as sf

from cadencia.osumania import HinamizawaClient

LINK = "https://osu.ppy.sh/beatmapsets/2043161#mania/4855704"


def ranked_set(*chart_ids):
    return {"id": 2043161, "status": "ranked", "beatmaps": [
        {"id": chart_id, "mode": "mania", "cs": 4, "ranked": 1}
        for chart_id in chart_ids
    ]}


def beatmap(objects, *, keys=4, mode=3, timing="0,500,4,2,1,100,1,0", chart_id=4855704):
    return f"""osu file format v14

[General]
AudioFilename: audio.wav
AudioLeadIn: 5000
Mode: {mode}

[Metadata]
Title: Test recording
Artist: Test artist
Creator: Test mapper
Version: Hyper
BeatmapID: {chart_id}
BeatmapSetID: 2043161

[Difficulty]
CircleSize: {keys}

[TimingPoints]
{timing}

[HitObjects]
{objects}
""".encode()


@pytest.fixture
def package(tmp_path):
    audio = io.BytesIO()
    rate = 22050
    times = np.arange(rate * 4) / rate
    samples = 0.1 * np.sin(2 * np.pi * 440 * times)
    for start in [rate // 2, rate, rate * 2, rate * 3]:
        samples[start:start + 100] += np.hanning(100) * 0.5
    sf.write(audio, samples, rate, format="WAV")
    path = tmp_path / "example.osz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("audio.wav", audio.getvalue())
        archive.writestr("other.osu", beatmap("64,192,100,1,0", chart_id=1))
        archive.writestr("hyper.osu", beatmap("64,192,500,1,0\n192,192,500,1,0\n320,192,1000,128,0,1500:0:0:0:0\n448,192,2000,1,0"))
    return path


@pytest.fixture
def downloaded_chart(tmp_path, package, monkeypatch):
    calls = []

    def download(self, set_id, destination):
        calls.append(set_id)
        shutil.copyfile(package, destination)
        return {"provider": "local test fixture"}

    monkeypatch.setattr(HinamizawaClient, "_download", download)
    monkeypatch.setattr(HinamizawaClient, "get_ranked_mania", lambda self, params: {"beatmapsets": [ranked_set(4855704)]})
    root = tmp_path / "data"
    return HinamizawaClient(data_dir=root).download_charts({"count": 1})[0], root, calls
