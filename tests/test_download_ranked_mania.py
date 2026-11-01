import io
import zipfile

import pytest
import soundfile as sf

from cadencia.osumania import HinamizawaClient, MissingChartAudioError
from scripts.download_ranked_mania import download_ranked_four_key
from conftest import LINK, ranked_set


def test_script_selects_four_key_charts_across_pages():
    seven_key = ranked_set(100)
    seven_key["id"] = 9
    seven_key["beatmaps"][0]["cs"] = 7
    mixed = ranked_set(4855704, 4855704, 1, 2, 3)
    mixed["beatmaps"][3]["cs"] = 7
    mixed["beatmaps"][4]["mode"] = "osu"

    class Client:
        def __init__(self):
            self.pages = []
            self.downloads = []

        def get_ranked_mania(self, params):
            assert params["query"] == "artist"
            self.pages.append(params["page"])
            return {"beatmapsets": [[seven_key], [mixed]][params["page"]]}

        def download_charts(self, params):
            self.downloads.append(params)
            return params["urls"]

    client = Client()
    assert download_ranked_four_key(client, count=2, query="artist") == [
        LINK, "https://osu.ppy.sh/beatmapsets/2043161#mania/1",
    ]
    assert client.pages == [0, 1]
    assert client.downloads == [
        {"urls": [LINK]}, {"urls": ["https://osu.ppy.sh/beatmapsets/2043161#mania/1"]},
    ]


def test_script_reports_insufficient_four_key_results():
    class Client:
        def get_ranked_mania(self, params):
            return {"beatmapsets": []}

        def download_charts(self, params):
            pytest.fail("No charts selected")

    with pytest.raises(RuntimeError, match="0 of 100"):
        download_ranked_four_key(Client())


def test_script_does_not_loop_on_repeated_non_four_key_pages():
    seven_key = ranked_set(1)
    seven_key["beatmaps"][0]["cs"] = 7

    class Client:
        def get_ranked_mania(self, params):
            return {"beatmapsets": [seven_key]}

    with pytest.raises(RuntimeError, match="without advancing"):
        download_ranked_four_key(Client())


def test_virtual_audio_is_skipped_and_successes_are_counted(tmp_path, package, monkeypatch, caplog):
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(package) as source, zipfile.ZipFile(archive_bytes, "w") as archive:
        for name in source.namelist():
            payload = source.read(name)
            if name == "hyper.osu":
                payload = payload.replace(b"AudioFilename: audio.wav", b"AudioFilename: virtual")
            archive.writestr(name, payload)
    client = HinamizawaClient(data_dir=tmp_path / "data")
    monkeypatch.setattr(client, "get_ranked_mania", lambda params: {"beatmapsets": [ranked_set(4855704, 1)]})

    def download(set_id, destination):
        destination.write_bytes(archive_bytes.getvalue())
        return {}

    monkeypatch.setattr(client, "_download", download)
    charts = download_ranked_four_key(client, count=1)
    assert [chart.id for chart in charts] == [1]
    assert charts[0].song.path.is_file()
    assert "Skipping" in caplog.text and LINK in caplog.text and "virtual" in caplog.text
    assert not (client.data_dir / "osumania/charts/4855704").exists()


def test_script_counts_only_successes_when_search_is_exhausted():
    class Client:
        def get_ranked_mania(self, params):
            return {"beatmapsets": [ranked_set(4855704)] if params["page"] == 0 else []}

        def download_charts(self, params):
            raise MissingChartAudioError("AudioFilename: virtual")

    with pytest.raises(RuntimeError, match="processing 0 of 1"):
        download_ranked_four_key(Client(), count=1)


@pytest.mark.parametrize("error", [
    RuntimeError("Download failed"), sf.LibsndfileError(29),
    ValueError("Malformed chart"), FileNotFoundError("Missing cached song"),
])
def test_script_logs_chart_errors_and_continues(error, caplog):
    class Client:
        def get_ranked_mania(self, params):
            return {"beatmapsets": [ranked_set(4855704, 1)]}

        def download_charts(self, params):
            if params["urls"] == [LINK]:
                raise error
            return params["urls"]

    assert download_ranked_four_key(Client(), count=1) == [
        "https://osu.ppy.sh/beatmapsets/2043161#mania/1",
    ]
    assert LINK in caplog.text
    assert type(error).__name__ in caplog.text


def test_script_allows_keyboard_interrupts():
    class Client:
        def get_ranked_mania(self, params):
            return {"beatmapsets": [ranked_set(4855704)]}

        def download_charts(self, params):
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        download_ranked_four_key(Client(), count=1)


def test_script_skips_malformed_chart_metadata(caplog):
    class Client:
        def get_ranked_mania(self, params):
            return {"beatmapsets": [ranked_set("invalid-id", 1)]}

        def download_charts(self, params):
            return params["urls"]

    assert download_ranked_four_key(Client(), count=1) == [
        "https://osu.ppy.sh/beatmapsets/2043161#mania/1",
    ]
    assert "invalid-id" in caplog.text and "ValueError" in caplog.text


def test_catalogue_failures_still_stop_the_batch():
    class Client:
        def get_ranked_mania(self, params):
            raise RuntimeError("Catalogue unavailable")

    with pytest.raises(RuntimeError, match="Catalogue unavailable"):
        download_ranked_four_key(Client(), count=1)
