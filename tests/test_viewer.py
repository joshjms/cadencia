import json
import urllib.error
import urllib.request

import pytest

import cadencia
from cadencia.viewer import LAYERS, manifest, selected_layers


def test_layer_selection():
    assert selected_layers(None) == LAYERS
    assert selected_layers(["lane0, lane1, waveform"]) == ["lane0", "lane1", "waveform"]
    assert selected_layers(["waveform", "lane1", "lane1"]) == ["lane1", "waveform"]
    assert selected_layers("chroma") == ["chroma"]
    assert selected_layers([]) == []
    with pytest.raises(ValueError, match="Unknown"):
        selected_layers(["lane4"])


def test_http_viewer_assets_config_manifest_and_audio_ranges(downloaded_chart):
    chart, root, _ = downloaded_chart
    server = cadencia.osumania.visualize(port="0", include=["lane0, lane1, waveform"], data_dir=root, block=False)
    base = f"http://127.0.0.1:{server.server_port}"

    def get(path, headers=None):
        return urllib.request.urlopen(urllib.request.Request(base + path, headers=headers or {}), timeout=5)

    try:
        with get("/api/config") as response:
            assert json.load(response)["include"] == ["lane0", "lane1", "waveform"]
        with get("/api/manifest") as response:
            value = json.load(response)
        assert value["schema_version"] == "cadencia.library.v1"
        assert value["songs"][0]["charts"][0]["chart_id"] == "4855704"
        with get("/") as response:
            assert b"Cadencia" in response.read()
        for asset in ["/app.js", "/style.css"]:
            with get(asset) as response:
                assert response.status == 200
                assert len(response.read()) > 100
        audio = "/data/" + value["songs"][0]["audio_path"]
        with get(audio, {"Range": "bytes=0-15"}) as response:
            assert response.status == 206
            assert response.read() == chart.song.path.read_bytes()[:16]
            assert response.headers["Content-Range"].startswith("bytes 0-15/")
        with get(audio, {"Range": "bytes=-10"}) as response:
            assert response.read() == chart.song.path.read_bytes()[-10:]
        with pytest.raises(urllib.error.HTTPError) as error:
            get(audio, {"Range": "bytes=9999999-"})
        assert error.value.code == 416
        with pytest.raises(urllib.error.HTTPError) as error:
            get("/data/%2e%2e/pyproject.toml")
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_empty_library(tmp_path):
    assert manifest(tmp_path)["songs"] == []
