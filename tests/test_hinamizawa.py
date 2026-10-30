import io
import json
import urllib.error
import urllib.request
import zipfile
from urllib.parse import parse_qs, urlsplit

import pytest

from cadencia.osumania import HinamizawaClient, load_chart
from conftest import LINK, beatmap, ranked_set


def response(value, url="https://mirror.hinamizawa.ai/example"):
    stream = io.BytesIO(value if isinstance(value, bytes) else json.dumps(value).encode())
    stream.url = url
    return stream


def test_search_uses_get_and_documented_filters(tmp_path, monkeypatch):
    requests = []
    result = {"beatmapsets": [ranked_set(4855704)], "source": "local", "total_count": 1}

    def open_url(request, timeout):
        requests.append(request)
        assert timeout == 12
        return response(result)

    monkeypatch.setattr(urllib.request, "urlopen", open_url)
    params = {"query": "artist & title", "page": 2, "limit": 25, "pp_enrichment": True}
    client = HinamizawaClient(data_dir=tmp_path, timeout=12)
    assert client.get_ranked_mania(params) == result
    request = requests[0]
    assert request.get_method() == "GET"
    assert request.get_header("User-agent") == "cadencia/0.1.0"
    assert urlsplit(request.full_url).path == "/v3/osu/beatmaps/search/v2"
    assert parse_qs(urlsplit(request.full_url).query) == {
        "query": ["artist & title"], "page": ["2"], "limit": ["25"],
        "mode": ["3"], "status": ["ranked"], "explicit": ["show"],
        "sort": ["ranked_asc"], "pp_enrichment": ["true"],
    }
    assert params["page"] == 2
    assert "mode" not in params


@pytest.mark.parametrize("params", [
    {"mode": 0}, {"status": "loved"}, {"limit": 101}, {"limit": "all"},
    {"page": -1}, {"page": 0.5}, {"count": 10}, {"typo": "ignored"},
])
def test_invalid_search_parameters_fail_before_network(tmp_path, monkeypatch, params):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid parameters reached the network")

    monkeypatch.setattr(urllib.request, "urlopen", unexpected)
    with pytest.raises(ValueError):
        HinamizawaClient(data_dir=tmp_path).get_ranked_mania(params)


def test_pagination_deduplicates_and_downloads_each_set_once(tmp_path, package, monkeypatch):
    client = HinamizawaClient(data_dir=tmp_path)
    queries, downloads = [], []
    unsupported = {"id": 9, "beatmaps": [
        {"id": 11, "mode": "osu", "cs": 4, "ranked": 1},
        {"id": 12, "mode": "mania", "cs": 4, "ranked": 4},
    ]}
    pages = [
        {"beatmapsets": [unsupported]},
        {"beatmapsets": [ranked_set(4855704, 4855704, 1)]},
    ]

    def search(params):
        queries.append(params)
        return pages[params["page"]]

    def open_url(request, timeout):
        downloads.append(request.full_url)
        return response(package.read_bytes(), "https://example.com/mirror.osz")

    monkeypatch.setattr(client, "get_ranked_mania", search)
    monkeypatch.setattr(urllib.request, "urlopen", open_url)
    params = {"count": 2, "limit": 10}
    charts = client.download_charts(params)
    assert params == {"count": 2, "limit": 10}
    assert [chart.id for chart in charts] == [4855704, 1]
    assert [query["page"] for query in queries] == [0, 1]
    assert len(downloads) == 1
    assert downloads[0].endswith("/api/v1/hinai/d/2043161?no_video=true")
    assert charts[0].metadata["resolved_download_url"] == "https://example.com/mirror.osz"
    assert load_chart(url=LINK, data_dir=tmp_path).to_dict() == charts[0].to_dict()
    assert all(chart.song.path.is_file() for chart in charts)
    assert [chart.id for chart in client.download_charts(params)] == [4855704, 1]
    assert len(downloads) == 1


def test_search_exhaustion_reports_shortfall(tmp_path, monkeypatch):
    client = HinamizawaClient(data_dir=tmp_path)
    monkeypatch.setattr(client, "get_ranked_mania", lambda params: {"beatmapsets": []})
    with pytest.raises(RuntimeError, match="0 of 100"):
        client.download_charts()
    assert not (tmp_path / "osumania").exists()


def test_repeated_pages_do_not_loop_forever(tmp_path, monkeypatch):
    client = HinamizawaClient(data_dir=tmp_path)
    non_mania = ranked_set(1)
    non_mania["beatmaps"][0]["mode"] = "osu"
    monkeypatch.setattr(client, "get_ranked_mania", lambda params: {"beatmapsets": [non_mania]})
    with pytest.raises(RuntimeError, match="without advancing"):
        client.download_charts()


@pytest.mark.parametrize("value", [[], {"error": "unavailable"}])
def test_invalid_response_is_not_treated_as_search_exhaustion(tmp_path, monkeypatch, value):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: response(value))
    with pytest.raises(ValueError, match="Unexpected"):
        HinamizawaClient(data_dir=tmp_path).get_ranked_mania()


def test_retries_rate_limit_with_retry_after(tmp_path, monkeypatch):
    attempts, waits = [], []

    def open_url(request, timeout):
        attempts.append(request)
        if len(attempts) == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "rate limited", {"Retry-After": "3"}, None)
        return response({"beatmapsets": []})

    monkeypatch.setattr(urllib.request, "urlopen", open_url)
    monkeypatch.setattr("cadencia.osumania.hinamizawa.time.sleep", waits.append)
    HinamizawaClient(data_dir=tmp_path).get_ranked_mania()
    assert len(attempts) == 2
    assert waits == [3]


def test_http_client_error_is_not_retried(tmp_path, monkeypatch):
    attempts = []

    def open_url(request, timeout):
        attempts.append(request)
        raise urllib.error.HTTPError(request.full_url, 404, "missing", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", open_url)
    with pytest.raises(RuntimeError, match="HTTP 404"):
        HinamizawaClient(data_dir=tmp_path).get_ranked_mania()
    assert len(attempts) == 1


def test_failed_download_does_not_publish_cache(tmp_path, monkeypatch):
    client = HinamizawaClient(data_dir=tmp_path)
    monkeypatch.setattr(client, "get_ranked_mania", lambda params: {"beatmapsets": [ranked_set(4855704)]})

    def download(set_id, destination):
        destination.write_bytes(b"partial archive")
        raise RuntimeError("Connection lost")

    monkeypatch.setattr(client, "_download", download)
    with pytest.raises(RuntimeError, match="Connection lost"):
        client.download_charts({"count": 1})
    assert list((tmp_path / "osumania" / "charts").iterdir()) == []


def test_client_does_not_silently_skip_non_four_key_charts(tmp_path, package, monkeypatch):
    client = HinamizawaClient(data_dir=tmp_path / "data")
    seven_key = ranked_set(4855704)
    seven_key["beatmaps"][0]["cs"] = 7
    monkeypatch.setattr(client, "get_ranked_mania", lambda params: {"beatmapsets": [seven_key]})
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(package) as source, zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("audio.wav", source.read("audio.wav"))
        archive.writestr("seven.osu", beatmap("64,192,500,1,0", keys=7))
    downloads = []

    def open_url(request, timeout):
        downloads.append(request.full_url)
        return response(archive_bytes.getvalue())

    monkeypatch.setattr(urllib.request, "urlopen", open_url)
    with pytest.raises(ValueError, match="four-key"):
        client.download_charts({"count": 1})
    assert len(downloads) == 1


def test_explicit_urls_bypass_search_and_reuse_archives(tmp_path, package, monkeypatch):
    client = HinamizawaClient(data_dir=tmp_path)
    downloads = []

    def unexpected(params):
        pytest.fail("Explicit selection should not trigger a search")

    def open_url(request, timeout):
        downloads.append(request.full_url)
        return response(package.read_bytes())

    monkeypatch.setattr(client, "get_ranked_mania", unexpected)
    monkeypatch.setattr(urllib.request, "urlopen", open_url)
    charts = client.download_charts({"urls": [LINK, LINK]})
    charts.extend(client.download_charts({"urls": ["https://osu.ppy.sh/beatmapsets/2043161#mania/1"]}))
    assert [chart.id for chart in charts] == [4855704, 1]
    assert len(downloads) == 1


@pytest.mark.parametrize("params", [
    {"urls": LINK}, {"urls": [LINK], "count": 1}, {"urls": ["invalid"]},
])
def test_explicit_urls_are_validated_before_downloading(tmp_path, monkeypatch, params):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("Network attempted"))
    with pytest.raises(ValueError):
        HinamizawaClient(data_dir=tmp_path).download_charts(params)
