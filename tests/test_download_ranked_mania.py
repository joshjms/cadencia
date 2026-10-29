import pytest

from download_ranked_mania import download_ranked_four_key
from conftest import LINK, ranked_set


def test_script_selects_four_key_charts_across_pages_before_downloading():
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
    assert len(client.downloads) == 1


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
