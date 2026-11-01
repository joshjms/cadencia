"""Explicit chart acquisition through the public Hinamizawa mirror."""

import json
import logging
import math
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlencode

from . import loader
from .chart import Chart

logger = logging.getLogger(__name__)
BASE_URL = "https://mirror.hinamizawa.ai"
SEARCH_PARAMS = {
    "query", "mode", "status", "sort", "page", "limit", "creator", "genre", "language",
    "min_stars", "max_stars", "min_bpm", "max_bpm", "min_length", "max_length",
    "min_pp", "max_pp", "pp_enrichment", "explicit", "extra",
}


def _integer(value, name, minimum, maximum=None):
    if isinstance(value, bool) or not str(value).isdigit():
        raise ValueError(f"{name} must be an integer >= {minimum}")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be between {minimum} and {maximum or 'unbounded'}")
    return value


class HinamizawaClient:
    """Search ranked mania sets and download/preprocess selected charts.

    ``params`` is a mapping of documented advanced-search query parameters.
    ``download_charts`` also accepts ``count`` (default 100), counting individual
    difficulties, including already processed ones, or an explicit ``urls`` list.
    Search fixes ranked mania, includes explicit maps, and defaults to
    oldest-ranked-first ordering. Key-count selection belongs to the caller.
    """

    def __init__(self, *, data_dir=None, timeout=180, user_agent="cadencia/1.0"):
        self.data_dir = loader.data_root(data_dir)
        self.timeout = timeout
        self.user_agent = user_agent
        self._archives = {}
        self._ratings = {}

    def _open(self, url):
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        for attempt in range(3):
            try:
                return urllib.request.urlopen(request, timeout=self.timeout)
            except urllib.error.HTTPError as error:
                delay = 2 ** attempt
                retry_after = error.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds()
                retryable = error.code == 429 or 500 <= error.code < 600
                details = f"HTTP {error.code}"
                if retry_after:
                    details += f"; Retry-After: {retry_after}"
                forensics = error.headers.get("x-hinai-forensics")
                if forensics:
                    details += f"; diagnostics: {forensics}"
                error.close()
                if not retryable or attempt == 2 or delay > 60:
                    raise RuntimeError(f"GET {url} failed: {details}") from error
                time.sleep(max(0, delay))
            except urllib.error.URLError as error:
                raise RuntimeError(f"GET {url} failed: {error}") from error

    def get_ranked_mania(self, params=None) -> dict:
        """GET one page, returning the API object with nested ``beatmapsets``.

        Pages are zero-based and contain at most 100 sets. Child ``beatmaps``
        include all mania key counts; only four-key charts can be preprocessed.
        The ranked category includes legacy Approved maps, like osu! search.
        """
        query = dict(params or {})
        unknown = query.keys() - SEARCH_PARAMS
        if unknown:
            raise ValueError(f"Unknown search parameters: {', '.join(sorted(unknown))}")
        if str(query.get("mode", 3)) != "3" or query.get("status", "ranked") != "ranked":
            raise ValueError("get_ranked_mania requires mode=3 and status=ranked")
        query = {"explicit": "show", "sort": "ranked_asc", "limit": 100, "page": 0,
                 **query, "mode": 3, "status": "ranked"}
        query["limit"] = _integer(query["limit"], "limit", 1, 100)
        query["page"] = _integer(query["page"], "page", 0)
        query = {key: str(value).lower() if isinstance(value, bool) else value
                 for key, value in query.items() if value is not None}
        with self._open(f"{BASE_URL}/v3/osu/beatmaps/search/v2?{urlencode(query)}") as response:
            result = json.load(response)
        if not isinstance(result, dict) or not isinstance(result.get("beatmapsets"), list):
            raise ValueError("Unexpected Hinamizawa search response: expected beatmapsets array")
        for beatmapset in result['beatmapsets']:
            for beatmap in beatmapset.get('beatmaps', []):
                rating = beatmap.get('difficulty_rating')
                if isinstance(rating, (int, float)) and not isinstance(rating, bool) and math.isfinite(rating) and rating > 0:
                    self._ratings[int(beatmap['id'])] = float(rating)
        return result

    def _download(self, set_id: int, destination: Path) -> dict:
        url = f"{BASE_URL}/api/v1/hinai/d/{set_id}?no_video=true"
        with self._open(url) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)
            resolved_url = response.url
        return dict(download_url=url, resolved_download_url=resolved_url,
                    downloaded_at=datetime.now(timezone.utc).isoformat(),
                    provider="Hinamizawa public beatmap mirror")

    def _prepare_chart(self, set_id: int, chart_id: int, archives: dict) -> Chart:
        url = f"https://osu.ppy.sh/beatmapsets/{set_id}#mania/{chart_id}"
        folder = self.data_dir / "osumania" / "charts" / str(chart_id)
        trace = folder / "trace.json"
        if trace.is_file():
            chart = loader.load_chart(url=url, data_dir=self.data_dir)
        elif (folder / "archive.osz").is_file():
            # Reprocessing an existing archive is explicit client work, never load_chart work.
            provenance_path = folder / "download.json"
            provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
            chart = loader._build_chart(folder, set_id, chart_id, url, provenance)
        else:
            if folder.exists() and any(folder.iterdir()):
                raise FileExistsError(f"Incomplete cache at {folder}; inspect it before retrying")
            folder.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=f".{chart_id}-", dir=folder.parent) as temporary:
                staging = Path(temporary)
                if set_id in archives and archives[set_id][0].is_file():
                    source, provenance = archives[set_id]
                    shutil.copyfile(source, staging / "archive.osz")
                else:
                    provenance = self._download(set_id, staging / "archive.osz")
                (staging / "download.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
                loader._build_chart(staging, set_id, chart_id, url, provenance)
                if folder.exists():
                    folder.rmdir()
                staging.rename(folder)
            chart = Chart.read(trace)
        if (folder / "archive.osz").is_file():
            path = folder / "download.json"
            archives[set_id] = (folder / "archive.osz", json.loads(path.read_text()) if path.is_file() else {})
        if chart_id in self._ratings:
            chart.metadata['star_rating'] = self._ratings[chart_id]
            chart.metadata['star_rating_source'] = 'Hinamizawa difficulty_rating'
            chart.save()
        return chart

    def download_charts(self, params=None) -> list[Chart]:
        """Download/preprocess explicit ``urls`` or ``count`` search results.

        Reuses processed traces and archives downloaded by this client.
        Returns local Charts. Raises if the search is exhausted before ``count``
        unique charts are processed; successfully cached charts remain available
        for a later retry. No key counts are silently skipped: unsupported charts
        raise a preprocessing error. Callers can select charts using
        ``get_ranked_mania`` and pass their URLs as ``{"urls": [...]}``.
        """
        query = dict(params or {})
        if "urls" in query:
            urls = query.pop("urls")
            if query or not isinstance(urls, (list, tuple)):
                raise ValueError("urls must be a list of chart URLs and cannot be combined with search parameters")
            identities = dict.fromkeys(loader.chart_ids(url) for url in urls)
            charts, archives = [], self._archives
            for set_id, chart_id in identities:
                charts.append(self._prepare_chart(set_id, chart_id, archives))
                logger.info("Processed %s/%s: %s", len(charts), len(identities), charts[-1].metadata["source_url"])
            return charts
        count = _integer(query.pop("count", 100), "count", 1)
        page = _integer(query.get("page", 0), "page", 0)
        charts, seen, seen_sets, archives = [], set(), set(), self._archives
        while len(charts) < count:
            result = self.get_ranked_mania({**query, "page": page})
            if not result["beatmapsets"]:
                raise RuntimeError(f"Search exhausted after {len(charts)} of {count} ranked charts")
            set_ids = {int(beatmapset["id"]) for beatmapset in result["beatmapsets"]}
            if set_ids <= seen_sets:
                raise RuntimeError("Hinamizawa pagination repeated previously seen sets without advancing")
            seen_sets.update(set_ids)
            for beatmapset in result["beatmapsets"]:
                for beatmap in beatmapset.get("beatmaps", []):
                    mode = beatmap.get("mode_int", beatmap.get("mode"))
                    status = beatmap.get("ranked", beatmap.get("status", beatmapset.get("status")))
                    if mode not in (3, "mania") or status not in (1, 2, "ranked", "approved"):
                        continue
                    chart_id = int(beatmap["id"])
                    if chart_id in seen:
                        continue
                    seen.add(chart_id)
                    charts.append(self._prepare_chart(int(beatmapset["id"]), chart_id, archives))
                    logger.info("Processed %s/%s: %s", len(charts), count, charts[-1].metadata["source_url"])
                    if len(charts) == count:
                        return charts
            # Ignore short pages: upstream post-filtering may produce them.
            page += 1
        return charts
