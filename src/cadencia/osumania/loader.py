import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from urllib.parse import urlsplit

from ..analysis import analyze_song
from .chart import Chart
from .parser import parse_chart, sections, validate_headers

DOWNLOAD_URL = "https://mirror.hinamizawa.ai/api/v1/hinai/d/{set_id}?no_video=true"


def data_root(data_dir=None) -> Path:
    return Path(data_dir or os.environ.get("CADENCIA_DATA_DIR", "data")).resolve()


def chart_ids(link: str) -> tuple[int, int]:
    markdown = re.fullmatch(r"\[[^\]]*\]\((https://[^)]+)\)", link.strip())
    url = urlsplit(markdown[1] if markdown else link.strip())
    path = re.fullmatch(r"/beatmapsets/(\d+)/?", url.path)
    fragment = re.fullmatch(r"mania/(\d+)", url.fragment)
    if url.scheme != "https" or url.hostname != "osu.ppy.sh" or not path or not fragment:
        raise ValueError("Use an osu!mania link: https://osu.ppy.sh/beatmapsets/<set_id>#mania/<chart_id>")
    return int(path[1]), int(fragment[1])


def _download(set_id: int, destination: Path) -> dict:
    url = DOWNLOAD_URL.format(set_id=set_id)
    request = urllib.request.Request(url, headers={"User-Agent": "cadencia/0.1.0"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            with destination.open("wb") as output:
                shutil.copyfileobj(response, output)
            resolved_url = response.url
    except (urllib.error.URLError, OSError) as error:
        raise RuntimeError(f"Could not download beatmap set {set_id} from {url}: {error}") from error
    return dict(download_url=url, resolved_download_url=resolved_url,
                downloaded_at=datetime.now(timezone.utc).isoformat(), provider="Hinamizawa public beatmap mirror")


def _archive_files(archive: zipfile.ZipFile) -> dict[str, str]:
    files = {}
    for member in archive.infolist():
        name = member.filename.replace("\\", "/")
        path = Path(name)
        if path.is_absolute() or PureWindowsPath(name).drive or ".." in path.parts:
            raise ValueError(f"Unsafe archive path: {name}")
        if member.is_dir():
            continue
        if (member.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError(f"Archive symlink is unsupported: {name}")
        if name.casefold() in files:
            raise ValueError(f"Ambiguous archive filename: {name}")
        files[name.casefold()] = member.filename
    return files


def _build_chart(folder: Path, set_id: int, chart_id: int, link: str, provenance: dict) -> Chart:
    archive_path = folder / "archive.osz"
    with zipfile.ZipFile(archive_path) as archive:
        files = _archive_files(archive)
        selected = []
        for name in files.values():
            if name.lower().endswith(".osu"):
                payload = archive.read(name)
                headers, _ = sections(payload)
                if headers.get("Metadata", {}).get("BeatmapID") == str(chart_id):
                    selected.append((name, payload, headers))
        if len(selected) != 1:
            raise ValueError(f"Expected exactly one beatmap with ID {chart_id}; found {len(selected)}")
        name, payload, headers = selected[0]
        validate_headers(headers)
        filename = headers["General"]["AudioFilename"].strip('"').replace("\\", "/")
        relative = Path(filename)
        if relative.is_absolute() or PureWindowsPath(filename).drive or ".." in relative.parts:
            raise ValueError("Unsafe AudioFilename in chart")
        audio_key = (Path(name.replace("\\", "/")).parent / relative).as_posix().casefold()
        if audio_key not in files:
            raise FileNotFoundError(f"Associated song is missing from archive: {filename}")
        song_path = folder / ("song" + Path(filename).suffix.lower())
        song_path.write_bytes(archive.read(files[audio_key]))
        (folder / "chart.osu").write_bytes(payload)

    metadata = headers.get("Metadata", {})
    features = analyze_song(song_path, metadata.get("TitleUnicode") or metadata.get("Title", "Untitled"),
                            metadata.get("ArtistUnicode") or metadata.get("Artist", "Unknown"))
    parsed = parse_chart(payload, features["song"].duration_s)
    parsed["metadata"].update(
        source_url=link, beatmapset_id=set_id, beatmap_id=chart_id,
        archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        analysis=features.pop("analysis_metadata"), **provenance,
    )
    chart = Chart(id=chart_id, game="osumania", **features, **parsed)
    chart.save()
    return chart


def load_chart(link: str, download: bool = True, *, data_dir=None) -> Chart:
    set_id, chart_id = chart_ids(link)
    link = f"https://osu.ppy.sh/beatmapsets/{set_id}#mania/{chart_id}"
    folder = data_root(data_dir) / "osumania" / "charts" / str(chart_id)
    trace = folder / "trace.json"
    if trace.is_file():
        chart = Chart.read(trace)
        if chart.id != chart_id or chart.metadata["beatmapset_id"] != set_id:
            raise ValueError("Cached chart identity does not match the requested link")
        return chart
    archive = folder / "archive.osz"
    if archive.is_file():
        provenance_path = folder / "download.json"
        provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
        return _build_chart(folder, set_id, chart_id, link, provenance)
    if not download:
        raise FileNotFoundError(f"Chart {chart_id} is not cached at {folder}; automatic downloads are disabled")

    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{chart_id}-", dir=folder.parent) as temporary:
        staging = Path(temporary)
        provenance = _download(set_id, staging / "archive.osz")
        (staging / "download.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        _build_chart(staging, set_id, chart_id, link, provenance)
        if folder.exists():
            if any(folder.iterdir()):
                raise FileExistsError(f"Incomplete cache at {folder}; inspect it before retrying")
            folder.rmdir()
        staging.rename(folder)
    return Chart.read(trace)
