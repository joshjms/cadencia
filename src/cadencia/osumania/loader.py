import hashlib
import json
import os
import re
import zipfile
from pathlib import Path, PureWindowsPath
from urllib.parse import urlsplit

from ..analysis import analyze_song
from .chart import Chart
from .parser import parse_chart, sections, validate_headers


class MissingChartAudioError(FileNotFoundError):
    """A chart has no standalone song recording that Cadencia can analyze."""


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
            explanation = (
                "This chart uses virtual audio; rendering storyboard samples and keysounds is not supported."
                if filename.casefold() == "virtual" else "Associated song is missing from the archive."
            )
            raise MissingChartAudioError(
                f"Chart {chart_id} in set {set_id}: {explanation} AudioFilename: {filename!r} ({link})"
            )
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


def load_chart(*, name: str | None = None, url: str | None = None, data_dir=None) -> Chart:
    """Read a preprocessed local chart by URL or case-insensitive exact name.

    Names match a song title or ``Artist - Title [Difficulty]``. Ambiguous
    titles require the full name or URL. This never downloads or analyzes audio.
    """
    if (name is None) == (url is None):
        raise ValueError("Provide exactly one of name= or url=")
    charts = data_root(data_dir) / "osumania" / "charts"
    if url is not None:
        set_id, chart_id = chart_ids(url)
        trace = charts / str(chart_id) / "trace.json"
        if not trace.is_file():
            raise FileNotFoundError(f"Chart {chart_id} is not preprocessed locally at {trace}")
        chart = Chart.read(trace)
        if chart.id != chart_id or chart.metadata["beatmapset_id"] != set_id:
            raise ValueError("Cached chart identity does not match the requested link")
        return chart

    if not name.strip():
        raise ValueError("name= must not be empty")
    matches = []
    for trace in sorted(charts.glob("*/trace.json")):
        value = json.loads(trace.read_text(encoding="utf-8"))
        song, metadata = value["song"], value["metadata"]
        headers = metadata.get("headers", {}).get("Metadata", {})
        titles = {song["title"], headers.get("Title", ""), headers.get("TitleUnicode", "")}
        artists = {song["artist"], headers.get("Artist", ""), headers.get("ArtistUnicode", "")}
        names = titles | {
            f"{artist} - {title} [{metadata['difficulty']}]"
            for artist in artists if artist for title in titles if title
        }
        if name.strip().casefold() in {candidate.casefold() for candidate in names}:
            matches.append(trace)
    if not matches:
        raise FileNotFoundError(f"No preprocessed local chart named {name!r} in {charts}")
    if len(matches) > 1:
        raise ValueError(f"Multiple local charts match {name!r}; use url= or Artist - Title [Difficulty]")
    return Chart.read(matches[0])
