# Cadencia

A Python library for rhythm chart analysis with a shared `Chart` representation and a JavaScript timeline viewer. The first supported format is **four-key osu!mania**, with separate `Hit` and `Hold` events in lanes 0–3.

## Install and run

Requires Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run python
```

```python
import cadencia

link = "https://osu.ppy.sh/beatmapsets/2043161#mania/4855704"
chart = cadencia.osumania.load_chart(link)

cadencia.osumania.visualize(port="8080")
```

Open [the timeline](http://127.0.0.1:8080/). The server blocks until Ctrl+C. Audio playback, seeking, zooming and the event inspector share the same timeline.

```python
chart = cadencia.osumania.load_chart(link, download=False)

cadencia.osumania.visualize(
    port="8080",
    include=["lane0", "lane1", "waveform"],
)
```

`download=False` raises `FileNotFoundError` when the chart is absent locally. A cached `trace.json` loads without downloading or recomputing features. If only the original archive is present, the library can rebuild the trace locally.

Both `include=["lane0", "lane1", "waveform"]` and `include=["lane0, lane1, waveform"]` work. Omitting `include` displays every layer. Available names:

```python
[
    "lane0", "lane1", "lane2", "lane3",
    "waveform", "onset_strength", "candidate_onsets", "rms_energy",
    "spectral_centroid", "harmonic_energy_fraction", "chroma",
    "estimated_beats", "timing",
]
```

For notebooks or embedding, `visualize(block=False)` returns a running HTTP server. Call `server.shutdown()` and `server.server_close()` when finished. `port=0` selects an available port, exposed as `server.server_port`.

## Chart and events

`cadencia.Chart` contains the shared song, events and audio features. `cadencia.osumania.Chart` extends it with osu!mania loading and uses independent `cadencia.osumania.Event`, `Hit` and `Hold` types. Future game adapters can define their own event types against the shared chart representation.

```python
assert isinstance(chart, cadencia.Chart)

song_path = chart.song.path
lane_events = chart.lanes[0]
all_events = chart.events

for event in all_events:
    if isinstance(event, cadencia.osumania.Hold):
        print(event.lane, event.start_s, event.end_s, event.duration_s)
```

`chart.lanes` always has keys `0`, `1`, `2`, `3`, including empty lanes. `chart.events` returns all events in time order. Events retain their source line, object flags, source x coordinate and stable source ID. Simultaneous events stay separate. Holds use the explicit release time in the source file.

Seven-key maps and other game modes raise `ValueError`. Unsupported or malformed objects are retained in `chart.raw_events` with diagnostics, and the metadata reports partial decoding. Events beyond the recording bounds retain their timestamps with `analysis_eligible=False`.

## Cache and provenance

The example creates:

```text
data/osumania/charts/4855704/
    archive.osz
    chart.osu
    song.mp3
    download.json
    trace.json
```

`archive.osz` is the original beatmap set. `chart.osu` is the selected difficulty, and `song.<extension>` preserves its associated audio bytes. The trace stores the song filename, metadata, four event lanes, every feature series, timing points, source records and diagnostics. `chart.trace_path`, `chart.to_dict()` and `chart.save()` expose this representation. `cadencia.osumania.Chart.read(path)` reads an existing trace.

Downloads use the [Hinamizawa public beatmap mirror](https://mirror.hinamizawa.ai/docs/beatmap-download). The provider, requested URL, resolved URL and download time are recorded. SHA-256 hashes identify the archive, selected chart and song. Only the requested chart and its associated song are analyzed. Downloads and analysis are staged before a new cache becomes visible.

The default data directory is `data/` relative to the current working directory. Set `CADENCIA_DATA_DIR`, or pass `data_dir="/path/to/data"` to both `load_chart` and `visualize`, to choose another location. Cached media are ignored by version control.

## Audio features and timing

All event and feature times are seconds relative to the original audio start. Object times are divided by 1,000; no lead-in or calibration shift is applied. Source tempo and meter come from uninherited timing points. Inherited scroll controls remain context and do not change event times or BPM. See the [osu! file format](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_%28file_format%29).

The original audio is retained for playback. Analysis averages channels to mono and resamples to 22,050 Hz with `soxr_hq`. Centered 2,048-sample windows advance by 441 samples (20 ms), with zero padding at the boundaries. Analysis settings and library versions are recorded in `chart.metadata["analysis"]`.

| Attribute | Representation |
| --- | --- |
| `song` | Local audio path, title, artist, duration, native sample rate/channels and hash |
| `waveform` | Minimum and maximum amplitude in each 20 ms bin; timestamps mark bin starts |
| `onset_strength` | Librosa positive log-mel spectral flux; a measure of changing spectral energy |
| `candidate_onsets` | Local peaks in the onset envelope, with their original strengths |
| `rms_energy` | Root-mean-square amplitude in rectangular windows, in linear full-scale units |
| `spectral_centroid` | Magnitude-weighted spectral center in Hz; null for silent frames |
| `harmonic_energy_fraction` | Squared HPSS harmonic magnitude divided by total squared magnitude; null for silent frames |
| `chroma` | Twelve harmonic pitch-class values per frame, C through B, normalized to the frame maximum |
| `estimated_beats` | Beat times estimated independently from the audio onset envelope |

Feature series expose `.times`, `.values`, `.unit` and optional `.labels`. Frame timestamps mark window centers. Waveform bins expose `.times`, `.minimum`, `.maximum` and `.bin_width_s`.

Candidate detection divides flux by its per-song 99th percentile and clips to [0, 1]. Peak parameters are `pre_max=3`, `post_max=3`, `pre_avg=10`, `post_avg=10`, `delta=0.07`, `wait=3`, without backtracking. The trace retains raw flux. The harmonic fraction is an HPSS mixture proxy; candidate onsets and estimated beats are audio estimates, not chart labels. The viewer scales flux and RMS by their displayed per-song maxima and shows those maxima.

## Development

```sh
uv run pytest
uv build
```

The viewer ships inside the Python package and needs no frontend build or Node installation. Tests cover parsing, explicit holds, source timing, feature extraction, cache reuse, failed imports, lane filtering, HTTP assets and byte-range audio playback.
