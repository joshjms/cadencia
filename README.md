# Cadencia

A Python library for rhythm chart analysis with a shared `Chart` representation and a JavaScript timeline viewer. The first supported format is **four-key osu!mania**, with separate `Hit` and `Hold` events in lanes 0–3.

## Install and run

Requires Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

Download and preprocess 100 ranked four-key osu!mania charts:

```sh
uv run download_ranked_mania.py
```

Or use the library interactively:

```sh
uv sync
uv run python
```

```python
import cadencia

client = cadencia.osumania.HinamizawaClient()
page = client.get_ranked_mania({"limit": 10, "page": 0})
charts = client.download_charts({
    "urls": ["https://osu.ppy.sh/beatmapsets/2043161#mania/4855704"]
})
chart = cadencia.osumania.load_chart(url=charts[0].metadata["source_url"])

cadencia.osumania.visualize(port="8080")
```

Open [the timeline](http://127.0.0.1:8080/). The server blocks until Ctrl+C. Audio playback, seeking, zooming and the event inspector share the same timeline.

```python
chart = cadencia.osumania.load_chart(
    name=f"{chart.song.artist} - {chart.song.title} [{chart.metadata['difficulty']}]"
)

cadencia.osumania.visualize(
    port="8080",
    include=["lane0", "lane1", "waveform"],
)
```

`load_chart` requires exactly one of `name=` or `url=` and immediately reads an existing local `trace.json`. It never downloads or recomputes features, and raises `FileNotFoundError` if no preprocessed chart exists, even when an original archive is present. Names match an exact song title or `Artist - Title [Difficulty]`, ignoring case. Ambiguous names raise `ValueError`; use the full name or URL to select a difficulty.

## Ranked mania downloads

`HinamizawaClient.get_ranked_mania(params)` makes a GET request to the mirror's [advanced search API](https://mirror.hinamizawa.ai/docs/beatmap-search), returning its JSON object with a `beatmapsets` array and nested `beatmaps` difficulties. It fixes `mode=3` and `status=ranked`, defaults to `explicit=show`, `sort=ranked_asc`, `limit=100`, and zero-based `page=0`. The ranked category includes legacy Approved maps. Search results include all mania key counts.

`HinamizawaClient.download_charts(params)` accepts either an explicit `{"urls": [...]}` selection or the same search filters plus `count` (default 100). It downloads the selected charts' sets and runs chart and audio preprocessing, without silently skipping any key counts. Cadencia's parser still raises `ValueError` for unsupported charts. Four-key filtering and pagination for the 100-chart dataset live in `download_ranked_mania.py`: the script selects individual ranked four-key difficulties and passes their URLs to the client.

The client counts cached charts toward the requested total, reuses their traces, and reuses downloaded set archives across calls on the same client. If only an archive exists, the client rebuilds its trace. Search exhaustion, download failures and preprocessing failures raise errors; already processed charts remain available for rerunning the batch. Requests retry HTTP 429 and 5xx responses up to three attempts, honoring `Retry-After` delays up to 60 seconds; longer delays are reported to the caller.

```python
client = cadencia.osumania.HinamizawaClient(data_dir="data")
page = client.get_ranked_mania({"query": "Camellia", "limit": 25, "page": 0})
# Select chart URLs from the search results before downloading.
charts = client.download_charts({
    "urls": ["https://osu.ppy.sh/beatmapsets/2043161#mania/4855704"]
})
```

The script defaults to 100 charts and logs progress for each processed difficulty.

Charts without a standalone song recording raise `cadencia.osumania.MissingChartAudioError`. For example, some maps use `AudioFilename: virtual` and construct their music from storyboard samples and keysounds, which Cadencia does not render. The script catches individual download and preprocessing failures, logs the chart URL, exception type and message, and continues until the requested number of charts has been successfully processed. Keyboard interrupts still stop the run, and catalogue request failures or search exhaustion remain fatal. Rerunning reuses already processed charts.

```sh
uv run download_ranked_mania.py
uv run download_ranked_mania.py --count 10 --query Camellia --data-dir /path/to/data
```

`uv run main.py --manifest links.json` builds a beat dataset from charts already preprocessed locally. Its manifest is a JSON array of objects containing `link` URLs.

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

## NumPy descriptors

Pass a loaded chart to the top-level descriptor getters:

```python
import numpy as np

onsets = cadencia.get_onset_strength(chart, sampling_rate=100)
rms = cadencia.get_rms_energy(chart, sampling_rate=100)
chroma = cadencia.get_chroma(chart, sampling_rate=100)
waveform = cadencia.get_waveform(chart, sampling_rate=22050)

times = np.arange(len(onsets)) / 100
```

`sampling_rate` is the number of output samples per second. Sample `k` represents `k / sampling_rate` seconds from audio zero, and every timestamp is strictly before the song duration. Getters using the same rate return the same number of samples. Positive fractional rates are supported.

| Getter | Array shape | Default rate |
| --- | --- | --- |
| `get_waveform` | `(N,)`, mono audio amplitude | 22,050 Hz |
| `get_onset_strength` | `(N,)` | 50 Hz |
| `get_rms_energy` | `(N,)` | 50 Hz |
| `get_spectral_centroid` | `(N,)` | 50 Hz |
| `get_harmonic_energy_fraction` | `(N,)` | 50 Hz |
| `get_chroma` | `(N, 12)`, columns C through B | 50 Hz |
| `get_candidate_onsets` | `(N,)`, integer counts per bin | 50 Hz |
| `get_estimated_beats` | `(N,)`, integer counts per bin | 50 Hz |

Continuous descriptors use linear interpolation of cached feature values and hold the endpoint value beyond the last cached frame. A higher output rate interpolates the original 50 Hz analysis; it does not increase its measurement resolution. Undefined centroid and harmonic-fraction values remain `np.nan`, including interpolation intervals touching them. The original cached values and units are preserved.

`get_waveform` reads the original audio, averages channels and resamples with `soxr_hq` antialiasing. It returns audio samples; `chart.waveform` remains the min/max envelope used by the visualizer.

Candidate onsets and estimated beats are discrete event trains. Output bin `k` counts events in `[k / sampling_rate, (k + 1) / sampling_rate)`, clipped to the recording duration. Counts can exceed one at low rates. They are not interpolated into continuous curves, and candidate counts do not encode onset strength.

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

The default data directory is `data/` relative to the current working directory. Set `CADENCIA_DATA_DIR`, or pass `data_dir="/path/to/data"` to `HinamizawaClient`, `load_chart` and `visualize`, to choose another location. Cached media are ignored by version control.

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
