"""Build and read compact hit-prediction datasets from local chart traces."""

import hashlib
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import descriptors
from .osumania.chart import Chart
from .osumania.events import Hit
from .osumania.loader import data_root

logger = logging.getLogger(__name__)
SCHEMA_VERSION = "cadencia.hit_dataset.v1"
FEATURES = [
    "waveform", "onset_strength", "rms_energy", "spectral_centroid",
    "harmonic_energy_fraction", "chroma", "candidate_onsets", "estimated_beats",
    "hit_history",
]
FEATURE_COLUMNS = [
    "waveform", "onset_strength", "rms_energy", "spectral_centroid",
    "harmonic_energy_fraction",
    *[f"chroma_{name}" for name in ("C", "C_sharp", "D", "D_sharp", "E", "F", "F_sharp", "G", "G_sharp", "A", "A_sharp", "B")],
    "candidate_onsets", "estimated_beats", "hit_history",
]


def chart_samples(chart, *, window_s=2, sampling_rate=50):
    """Return X[N, steps, 20], binary y[N], and target times[N].

    This preserves the existing beat_dataset baseline: candidate times are the
    union of beat-grid timestamps (rounded to milliseconds) and eligible tap
    times. Holds do not count as taps. Historical hits exclude the target itself.
    """
    if not np.isfinite(window_s) or not np.isfinite(sampling_rate) or window_s <= 0 or sampling_rate <= 0:
        raise ValueError("window_s and sampling_rate must be positive and finite")
    steps = round(window_s * sampling_rate)
    if steps < 1 or not np.isclose(steps, window_s * sampling_rate):
        raise ValueError("window_s * sampling_rate must be a positive integer")
    beats = np.asarray([beat["time_s"] for beat in chart.beat_grid], dtype=float)
    if not len(beats):
        raise ValueError("The chart has no BPM beat grid")
    hits = np.sort(np.asarray([
        event.start_s for event in chart.events
        if isinstance(event, Hit) and event.analysis_eligible
    ], dtype=float))
    targets = np.unique(np.concatenate([np.round(beats, 3), hits]))
    signals = {
        name: getattr(descriptors, f"get_{name}")(chart, sampling_rate=sampling_rate)
        for name in FEATURES[:-1]
    }
    times = np.arange(len(signals["onset_strength"])) / sampling_rate
    ends = np.searchsorted(times, targets)
    complete = (targets >= window_s) & (targets < chart.song.duration_s) & (ends >= steps)
    targets, ends = targets[complete], ends[complete]
    if not len(targets):
        raise ValueError("The chart has no complete prediction windows")
    windows = ends[:, None] - steps + np.arange(steps)
    channels = []
    for name, values in signals.items():
        if len(values) != len(times):
            raise ValueError(f"Descriptor {name} does not cover the expected audio duration")
        values = np.asarray(values[windows], dtype=np.float32)
        channels.append(values[..., None] if values.ndim == 2 else values)
    edges = targets[:, None] - window_s + np.linspace(0, window_s, steps + 1)
    edges[:, -1] = targets
    history = np.diff(np.searchsorted(hits, edges, side="left"), axis=1)
    channels.append(history[..., None].astype(np.float32))
    return np.concatenate(channels, axis=-1), np.isin(targets, hits).astype(np.uint8), targets


def _assign_splits(entries, seed):
    """Keep shared sets and byte-identical recordings together, transitively."""
    parent = list(range(len(entries)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owners = {}
    for index, entry in enumerate(entries):
        for key in [("set", entry["song_id"]), ("audio", entry["audio_sha256"])]:
            if key in owners:
                parent[root(index)] = root(owners[key])
            else:
                owners[key] = index
    groups = {}
    for index, entry in enumerate(entries):
        groups.setdefault(root(index), []).append(entry)
    components = sorted(groups.values(), key=lambda group: min(entry["chart_id"] for entry in group))
    order = np.random.default_rng(seed).permutation(len(components))
    holdout = max(1, round(len(components) * 0.1)) if len(components) >= 3 else 0
    assignment = {int(index): "train" for index in order}
    for index in order[:holdout]:
        assignment[int(index)] = "test"
    for index in order[holdout:2 * holdout]:
        assignment[int(index)] = "validation"
    for index, group in enumerate(components):
        group_id = min(entry["song_id"] for entry in group)
        for entry in group:
            entry.update(group_id=group_id, split=assignment[index])
    return len(components)


def build_dataset(output, *, data_dir=None, window_s=2, sampling_rate=50, seed=42):
    """Write one compressed NPZ per cached chart and an auditable JSON manifest.

    Only existing local traces/audio are read. Per-chart failures are logged and
    recorded. The completed directory is published after every shard is written.
    """
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Dataset already exists: {output}; choose a new output directory")
    root = data_root(data_dir)
    traces = sorted((root / "osumania/charts").glob("*/trace.json"), key=lambda path: int(path.parent.name))
    if not traces:
        raise FileNotFoundError(f"No preprocessed charts found in {root / 'osumania/charts'}")
    output.parent.mkdir(parents=True, exist_ok=True)
    entries, failures = [], []
    missing = np.zeros(len(FEATURE_COLUMNS), dtype=np.int64)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        for index, trace in enumerate(traces, 1):
            shard = None
            try:
                chart = Chart.read(trace)
                x, y, times = chart_samples(chart, window_s=window_s, sampling_rate=sampling_rate)
                shard = staging / f"{chart.id}.npz"
                np.savez_compressed(shard, x=x, y=y, time_s=times)
                entry = dict(
                    chart_id=chart.id, song_id=int(chart.metadata["beatmapset_id"]),
                    title=chart.song.title, artist=chart.song.artist, difficulty=chart.metadata["difficulty"],
                    source_url=chart.metadata["source_url"], source_trace=str(trace),
                    trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                    audio_sha256=chart.song.sha256, chart_sha256=chart.metadata["chart_sha256"],
                    file=shard.name, sha256=hashlib.sha256(shard.read_bytes()).hexdigest(),
                    rows=len(y), positives=int(y.sum()), negatives=int(len(y) - y.sum()),
                )
                feature_missing = (~np.isfinite(x)).sum(axis=(0, 1))
                entries.append(entry)
                missing += feature_missing
                logger.info("Dataset %s/%s: chart %s, %s rows", index, len(traces), chart.id, len(y))
            except Exception as error:
                if shard is not None:
                    shard.unlink(missing_ok=True)
                failures.append(dict(trace=str(trace), error_type=type(error).__name__, error=str(error)))
                logger.warning("Skipping %s: %s: %s", trace, type(error).__name__, error)
        if not entries:
            raise ValueError("No charts produced usable dataset examples")
        groups = _assign_splits(entries, seed)
        splits = {}
        for split in ("train", "validation", "test"):
            selected = [entry for entry in entries if entry["split"] == split]
            splits[split] = dict(
                charts=len(selected), songs=len({entry["song_id"] for entry in selected}),
                groups=len({entry["group_id"] for entry in selected}), rows=sum(entry["rows"] for entry in selected),
                positives=sum(entry["positives"] for entry in selected),
            )
        manifest = dict(
            schema_version=SCHEMA_VERSION, created_at=datetime.now(timezone.utc).isoformat(),
            data_dir=str(root), window_s=window_s, sampling_rate=sampling_rate,
            steps=round(window_s * sampling_rate), features=FEATURES, feature_columns=FEATURE_COLUMNS,
            input_dtype="float32", target="Eligible tap Hit at target time: 0=no tap, 1=one or more taps; holds excluded",
            candidate_times="Union of BPM beat grid rounded to milliseconds and known eligible Hit times",
            history="Tap counts in half-open bins ending before or at the target; target hits excluded",
            limitations=[
                "Candidate times include ground-truth Hit timestamps; this is a conditional classification baseline, not end-to-end chart generation.",
                "Audio descriptors use centered analysis windows and are not strictly causal.",
                "Undefined descriptors remain NaN; impute and normalize using training data only.",
                "Fewer than three independent groups produces a train-only dataset.",
            ],
            split_seed=seed, split_policy="Approximately 80/10/10 independent groups; shared sets or identical audio hashes are kept together",
            charts=len(entries), songs=len({entry["song_id"] for entry in entries}), groups=groups,
            rows=sum(entry["rows"] for entry in entries), positives=sum(entry["positives"] for entry in entries),
            nonfinite_values_by_feature=dict(zip(FEATURE_COLUMNS, missing.tolist())),
            splits=splits, shards=entries, failures=failures,
        )
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        staging.rename(output)
    return manifest


def load_dataset(path):
    """Read and validate a dataset, returning row metadata, X, and binary y."""
    path = Path(path)
    if path.is_dir():
        path /= "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Unsupported hit dataset schema or feature order")
    frames, inputs, targets = [], [], []
    for entry in manifest["shards"]:
        shard = path.parent / entry["file"]
        if shard.parent.resolve() != path.parent.resolve():
            raise ValueError("Dataset shards must be stored beside the manifest")
        if hashlib.sha256(shard.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Dataset shard checksum mismatch: {shard}")
        with np.load(shard, allow_pickle=False) as arrays:
            x, y, times = arrays["x"], arrays["y"], arrays["time_s"]
        if (x.shape != (entry["rows"], manifest["steps"], len(FEATURE_COLUMNS))
                or y.shape != (len(x),) or times.shape != (len(x),)
                or x.dtype != np.float32 or not np.isin(y, [0, 1]).all()):
            raise ValueError(f"Invalid dataset shard arrays: {shard}")
        frames.append(pd.DataFrame(dict(
            song_id=entry["song_id"], chart_id=entry["chart_id"], group_id=entry["group_id"],
            split=entry["split"], time_s=times,
        )))
        inputs.append(x)
        targets.append(y)
    if not inputs or sum(len(x) for x in inputs) != manifest["rows"]:
        raise ValueError("Dataset row count does not match the manifest")
    return pd.concat(frames, ignore_index=True), np.concatenate(inputs), np.concatenate(targets).astype(np.float32)
