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
