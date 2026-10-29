import argparse
import json
from pathlib import Path

import cadencia
import numpy as np
import pandas as pd


def beat_dataset(chart, window_s=2, sampling_rate=50):
    if window_s <= 0 or sampling_rate <= 0:
        raise ValueError("window_s and sampling_rate must be positive")
    samples = round(window_s * sampling_rate)
    if samples < 1:
        raise ValueError("The window must contain at least one sample")
    beats = np.array([beat["time_s"] for beat in chart.beat_grid])
    if not len(beats):
        raise ValueError("The chart has no BPM beat grid")
    hits = np.array([
        event.start_s for event in chart.events
        if isinstance(event, cadencia.osumania.Hit) and event.analysis_eligible
    ])
    beats = np.unique(np.concatenate([np.round(beats, 3), hits]))
    labels = np.isin(beats, hits).astype(int)
    names = [
        "waveform", "onset_strength", "rms_energy", "spectral_centroid",
        "harmonic_energy_fraction", "chroma", "candidate_onsets", "estimated_beats",
    ]
    signals = {
        name: getattr(cadencia, f"get_{name}")(chart, sampling_rate=sampling_rate)
        for name in names
    }
    times = np.arange(len(signals["onset_strength"])) / sampling_rate
    ends = np.searchsorted(times, beats)
    complete = (beats >= window_s) & (ends >= samples)
    windows = ends[complete, None] - samples + np.arange(samples)
    edges = beats[complete, None] - window_s + np.linspace(0, window_s, samples + 1)
    edges[:, -1] = beats[complete]
    hit_history = np.diff(np.searchsorted(hits, edges, side="left"), axis=1)
    return pd.DataFrame({
        "song_id": chart.metadata["beatmapset_id"],
        "chart_id": chart.id,
        "time_s": beats[complete],
        **{name: values[windows].tolist() for name, values in signals.items()},
        "hit_history": hit_history.tolist(),
        "hit": np.eye(2, dtype=int)[labels[complete]].tolist(),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest")
    parser.add_argument("--output", default="data/beat_dataset.json")
    args = parser.parse_args()
    links = [
        "https://osu.ppy.sh/beatmapsets/2043161#mania/4855704",
        "https://osu.ppy.sh/beatmapsets/575053#mania/1241954",
        "https://osu.ppy.sh/beatmapsets/206887#mania/501512",
        "https://osu.ppy.sh/beatmapsets/919633#mania/2069131",
    ]
    if args.manifest:
        links = [song["link"] for song in json.loads(Path(args.manifest).read_text())]
    frames = []
    for link in links:
        print(f"Loading {link}", flush=True)
        frames.append(beat_dataset(cadencia.osumania.load_chart(link)))
    dataset = pd.concat(frames, ignore_index=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_json(args.output, orient="records", indent=2)
    print(dataset)
    return dataset


if __name__ == "__main__":
    main()
