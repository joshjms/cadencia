import argparse
import json
from pathlib import Path

import cadencia
import numpy as np
import pandas as pd
from cadencia.dataset import FEATURES, chart_samples


def beat_dataset(chart, window_s=2, sampling_rate=50):
    x, labels, times = chart_samples(chart, window_s=window_s, sampling_rate=sampling_rate)
    columns = {}
    offset = 0
    for name in FEATURES:
        width = 12 if name == "chroma" else 1
        values = x[:, :, offset:offset + width]
        columns[name] = (values[:, :, 0] if width == 1 else values).tolist()
        offset += width
    return pd.DataFrame({
        "song_id": chart.metadata["beatmapset_id"],
        "chart_id": chart.id,
        "time_s": times, **columns,
        "hit": np.eye(2, dtype=int)[labels].tolist(),
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
        frames.append(beat_dataset(cadencia.osumania.load_chart(url=link)))
    dataset = pd.concat(frames, ignore_index=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_json(args.output, orient="records", indent=2)
    print(dataset)
    return dataset


if __name__ == "__main__":
    main()
