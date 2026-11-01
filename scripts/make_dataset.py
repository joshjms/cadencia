"""Build a hit-prediction dataset from all existing local osu!mania traces."""

import argparse
import logging

from cadencia.dataset import build_dataset
from cadencia.osumania.loader import data_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="Existing chart data root")
    parser.add_argument("--output", help="New dataset directory (default: <data-dir>/ranked_beat_dataset)")
    parser.add_argument("--window-s", type=float, default=2)
    parser.add_argument("--sampling-rate", type=float, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output = args.output or data_root(args.data_dir) / "ranked_beat_dataset"
    manifest = build_dataset(output, data_dir=args.data_dir, window_s=args.window_s,
                             sampling_rate=args.sampling_rate, seed=args.seed)
    print(f"Saved {manifest['rows']:,} examples from {manifest['charts']} charts to {output}")
    print(f"Splits: {manifest['splits']}")
    return manifest


if __name__ == "__main__":
    main()
