"""Download and preprocess 100 ranked four-key osu!mania charts by default."""

import argparse
import logging

from cadencia.osumania import HinamizawaClient

logger = logging.getLogger(__name__)


def download_ranked_four_key(client, *, count=100, query=""):
    """Process four-key charts, logging and skipping individual chart failures."""
    if count < 1:
        raise ValueError("count must be positive")
    charts, seen, seen_sets = [], set(), set()
    page = 0
    while len(charts) < count:
        result = client.get_ranked_mania({"query": query, "page": page})
        beatmapsets = result["beatmapsets"]
        if not beatmapsets:
            raise RuntimeError(f"Search exhausted after processing {len(charts)} of {count} four-key ranked charts")
        set_ids = {int(beatmapset["id"]) for beatmapset in beatmapsets}
        if set_ids <= seen_sets:
            raise RuntimeError("Hinamizawa pagination repeated previously seen sets without advancing")
        seen_sets.update(set_ids)
        for beatmapset in beatmapsets:
            for beatmap in beatmapset.get("beatmaps", []):
                url = f"chart in beatmapset {beatmapset['id']}"
                try:
                    url = f"https://osu.ppy.sh/beatmapsets/{beatmapset['id']}#mania/{beatmap['id']}"
                    mode = beatmap.get("mode_int", beatmap.get("mode"))
                    status = beatmap.get("ranked", beatmap.get("status", beatmapset.get("status")))
                    if mode not in (3, "mania") or beatmap.get("cs") != 4 or status not in (1, 2, "ranked", "approved"):
                        continue
                    chart_id = int(beatmap["id"])
                    if chart_id in seen:
                        continue
                    seen.add(chart_id)
                    url = f"https://osu.ppy.sh/beatmapsets/{int(beatmapset['id'])}#mania/{chart_id}"
                    charts.extend(client.download_charts({"urls": [url]}))
                except Exception as error:
                    logger.warning("Skipping %s: %s: %s", url, type(error).__name__, error)
                    continue
                logger.info("Ready: %s/%s ranked four-key charts", len(charts), count)
                if len(charts) == count:
                    return charts
        page += 1
    return charts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100, help="Number of individual charts (default: 100)")
    parser.add_argument("--data-dir", help="Data root (defaults to CADENCIA_DATA_DIR or data/)")
    parser.add_argument("--query", default="", help="Optional title, artist, mapper or tag search")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    charts = download_ranked_four_key(HinamizawaClient(data_dir=args.data_dir), count=args.count, query=args.query)
    print(f"Ready: {len(charts)} ranked four-key charts.")
    return charts


if __name__ == "__main__":
    main()
