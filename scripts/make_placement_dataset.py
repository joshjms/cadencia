"""Cache frozen MERT features and build four-lane placement training data."""

import argparse
import json
import hashlib
from importlib.metadata import version
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html.parser import HTMLParser
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from cadencia.mert import FrozenMert, MertConfig
from cadencia.placement import PlacementConfig
from cadencia.osumania.loader import data_root
from cadencia.placement_data import build_placement_dataset


def _valid_rating(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


class _BeatmapsetJSON(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.active = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == 'script' and dict(attrs).get('id') == 'json-beatmapset':
            self.active = True

    def handle_data(self, data):
        if self.active:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == 'script':
            self.active = False


def _fetch_set_ratings(set_id):
    """One bounded metadata request per set, without chart downloads or API auth."""
    url = f'https://osu.ppy.sh/beatmapsets/{set_id}'
    for attempt in range(2):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'cadencia/1.0'})
            with urllib.request.urlopen(request, timeout=12) as response:
                page = response.read(10 * 1024 * 1024).decode('utf-8')
            parser = _BeatmapsetJSON()
            parser.feed(page)
            result = json.loads(''.join(parser.parts))
            if int(result['id']) != int(set_id):
                raise ValueError('Beatmapset metadata identity mismatch')
            return {int(beatmap['id']): float(beatmap['difficulty_rating'])
                    for beatmap in result['beatmaps'] if _valid_rating(beatmap.get('difficulty_rating'))}
        except (OSError, ValueError, KeyError) as error:
            if attempt == 1:
                raise RuntimeError(f'Could not retrieve star ratings from {url}: {error}') from error
            time.sleep(1)


def _calculate_local_rating(chart):
    import rosu_pp_py as rosu

    if hashlib.sha256(chart.osu.read_bytes()).hexdigest() != chart.metadata['chart_sha256']:
        raise ValueError('Cached .osu checksum mismatch')
    beatmap = rosu.Beatmap(path=str(chart.osu))
    if beatmap.mode != rosu.GameMode.Mania or beatmap.is_suspicious():
        raise ValueError('Expected a valid mania beatmap')
    rating = float(rosu.Difficulty(lazer=False, mods=0).calculate(beatmap).stars)
    if not _valid_rating(rating):
        raise ValueError('Calculated star rating is not positive')
    return rating


def fetch_difficulty_labels(root, labels):
    """Fetch direct official metadata concurrently and persist checksum-bound ratings."""
    root = Path(root)
    charts = []
    for trace in (root / 'osumania/charts').glob('*/trace.json'):
        value = json.loads(trace.read_text())
        charts.append(SimpleNamespace(id=value['id'], metadata=value['metadata'], osu=trace.parent / 'chart.osu'))
        del value

    cache_file = root / 'star_rating_cache.json'
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {'schema': 'cadencia.star_ratings.v1', 'ratings': {}}
    if cache.get('schema') != 'cadencia.star_ratings.v1':
        raise ValueError('Unsupported star-rating cache')
    missing = {}
    for chart in charts:
        key = str(chart.id)
        if key in labels:
            if not _valid_rating(labels[key]):
                raise ValueError(f'Invalid explicit star rating for {key}')
            continue
        record = cache['ratings'].get(key, {})
        if (_valid_rating(record.get('rating')) and record.get('chart_sha256') == chart.metadata['chart_sha256']
                and record.get('set_id') == chart.metadata['beatmapset_id']):
            labels[key] = record['rating']
        elif _valid_rating(chart.metadata.get('star_rating')):
            labels[key] = chart.metadata['star_rating']
        else:
            missing.setdefault(chart.metadata['beatmapset_id'], []).append(chart)
    # Complete cached charts can be rated offline. This removes network/rate
    # limits from the normal path and records the precise calculator version.
    local_count = 0
    for set_id, group in list(missing.items()):
        remaining = []
        for chart in group:
            try:
                rating = _calculate_local_rating(chart)
                labels[str(chart.id)] = rating
                cache['ratings'][str(chart.id)] = dict(
                    rating=rating, set_id=set_id, chart_sha256=chart.metadata['chart_sha256'],
                    source=f'rosu-pp-py/{version("rosu-pp-py")} NoMod stable, clock_rate=1',
                    fetched_at=datetime.now(timezone.utc).isoformat())
                local_count += 1
            except (ImportError, OSError, ValueError) as error:
                logging.warning('Offline rating unavailable for %s: %s', chart.id, error)
                remaining.append(chart)
        if remaining:
            missing[set_id] = remaining
        else:
            missing.pop(set_id)
    if local_count:
        temporary = cache_file.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(cache, indent=2))
        temporary.replace(cache_file)
        logging.info('Calculated %s missing ratings locally; %s/%s total labels', local_count, len(labels), len(charts))
    if missing:
        logging.info('Looking up %s chart ratings across %s beatmapsets', sum(map(len, missing.values())), len(missing))
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(_fetch_set_ratings, set_id): set_id for set_id in missing}
        for future in as_completed(pending):
            set_id = pending[future]
            try:
                ratings = future.result()
                for chart in missing[set_id]:
                    rating = ratings.get(chart.id)
                    if not _valid_rating(rating):
                        continue
                    labels[str(chart.id)] = rating
                    cache['ratings'][str(chart.id)] = dict(
                        rating=rating, set_id=set_id, chart_sha256=chart.metadata['chart_sha256'],
                        source=f'https://osu.ppy.sh/beatmapsets/{set_id}',
                        fetched_at=datetime.now(timezone.utc).isoformat())
                temporary = cache_file.with_suffix('.json.tmp')
                temporary.write_text(json.dumps(cache, indent=2))
                temporary.replace(cache_file)
                logging.info('Star ratings ready: %s/%s charts', len(labels), len(charts))
            except Exception as error:
                logging.warning('Skipping ratings for set %s: %s', set_id, error)
    unresolved = [chart.id for chart in charts if str(chart.id) not in labels]
    if unresolved:
        logging.warning('Missing ratings for chart IDs: %s; supply explicit labels', unresolved)
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir')
    parser.add_argument('--output')
    parser.add_argument('--difficulty-labels', help='JSON object mapping chart ID to osu! star rating')
    parser.add_argument('--fetch-difficulties', action='store_true', help='Look up and cache missing ratings from official beatmapset pages')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='MERT extraction device')
    parser.add_argument('--rhythm',action=argparse.BooleanOptionalAction,default=True,help='Pretrained beat/downbeat conditioning for high-resolution datasets')
    parser.add_argument('--high-resolution',action=argparse.BooleanOptionalAction,default=True,help='100 Hz queries and 10ms mel branch; requires MERT-v2')
    parser.add_argument('--encoder',choices=['mert-v1','mert-v2-30s','mert-v2-fullsong'],default='mert-v2-30s')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    labels = json.loads(Path(args.difficulty_labels).read_text()) if args.difficulty_labels else {}
    if not isinstance(labels, dict):
        parser.error('difficulty-labels must be a JSON object')
    if args.fetch_difficulties:
        labels = fetch_difficulty_labels(data_root(args.data_dir), labels)
    mert = MertConfig() if args.encoder=='mert-v1' else MertConfig.v2(full_song=args.encoder=='mert-v2-fullsong')
    config = PlacementConfig(audio_dim=mert.feature_dim,audio_hz=mert.frame_hz,grid_hz=100 if args.high_resolution else 50,use_mel=args.high_resolution,use_rhythm=args.high_resolution and args.rhythm)
    default_name = 'placement_dataset_v1' if args.encoder=='mert-v2-30s' and args.high_resolution else 'placement_dataset_'+args.encoder+('_10ms_mel' if args.high_resolution else '')
    result = build_placement_dataset(args.output or data_root(args.data_dir) / default_name,
                                     data_dir=args.data_dir, difficulty_labels=labels,
                                     extractor=FrozenMert(config=mert,device=args.device),config=config,seed=args.seed)
    print(f"Prepared {len(result['charts'])} charts; skipped {len(result['failures'])}")


if __name__ == '__main__':
    main()
