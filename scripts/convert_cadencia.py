"""Convert a portable .cadencia chart into an importable osu!mania .osz archive."""
import argparse
import math
from pathlib import Path
import tempfile
import zipfile

from cadencia.osumania.artifact import read_artifact


def osu_text(chart, *, title=None, artist=None, creator=None, version=None, constant_scroll=False):
    """Serialize exactly the chart that the difficulty calculator will evaluate."""
    def clean(value):
        text = str(value)
        if any(c in text for c in '\r\n\x00'):
            raise ValueError('Metadata must be a single line')
        return text
    title = clean(title or chart.get('title') or 'Generated chart')
    release_tag = None
    from cadencia.model_versions import ROOT
    import json
    for manifest_path in sorted((ROOT/'model_releases').glob('*.json')):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('artifacts',{}).get('decoder',{}).get('sha256') and manifest['artifacts']['decoder']['sha256'] == chart.get('model_sha256'):
            release_tag = manifest['version']
            break
    creator = creator or chart.get('author') or (f'Cadencia {release_tag}' if release_tag else 'Cadencia')
    artist, creator = clean(artist or chart.get('artist') or 'Unknown artist'), clean(creator)
    tags = 'cadencia generated 4k' + (f' {clean(release_tag)}' if release_tag else '')
    version = clean(version or chart.get('version') or f"4K Generated ({chart.get('difficulty', 'unknown')} star target)")
    bpm = chart.get('bpm')
    if bpm is None or not math.isfinite(bpm) or bpm <= 0:
        raise ValueError('Artifact needs a positive BPM for osu! export')
    offset = chart.get('beat_offset_s', 0.) * 1000
    timing = [dict(start_s=0., bpm=bpm)] if constant_scroll else chart.get('timing_map',[dict(start_s=offset/1000,bpm=bpm)])
    timing_points = '\n'.join(f"{p['start_s']*1000:.6f},{60000/p['bpm']:.9f},4,1,0,100,1,0" for p in timing)
    objects = []
    occupied_until = [-1] * 4
    for note in chart['notes']:
        lane = note['lane']
        start = round(note['time_s'] * 1000)
        if start < occupied_until[lane]:
            raise ValueError('Millisecond rounding creates overlapping notes')
        x = 64 + lane * 128
        if note.get('kind') == 'hold':
            end = round(note['end_s'] * 1000)
            if end <= start:
                raise ValueError('Hold is too short for millisecond precision')
            objects.append(f'{x},192,{start},128,0,{end}:0:0:0:0:')
            occupied_until[lane] = end
        else:
            objects.append(f'{x},192,{start},1,0,0:0:0:0:')
            occupied_until[lane] = start + 1
    osu = f'''osu file format v14

[General]
AudioFilename: audio.wav
AudioLeadIn: 0
PreviewTime: -1
Countdown: 0
SampleSet: Normal
Mode: 3
LetterboxInBreaks: 0
SpecialStyle: 0

[Editor]
DistanceSpacing: 1
BeatDivisor: 4
GridSize: 4
TimelineZoom: 1

[Metadata]
Title:{title}
TitleUnicode:{title}
Artist:{artist}
ArtistUnicode:{artist}
Creator:{creator}
Version:{version}
Source:{clean(chart.get('source_url', ''))}
Tags:{tags}
BeatmapID:0
BeatmapSetID:-1

[Difficulty]
HPDrainRate:5
CircleSize:4
OverallDifficulty:5
ApproachRate:5
SliderMultiplier:1.4
SliderTickRate:1

[Events]

[TimingPoints]
{timing_points}

[HitObjects]
''' + '\n'.join(objects) + '\n'
    return osu


def calculated_stars(chart):
    """Rate the serialized millisecond timings using rosu-pp's stable no-mod calculator."""
    import rosu_pp_py as rosu
    beatmap = rosu.Beatmap(content=osu_text(chart))
    return rosu.Difficulty(lazer=False, mods=0).calculate(beatmap).stars


def calibrate_chart(chart, target, *, tolerance=.03):
    """Search deterministic subsets; never change audio, tempo, lanes or hold lengths.

    Keep the original first/last heads. Favor musical accents, holds and model
    confidence, reducing secondary chord notes first. Evaluate actual ratings;
    subset difficulty is not assumed to be monotonic. Fail if no candidate meets
    tolerance, rather than publishing a mislabeled chart.
    """
    if not math.isfinite(target) or target <= 0 or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Star target and tolerance must be positive and finite')
    original = calculated_stars(chart)
    if abs(original-target) <= tolerance:
        return dict(chart), original
    if original < target:
        raise ValueError('This calibrator simplifies charts; provide a harder source chart')
    notes = chart['notes']
    if len(notes) < 3:
        raise ValueError('Too few notes to calibrate')
    bpm, offset = chart['bpm'], chart.get('beat_offset_s',0.)
    chord_rank = {}
    def importance(index):
        note = notes[index]
        beat = (note['time_s']-offset)*bpm/60
        # Accent hierarchy: bar, beat, eighth, then remaining subdivisions.
        accent = max((weight if abs(beat/step-round(beat/step))*step < .055 else 0)
                     for step,weight in [(4,4),(1,3),(.5,2),(.25,1)])
        rank = chord_rank.get(note['time_s'],0)
        chord_rank[note['time_s']] = rank+1
        return accent + (1.5 if note.get('kind')=='hold' else 0) + note['probability']*.5 - rank*.75
    scores = {i:importance(i) for i in range(len(notes))}
    order = [0,len(notes)-1] + sorted(range(1,len(notes)-1),key=lambda i:(-scores[i],i))
    measured = {}
    def evaluate(count):
        if count not in measured:
            selected = set(order[:count])
            candidate = dict(chart,notes=[n for i,n in enumerate(notes) if i in selected])
            measured[count] = (calculated_stars(candidate),candidate)
        return measured[count]
    # Scan across the full range, then inspect the neighborhood of the closest
    # measured candidate. This avoids relying on star monotonicity after deletion.
    counts = {2,len(notes)} | {round(2+(len(notes)-2)*i/64) for i in range(65)}
    for count in sorted(counts): evaluate(count)
    best = min(measured,key=lambda k:abs(measured[k][0]-target))
    radius = max(8,math.ceil(len(notes)/64))
    for count in range(max(2,best-radius),min(len(notes),best+radius)+1): evaluate(count)
    best = min(measured,key=lambda k:(abs(measured[k][0]-target),-k))
    rating,result = measured[best]
    if abs(rating-target) > tolerance:
        raise ValueError(f'Closest candidate is {rating:.4f} stars; requested {target} ± {tolerance}')
    result['candidate_policy'] = chart.get('candidate_policy','') + f'; deterministic note simplification calibrated by rosu-pp stable no-mod stars: {rating:.4f} (target {target})'
    return result,rating


def convert(source, out, *, title=None, artist=None, creator=None, version=None, target_stars=None, tolerance=.03, chart_out=None, constant_scroll=False):
    """Export, optionally simplifying notes using measured no-mod stable star feedback."""
    source, out = Path(source), Path(out)
    if out.suffix.lower() != '.osz':
        raise ValueError('Output must end in .osz')
    if out.exists():
        raise FileExistsError(out)
    if chart_out and Path(chart_out).exists():
        raise FileExistsError(chart_out)
    chart, audio = read_artifact(source)
    if target_stars is not None:
        chart, rating = calibrate_chart(chart, target_stars, tolerance=tolerance)
        version = version or f'4K Generated ({rating:.2f} stars)'
        print(f'Calibrated to {rating:.4f} stars; {len(chart["notes"])} notes')
    osu = osu_text(chart,title=title,artist=artist,creator=creator,version=version,constant_scroll=constant_scroll)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a completed archive, refusing to overwrite an existing export.
    import os
    with tempfile.NamedTemporaryFile(dir=out.parent, suffix='.osz') as temporary:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('chart.osu', osu.encode('utf-8'))
            archive.writestr('audio.wav', audio)
        os.link(temporary.name, out)
    if chart_out:
        from cadencia.osumania.artifact import write_artifact
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / 'audio.wav'
            wav.write_bytes(audio)
            write_artifact(chart_out, chart, wav)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Input .cadencia artifact')
    parser.add_argument('--out', type=Path, help='Output .osz (defaults beside input)')
    parser.add_argument('--title')
    parser.add_argument('--artist')
    parser.add_argument('--creator', help='Defaults to Cadencia plus the recognized model release')
    parser.add_argument('--version', help='Difficulty name, not a calculated star rating')
    parser.add_argument('--constant-scroll', action=argparse.BooleanOptionalAction, default=False, help='Use one nominal BPM for constant scrolling; preserves all hit objects')
    parser.add_argument('--target-stars', type=float, help='Measured no-mod stable star target; simplify notes to meet it')
    parser.add_argument('--tolerance', type=float, default=.03, help='Allowed absolute star error (default .03)')
    parser.add_argument('--chart-out', type=Path, help='Also save the exported chart as a .cadencia artifact')
    args = parser.parse_args()
    path = convert(args.source, args.out or args.source.with_suffix('.osz'), title=args.title,
                   artist=args.artist, creator=args.creator, version=args.version,
                   target_stars=args.target_stars, tolerance=args.tolerance, chart_out=args.chart_out,constant_scroll=args.constant_scroll)
    print(f'Saved {path}')


if __name__ == '__main__':
    main()
