"""Compare generated charts with human phrase patterns, chords and holds."""
import argparse
import hashlib
import json
from pathlib import Path

from cadencia.phrase_evaluation import PhraseConfig, evaluate, load_chart


def reference_paths(dataset, charts_dir, split, min_stars=None, max_stars=None):
    """Use original traces, not the quantized training labels or model outputs."""
    path = Path(dataset)/'manifest.json'
    manifest = json.loads(path.read_text())
    selected = [entry for entry in manifest['charts'] if entry['split'] == split
                and (min_stars is None or entry['difficulty'] >= min_stars)
                and (max_stars is None or entry['difficulty'] < max_stars)]
    if not selected:
        raise ValueError('No reference charts in the requested split/difficulty range')
    paths = []
    for entry in selected:
        trace = Path(charts_dir)/str(entry['chart_id'])/'trace.json'
        if hashlib.sha256(trace.read_bytes()).hexdigest() != entry['trace_sha256']:
            raise ValueError(f'Reference trace checksum mismatch: {trace}')
        paths.append(trace)
    return paths, hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generated', type=Path, nargs='+', required=True, help='Generated .cadencia charts')
    refs = parser.add_mutually_exclusive_group(required=True)
    refs.add_argument('--references', type=Path, nargs='+', help='Human trace.json or .cadencia files')
    refs.add_argument('--dataset', type=Path, help='Placement dataset manifest directory; selects original human traces')
    parser.add_argument('--charts-dir', type=Path, default=Path('data/osumania/charts'))
    parser.add_argument('--split', choices=['train','validation','test'], default='test')
    parser.add_argument('--min-stars', type=float)
    parser.add_argument('--max-stars', type=float, help='Exclusive upper limit on reference difficulties')
    parser.add_argument('--paired', action='store_true', help='Exactly one human/generated pair from the same recording and time origin')
    parser.add_argument('--lengths', type=int, nargs='+', default=[4,8,16,32], help='Phrase lengths in quarter-note beats')
    parser.add_argument('--stride', type=int, default=4, help='Window stride in quarter-note beats')
    parser.add_argument('--ticks-per-beat', type=int, default=48, help='Evaluation-only quantization, not a generation grid')
    parser.add_argument('--out', type=Path, default=Path('data/analysis/phrase_evaluation/report.json'))
    args = parser.parse_args()
    try:
        config = PhraseConfig(lengths=tuple(args.lengths), stride=args.stride, ticks_per_beat=args.ticks_per_beat)
        if not args.dataset and (args.min_stars is not None or args.max_stars is not None):
            raise ValueError('Difficulty filtering requires --dataset')
        if args.min_stars is not None and args.max_stars is not None and args.min_stars >= args.max_stars:
            raise ValueError('--min-stars must be below --max-stars')
        paths, digest = reference_paths(args.dataset, args.charts_dir, args.split, args.min_stars, args.max_stars) if args.dataset else (args.references, None)
        generated = [load_chart(path) for path in args.generated]
        references = [load_chart(path) for path in paths]
        report = evaluate(generated, references, config, paired=args.paired)
        def identity(path):
            return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        report['inputs'] = dict(generated=[identity(p) for p in args.generated], references=[identity(p) for p in paths],
                                dataset_sha256=digest, reference_split=args.split if args.dataset else None,
                                min_stars=args.min_stars, max_stars=args.max_stars)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(args.out)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'{error}\n')
    print(f'Compared {len(generated)} generated chart(s) with {len(references)} human chart(s).')
    print('Length (beats) | Rhythm match | Chord/hold match | Lane/hold match')
    for length, views in report['pattern_windows'].items():
        values = [views[name]['reference_match_rate'] for name in ('rhythm','chord_hold','lane_hold')]
        print(f'{length:>14} | '+' | '.join(f'{v:>11.1%}' if v is not None else '        N/A' for v in values))
    print('Nearest-reference phrase F1 (partial overlap):')
    for length, views in report['pattern_windows'].items():
        values = [views[name]['nearest_reference_f1'] for name in ('rhythm','chord_hold','lane_hold')]
        print(f'{length:>14} | '+' | '.join(f'{v:>11.1%}' if v is not None else '        N/A' for v in values))
    print('Matches measure reference resemblance, not correctness. See JSON for empty windows, diversity, distribution divergence and paired results.')
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
