"""Beat-relative chart pattern diagnostics, independent of training objectives.

These measure resemblance to reference charts, not a unique correct mapping.
Empty windows are reported separately and never count as pattern matches.
"""
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PhraseConfig:
    lengths: tuple[int, ...] = (4, 8, 16, 32)
    stride: int = 4
    ticks_per_beat: int = 48
    ngrams: tuple[int, ...] = (2, 4, 8)

    def __post_init__(self):
        values = (*self.lengths, self.stride, self.ticks_per_beat, *self.ngrams)
        if not self.lengths or not self.ngrams or any(type(n) is not int or n < 1 for n in values):
            raise ValueError('Window lengths, stride, resolution and n-gram lengths must be positive integers')
        if len(set(self.lengths)) != len(self.lengths) or len(set(self.ngrams)) != len(self.ngrams):
            raise ValueError('Duplicate window or n-gram lengths')


def load_chart(path):
    """Load a generated .cadencia artifact or original cached trace.json."""
    path = Path(path)
    if path.suffix == '.cadencia':
        from .osumania.artifact import read_artifact
        chart, _ = read_artifact(path)
        return chart
    trace = json.loads(path.read_text())
    if trace.get('game') != 'osumania' or 'lanes' not in trace:
        raise ValueError(f'Expected an osu!mania trace.json or .cadencia: {path}')
    notes = []
    for lane in trace['lanes'].values():
        for event in lane:
            if not event.get('analysis_eligible', True):
                continue
            note = dict(time_s=event['start_s'], lane=event['lane'],
                        kind='hold' if event['kind'] == 'hold' else 'tap')
            if note['kind'] == 'hold':
                note['end_s'] = event['end_s']
            notes.append(note)
    return dict(notes=sorted(notes, key=lambda n:(n['time_s'], n['lane'])),
                duration_s=trace['song']['duration_s'], timing_map=trace['timing_map'])


def beat_at(times, timing):
    """Float64 piecewise conversion; no constant-tempo assumption."""
    if not timing:
        raise ValueError('A reference beat timeline is required')
    starts = np.array([p['start_s'] for p in timing], float)
    origins = np.array([p['beat_position'] for p in timing], float)
    bpms = np.array([p['bpm'] for p in timing], float)
    if (not all(np.isfinite(a).all() for a in (starts, origins, bpms)) or
            (np.diff(starts) <= 0).any() or (np.diff(origins) <= 0).any() or (bpms <= 0).any()):
        raise ValueError('Invalid beat timeline')
    # Discontinuous beat origins could hide timing errors in the evaluation.
    expected = origins[:-1]+np.diff(starts)*bpms[:-1]/60
    if not np.allclose(expected, origins[1:], atol=1e-4, rtol=0):
        raise ValueError('Beat timeline is discontinuous')
    times = np.asarray(times, float)
    if not np.isfinite(times).all():
        raise ValueError('Nonfinite timestamp')
    indices = np.maximum(0, np.searchsorted(starts, times, side='right')-1)
    return origins[indices]+(times-starts[indices])*bpms[indices]/60


def _tokens(chart, config, timing=None):
    duration = chart['duration_s']
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid duration')
    timing = timing or chart.get('timing_map')
    if not timing:
        timing = [dict(start_s=chart.get('beat_offset_s', 0.), beat_position=0., bpm=chart['bpm'])]
    tokens = []
    errors = []
    for note in chart['notes']:
        start, lane = note['time_s'], note['lane']
        hold = note.get('kind') == 'hold'
        end = note['end_s'] if hold else start
        if (type(lane) is not int or lane not in range(4) or not 0 <= start <= end <= duration or
                (hold and end <= start) or note.get('kind', 'tap') not in ('tap','hold')):
            raise ValueError('Invalid note')
        beats = beat_at([start, end], timing)*config.ticks_per_beat
        ticks = np.floor(beats+.5).astype(np.int64)
        tokens.append((int(ticks[0]), lane, int(hold), int(ticks[1])))
        errors.extend((np.abs(beats-ticks)/config.ticks_per_beat).tolist()[:2 if hold else 1])
    extent = beat_at([0., duration], timing)*config.ticks_per_beat
    return sorted(tokens), extent, float(np.mean(errors)) if errors else 0.


def _windows(tokens, extent, length, config):
    size, stride = length*config.ticks_per_beat, config.stride*config.ticks_per_beat
    first = math.ceil(float(extent[0])/stride)*stride
    result = {}
    for start in range(first, math.floor(float(extent[1])-size+1e-6)+1, stride):
        active = []
        for head, lane, hold, release in tokens:
            if start <= head < start+size:
                active.append((head-start, lane, hold, release-head if hold else 0))
            elif hold and head < start < release:
                # Carry-in holds distinguish an occupied lane from an empty bar.
                active.append((-1, lane, 1, min(release-start, size+1)))
        result[start] = tuple(sorted(active))
    return result


def _views(signature):
    heads = [note for note in signature if note[0] >= 0]
    rhythm = tuple(sorted({note[0] for note in heads}))
    # Include carry-in holds even in the lane-invariant views.
    carries = tuple(sorted((note[2], note[3]) for note in signature if note[0] < 0))
    by_time = {}
    for head, _, hold, length in heads:
        by_time.setdefault(head, []).append((hold, length))
    return dict(rhythm=rhythm,
                chord_hold=(tuple((time, tuple(sorted(actions))) for time, actions in sorted(by_time.items())), carries),
                lane_hold=signature)


def profile(chart, config=PhraseConfig(), *, timing=None):
    tokens, extent, error = _tokens(chart, config, timing)
    windows = {str(length):_windows(tokens, extent, length, config) for length in config.lengths}
    patterns = {}
    for length, phrases in windows.items():
        views = {name:Counter() for name in ('rhythm','chord_hold','lane_hold')}
        for signature in phrases.values():
            if signature:
                for name, key in _views(signature).items():
                    views[name][key] += 1
        patterns[length] = views
    # Group simultaneous heads first: a chord is one joint action, not an
    # arbitrary ordering of its individual lane notes.
    groups = {}
    for head, lane, hold, _ in tokens:
        groups.setdefault(head, []).append((lane, hold))
    actions = [tuple(sorted(action)) for _, action in sorted(groups.items())]
    ngrams = {str(n):Counter(tuple(actions[i:i+n]) for i in range(len(actions)-n+1)) for n in config.ngrams}
    return dict(patterns=patterns, windows=windows, ngrams=ngrams,
                summary=dict(notes=len(tokens), holds=sum(t[2] for t in tokens), onset_groups=len(groups),
                             quantization_mean_absolute_beats=error,
                             windows={length:dict(total=len(phrases), active=sum(bool(p) for p in phrases.values()),
                                                  empty_fraction=sum(not p for p in phrases.values())/len(phrases) if phrases else None)
                                      for length, phrases in windows.items()}))


def compare_counts(generated, human):
    """Weighted reference coverage, unique recall and full-pattern distribution.

    A missing human vocabulary is unavailable evidence, not a perfect score.
    """
    g, h = sum(generated.values()), sum(human.values())
    common = set(generated) & set(human)
    result = dict(generated_instances=g, reference_instances=h,
                  generated_unique=len(generated), reference_unique=len(human),
                  matched_unique=len(common))
    if not h:
        return dict(result, reference_match_rate=None, reference_unique_recall=None,
                    js_divergence=None, matched_generated_instances=0)
    matched = sum(generated[key] for key in common)
    if not g:
        js = 1.
    else:
        js = 0.
        for key in set(generated) | set(human):
            p, q = generated[key]/g, human[key]/h
            mean = (p+q)/2
            if p: js += .5*p*math.log2(p/mean)
            if q: js += .5*q*math.log2(q/mean)
    return dict(result, reference_match_rate=matched/g if g else 0.,
                reference_unique_recall=len(common)/len(human),
                js_divergence=float(np.clip(js, 0, 1)), matched_generated_instances=matched)


def nearest_phrase_f1(generated, human, view):
    """Best reference overlap at corresponding beat positions, not bag-of-notes.

    An inverted index avoids comparing every generated/reference window pair.
    Lane/hold tokens include relative onset, lane, type and hold duration.
    """
    if not human:
        return None
    if not generated:
        return 0.
    def features(signature):
        if view == 'chord_hold':
            heads, carries = signature
            return Counter([('head', *head) for head in heads]+[('carry', *carry) for carry in carries])
        return Counter(signature)
    index = {}
    lengths = []
    empty = False
    for i, signature in enumerate(human):
        tokens = features(signature)
        lengths.append(sum(tokens.values()))
        empty |= not tokens
        for token, count in tokens.items():
            index.setdefault(token, []).append((i, count))
    weighted = 0.
    for signature, frequency in generated.items():
        tokens = features(signature)
        size = sum(tokens.values())
        overlaps = Counter()
        for token, count in tokens.items():
            for i, other in index.get(token, []):
                overlaps[i] += min(count, other)
        best = max((2*overlap/(size+lengths[i]) for i, overlap in overlaps.items()), default=float(not size and empty))
        weighted += frequency*best
    return weighted/sum(generated.values())


def evaluate(generated, references, config=PhraseConfig(), *, paired=False):
    if not generated or not references:
        raise ValueError('At least one generated chart and one human reference are required')
    if paired and (len(generated) != 1 or len(references) != 1):
        raise ValueError('Paired evaluation requires exactly one generated chart and one human chart of the same song')
    if paired and abs(generated[0]['duration_s']-references[0]['duration_s']) > .1:
        raise ValueError('Paired charts must use the same recording and time origin (durations differ)')
    # Paired comparisons use the human timeline for BOTH charts, so a wrong
    # generated BPM cannot make its own rhythmic grid look artificially correct.
    timing = references[0].get('timing_map') if paired else None
    if paired and not timing:
        timing = [dict(start_s=references[0].get('beat_offset_s',0.), beat_position=0., bpm=references[0]['bpm'])]
    gen = [profile(chart, config, timing=timing) for chart in generated]
    ref = [profile(chart, config, timing=timing) for chart in references]
    def aggregate(profiles, category, key, view=None):
        result = Counter()
        for item in profiles:
            counter = item[category][key] if view is None else item[category][key][view]
            result.update(counter)
        return result
    patterns = {}
    for length in config.lengths:
        patterns[str(length)] = {}
        for view in ('rhythm','chord_hold','lane_hold'):
            g, h = (aggregate(profiles,'patterns',str(length),view) for profiles in (gen,ref))
            patterns[str(length)][view] = dict(compare_counts(g,h), nearest_reference_f1=nearest_phrase_f1(g,h,view))
    ngrams = {str(n):compare_counts(aggregate(gen,'ngrams',str(n)), aggregate(ref,'ngrams',str(n))) for n in config.ngrams}
    paired_scores = None
    if paired:
        paired_scores = {}
        for length in patterns:
            a, b = gen[0]['windows'][length], ref[0]['windows'][length]
            starts = sorted(set(a) & set(b))
            active = [start for start in starts if a[start] or b[start]]
            paired_scores[length] = dict(compared_active_windows=len(active),
                exact_match_fraction={view:sum(_views(a[s])[view] == _views(b[s])[view] for s in active)/len(active) if active else None
                                      for view in ('rhythm','chord_hold','lane_hold')})
    return dict(schema='cadencia.phrase_evaluation.v1', mode='paired_same_song' if paired else 'reference_corpus_style',
                config=dict(lengths_beats=list(config.lengths),stride_beats=config.stride,ticks_per_beat=config.ticks_per_beat,ngram_lengths=list(config.ngrams)),
                pattern_windows=patterns, joint_action_ngrams=ngrams, paired_windows=paired_scores,
                generated=[p['summary'] for p in gen], references=[p['summary'] for p in ref],
                interpretation=dict(reference_match_rate='Fraction of generated active patterns present anywhere in the human corpus; higher means more reference coverage.',
                                    nearest_reference_f1='Mean best-reference phrase token F1, weighted by generated window frequency. Tokens retain relative beat positions; 1 exact feature match, 0 no overlap. Corpus size affects this score.',
                                    reference_unique_recall='Fraction of unique reference patterns present in generated charts; depends strongly on corpus size.',
                                    js_divergence='Base-2 Jensen-Shannon divergence: 0 identical pattern frequencies, 1 disjoint. No generated patterns with active references scores 1.',
                                    caveats='Diagnostic similarity, not musical correctness or a single quality score. Novel patterns can be valid. Empty windows excluded from pattern matches. Match recording/time origin manually for paired mode. Corpus mode uses each chart\'s own timeline. N-grams ignore timing and hold lengths; phrase views retain them. Only complete beat windows are evaluated; ticks round for evaluation only. Compare similar difficulty bands.'))
