"""Conservative audio repetition matching and beat-relative pattern references.

This is inference-time motif reuse, not a learned chorus classifier. Matching
uses ordered, song-centered MERT bar summaries and detected downbeat boundaries.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RepetitionConfig:
    bars: int = 8
    similarity: float = .70
    variation: float = .2

    def __post_init__(self):
        if type(self.bars) is not int or self.bars < 4:
            raise ValueError('Repeated sections need at least four bars')
        if not np.isfinite(self.similarity) or not 0 < self.similarity <= 1:
            raise ValueError('similarity must be in (0, 1]')
        if not np.isfinite(self.variation) or not 0 <= self.variation <= 1:
            raise ValueError('variation must be in [0, 1]')


def detect_repeats(features, config=RepetitionConfig()):
    """Return disjoint later sections and their earlier audio-matched references.

    No fixed-BPM fallback: without reliable downbeats, leave the chart alone.
    Eight-bar matching makes isolated similar bars insufficient evidence.
    """
    rhythm = features.get('rhythm', {})
    boundaries = np.asarray(rhythm.get('downbeat_times', []), float)
    if not rhythm.get('bar_boundaries_detected', False) or len(boundaries) < 2*config.bars+1:
        return boundaries, []
    frames = np.asarray(features['features'], float)
    times = np.asarray(features['times'], float)
    summaries = []
    valid = []
    for start, end in zip(boundaries, boundaries[1:]):
        rows = frames[(times >= start) & (times < end)]
        valid.append(len(rows) > 0)
        summaries.append(rows.mean(0) if len(rows) else np.zeros(frames.shape[1]))
    summaries = np.asarray(summaries)
    # Remove the common timbre of the song so that shared instrumentation alone
    # is less likely to match unrelated sections.
    centered = summaries - summaries.mean(0)
    norms = np.linalg.norm(centered, axis=1)
    floor = max(1e-8, float(np.median(norms))*.05)
    valid = np.asarray(valid) & (norms > floor)
    centered /= np.maximum(norms[:, None], 1e-8)
    matches = []
    used = set()
    n = config.bars
    target = n
    while target+n <= len(summaries):
        if not valid[target:target+n].all():
            target += 1
            continue
        best = None
        for source in range(target-n+1):
            # Never use an already-rewritten section as a new reference.
            if any(i in used for i in range(source, source+n)) or not valid[source:source+n].all():
                continue
            scores = np.sum(centered[source:source+n]*centered[target:target+n], axis=1)
            score = float(scores.mean())
            if score >= config.similarity and np.quantile(scores, .25) >= config.similarity-.1:
                if best is None or score > best['similarity']:
                    best = dict(source_bar=source, target_bar=target, bars=n, similarity=score)
        if best is None:
            target += 1
        else:
            matches.append(best)
            used.update(range(target, target+n))
            target += n
    return boundaries, matches


def reuse_patterns(chart, features, *, config=RepetitionConfig(), seed=42):
    """Reuse reference taps/chords/holds; selected complete bars keep variation.

    Changes are explicit: this option replaces patterns in matched target bars.
    It is unrelated to difficulty calibration and can increase or decrease note
    counts. Holds crossing section edges cause the entire match to be skipped.
    """
    boundaries, matches = detect_repeats(features, config)
    original = chart['notes']
    result = list(original)
    rng = np.random.default_rng(seed)
    applied = []
    for match in matches:
        source, target, n = (match[k] for k in ('source_bar', 'target_bar', 'bars'))
        s0, s1, t0, t1 = boundaries[[source, source+n, target, target+n]]
        edges = (s0, s1, t0, t1)
        if any(note['time_s'] < edge < note.get('end_s', note['time_s'])
               for edge in edges for note in original):
            continue
        reference = [note for note in original if s0 <= note['time_s'] < s1]
        if not reference:
            continue
        # A single lane permutation per section changes hand placement without
        # changing chord size, onset rhythm or hold duration in musical time.
        lanes = np.arange(4) if config.variation == 0 else rng.permutation(4)
        keep = rng.random(n) < config.variation
        # Do not cut holds at a boundary between reused and retained bars.
        source_edges = boundaries[source:source+n+1]
        target_edges = boundaries[target:target+n+1]
        for j in range(1, n):
            if any(note['time_s'] < edge < note.get('end_s', note['time_s'])
                   for edge in (source_edges[j], target_edges[j]) for note in original):
                keep[j] = keep[j-1]
        mapped = []
        for note in reference:
            bar = min(n-1, int(np.searchsorted(source_edges, note['time_s'], side='right')-1))
            if keep[bar]:
                continue
            copied = dict(note, lane=int(lanes[note['lane']]))
            for key in ('time_s', 'end_s'):
                if key in note:
                    copied[key] = float(np.interp(note[key], source_edges, target_edges))
            mapped.append(copied)
        retained = []
        for note in result:
            if not t0 <= note['time_s'] < t1:
                retained.append(note)
            else:
                bar = min(n-1, int(np.searchsorted(target_edges, note['time_s'], side='right')-1))
                if keep[bar]:
                    retained.append(note)
        candidate = sorted(retained+mapped, key=lambda note:(note['time_s'], note['lane']))
        # Reject instead of silently deleting a conflicting head or shortening a hold.
        ends = [-1.]*4
        legal = True
        for note in candidate:
            if note['time_s'] < ends[note['lane']]:
                legal = False
                break
            ends[note['lane']] = note.get('end_s', note['time_s']+1e-9)
        if not legal:
            continue
        result = candidate
        applied.append(dict(match, retained_variation_bars=np.flatnonzero(keep).tolist(), lane_permutation=lanes.tolist()))
    report = dict(schema='cadencia.repetition.v1', matches=applied, detected_matches=len(matches),
                  variation=config.variation, similarity_threshold=config.similarity,
                  notes_before=len(original), notes_after=len(result))
    return dict(chart, notes=result, repetition=report,
                candidate_policy=chart.get('candidate_policy', '')+'; audio-matched repeated-section motif reuse')
