import numpy as np
import pytest

from cadencia.repetition import RepetitionConfig, detect_repeats, reuse_patterns
from cadencia.osumania.artifact import validate_chart


def fixture():
    rng = np.random.default_rng(123)
    a, b = rng.normal(size=(2, 8, 64))
    frames = np.concatenate([a, b, a])
    boundaries = np.arange(25, dtype=float)*2
    features = dict(features=frames, times=boundaries[:-1]+.5,
                    rhythm=dict(downbeat_times=boundaries, bar_boundaries_detected=True))
    notes = []
    for bar in range(24):
        notes.append(dict(time_s=bar*2+.25, lane=bar%4, probability=.8, kind='hold', end_s=bar*2+.75))
        notes.append(dict(time_s=bar*2+(1 if bar<16 else 1.5), lane=(bar+1)%4, probability=.7, kind='tap'))
    chart = dict(notes=notes, duration_s=49., threshold=0., candidates=1000, difficulty=2.)
    return chart, features


def test_detects_ordered_repeated_phrase_not_unrelated_middle():
    _, features = fixture()
    _, matches = detect_repeats(features)
    assert matches == [dict(source_bar=0, target_bar=16, bars=8, similarity=pytest.approx(1.))]
    # Same individual bar features in a different order are not a phrase match.
    features['features'][16:] = features['features'][:8][::-1]
    assert detect_repeats(features)[1] == []


def test_taps_and_holds_follow_target_bar_timeline():
    chart, features = fixture()
    # A slower repeat: map onset and release through corresponding bar boundaries.
    features['rhythm']['downbeat_times'][17:] += np.arange(1, 9)*.2
    features['times'][16:] = features['rhythm']['downbeat_times'][16:24]+.5
    chart['duration_s'] = 52.
    result = reuse_patterns(chart, features, config=RepetitionConfig(variation=0))
    assert result['repetition']['matches']
    assert result['notes'][:32] == chart['notes'][:32]
    target = [n for n in result['notes'] if n['time_s'] >= 32]
    assert target[0]['time_s'] == pytest.approx(32+.25*1.1)
    assert target[0]['end_s'] == pytest.approx(32+.75*1.1)
    assert target[1]['time_s'] == pytest.approx(33.1)
    assert validate_chart(result, placement=True)['repetition'] == result['repetition']
    assert chart['notes'][33]['time_s'] == 33.5  # source chart was not mutated


def test_variation_is_seeded_and_all_variation_retains_original():
    chart, features = fixture()
    first = reuse_patterns(chart, features, seed=17)
    assert first == reuse_patterns(chart, features, seed=17)
    assert first['repetition']['matches'][0]['retained_variation_bars']
    assert reuse_patterns(chart, features, config=RepetitionConfig(variation=1))['notes'] == chart['notes']
    validate_chart(first, placement=True)


def test_missing_downbeats_static_audio_and_crossing_holds_are_safe():
    chart, features = fixture()
    features['rhythm']['bar_boundaries_detected'] = False
    assert reuse_patterns(chart, features)['notes'] == chart['notes']
    features['rhythm']['bar_boundaries_detected'] = True
    features['features'][:] = 1
    assert detect_repeats(features)[1] == []
    chart, features = fixture()
    chart['notes'][31].update(kind='hold', end_s=32.5)
    assert reuse_patterns(chart, features)['repetition']['matches'] == []


@pytest.mark.parametrize('kwargs', [dict(bars=2),dict(variation=-.1),dict(similarity=float('nan'))])
def test_config_validation(kwargs):
    with pytest.raises(ValueError): RepetitionConfig(**kwargs)
