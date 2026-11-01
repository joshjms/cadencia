import copy
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from cadencia.phrase_evaluation import PhraseConfig, beat_at, evaluate, profile
from cadencia.osumania.artifact import write_artifact
from scripts.evaluate_phrases import reference_paths


def chart():
    notes = []
    for bar in range(10):
        for beat, lane in [(0,0),(1,2),(1.5,1),(3,3)]:
            note = dict(time_s=(bar*4+beat)/2, lane=lane, kind='tap', probability=.9)
            if beat == 1:
                note.update(kind='hold', end_s=(bar*4+2.5)/2)
            notes.append(note)
        notes.append(dict(time_s=(bar*4+3)/2, lane=0, kind='tap', probability=.9))
    return dict(notes=sorted(notes,key=lambda n:(n['time_s'],n['lane'])),
                duration_s=20., timing_map=[dict(start_s=0.,beat_position=0.,bpm=120.)],
                bpm=120., threshold=0., candidates=8000, difficulty=2.)


def test_identical_chart_matches_all_joint_patterns_and_phrases():
    value = chart()
    report = evaluate([value],[value],paired=True)
    for views in report['pattern_windows'].values():
        for metric in views.values():
            assert metric['reference_match_rate'] == 1
            assert metric['reference_unique_recall'] == 1
            assert metric['js_divergence'] == 0
            assert metric['nearest_reference_f1'] == 1
    assert all(v['exact_match_fraction']['lane_hold'] == 1 for v in report['paired_windows'].values())
    assert all(v['reference_match_rate'] == 1 for v in report['joint_action_ngrams'].values())
    json.dumps(report,allow_nan=False)


def test_lane_reordering_preserves_rhythm_but_changes_pattern_scores():
    human = chart(); generated = copy.deepcopy(human)
    rng = np.random.default_rng(44)
    for note in generated['notes']: note['lane'] = int(rng.integers(4))
    report = evaluate([generated],[human])['pattern_windows']['4']
    assert report['rhythm']['reference_match_rate'] == 1
    assert report['chord_hold']['reference_match_rate'] == 1
    assert report['lane_hold']['reference_match_rate'] < 1


def test_same_per_lane_counts_but_shuffled_sequence_is_penalized():
    human = dict(chart(), notes=[dict(time_s=i*.5, lane=i%4, kind='tap') for i in range(40)])
    generated = copy.deepcopy(human)
    lanes = [n['lane'] for n in generated['notes']]
    np.random.default_rng(3).shuffle(lanes)
    for note, lane in zip(generated['notes'], lanes): note['lane'] = lane
    assert sorted(lanes) == sorted(n['lane'] for n in human['notes'])
    report = evaluate([generated],[human])
    assert report['pattern_windows']['8']['rhythm']['reference_match_rate'] == 1
    assert report['pattern_windows']['8']['lane_hold']['nearest_reference_f1'] < 1
    assert report['joint_action_ngrams']['4']['reference_match_rate'] < 1


def test_repeating_one_known_pattern_does_not_match_human_distribution():
    from collections import Counter
    from cadencia.phrase_evaluation import compare_counts
    result = compare_counts(Counter(a=100), Counter(a=50,b=50))
    assert result['reference_match_rate'] == 1
    assert result['reference_unique_recall'] == .5
    assert result['js_divergence'] > 0


def test_hold_duration_and_chord_structure_are_measured():
    human = chart(); generated = copy.deepcopy(human)
    for note in generated['notes']:
        if note['kind'] == 'hold': note['end_s'] -= .25
    views = evaluate([generated],[human])['pattern_windows']['4']
    assert views['rhythm']['reference_match_rate'] == 1
    assert views['chord_hold']['reference_match_rate'] == 0
    assert views['lane_hold']['reference_match_rate'] == 0
    assert 0 < views['lane_hold']['nearest_reference_f1'] < 1
    generated['notes'] = [n for n in human['notes'] if n['lane'] != 0 or (n['time_s']*2)%4 != 3]
    assert evaluate([generated],[human])['pattern_windows']['4']['chord_hold']['reference_match_rate'] == 0


def test_silence_cannot_inflate_match_rates_and_short_charts_are_unavailable():
    human = chart(); empty = dict(human,notes=[])
    report = evaluate([empty],[human])
    for metric in report['pattern_windows']['4'].values():
        assert metric['reference_match_rate'] == 0 and metric['js_divergence'] == 1
    assert report['generated'][0]['windows']['4']['empty_fraction'] == 1
    assert evaluate([empty],[empty])['pattern_windows']['4']['rhythm']['reference_match_rate'] is None
    short = dict(empty,duration_s=1.)
    assert evaluate([short],[short])['pattern_windows']['4']['lane_hold']['js_divergence'] is None


def test_cross_boundary_holds_count_as_active_and_preserve_release():
    value = dict(chart(),notes=[dict(time_s=1.,lane=0,kind='hold',end_s=5.)])
    phrases = profile(value)['windows']['4']
    assert phrases[4*48] == ((-1,0,1,4*48+1),)
    assert phrases[8*48] == ((-1,0,1,2*48),)
    assert profile(value)['summary']['windows']['4']['active'] == 3


def test_same_song_uses_human_clock_and_preserves_absolute_phase():
    human = chart(); generated = copy.deepcopy(human)
    generated['timing_map'] = [dict(start_s=0.,beat_position=0.,bpm=240.)]
    report = evaluate([generated],[human],paired=True)
    assert report['paired_windows']['4']['exact_match_fraction']['lane_hold'] == 1
    for note in generated['notes']:
        note['time_s'] += .1
        if 'end_s' in note: note['end_s'] += .1
    report = evaluate([generated],[human],paired=True)
    assert report['paired_windows']['4']['exact_match_fraction']['rhythm'] == 0


def test_piecewise_tempo_and_triplets():
    timing = [dict(start_s=0.,beat_position=0.,bpm=120.),dict(start_s=2.,beat_position=4.,bpm=60.)]
    np.testing.assert_allclose(beat_at([1.,2.,3.],timing),[2,4,5])
    value = dict(chart(), notes=[dict(time_s=1/6,lane=0,kind='tap')], timing_map=timing)
    tokens = profile(value)['windows']['4'][0]
    assert tokens[0][0] == 16
    with pytest.raises(ValueError,match='discontinuous'):
        beat_at([1.], [timing[0],dict(timing[1],beat_position=7.)])


def test_cli_and_original_trace_loading_with_source_integrity(tmp_path):
    value = chart();wav=tmp_path/'song.wav';sf.write(wav,np.zeros(20*8000),8000)
    generated=tmp_path/'gen.cadencia';write_artifact(generated,value,wav)
    trace=dict(game='osumania',song=dict(duration_s=20.),timing_map=value['timing_map'],lanes={str(lane):[] for lane in range(4)})
    for note in value['notes']:
        trace['lanes'][str(note['lane'])].append(dict(note,start_s=note['time_s'],analysis_eligible=True))
    directory=tmp_path/'charts/1';directory.mkdir(parents=True);reference=directory/'trace.json';reference.write_text(json.dumps(trace))
    dataset=tmp_path/'dataset';dataset.mkdir()
    entry=dict(chart_id=1,split='test',difficulty=2.,trace_sha256=hashlib.sha256(reference.read_bytes()).hexdigest())
    (dataset/'manifest.json').write_text(json.dumps(dict(charts=[entry])))
    out=tmp_path/'report.json'
    result=subprocess.run([sys.executable,'scripts/evaluate_phrases.py','--generated',str(generated),
                           '--dataset',str(dataset),'--charts-dir',str(tmp_path/'charts'),'--paired','--out',str(out)],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    report=json.loads(out.read_text())
    assert report['pattern_windows']['8']['lane_hold']['reference_match_rate'] == 1
    assert report['inputs']['reference_split'] == 'test'
    reference.write_text(reference.read_text()+' ')
    with pytest.raises(ValueError,match='checksum'):
        reference_paths(dataset,tmp_path/'charts','test')


@pytest.mark.parametrize('kwargs',[dict(lengths=()),dict(stride=0),dict(ticks_per_beat=-1),dict(lengths=(4,4))])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError): PhraseConfig(**kwargs)
