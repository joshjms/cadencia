import copy
import numpy as np
import pytest
from cadencia.beat_alignment import align_chart,load_prior
from cadencia.high_resolution import timing_from_beats
from cadencia.osumania.artifact import validate_chart


def chart(notes):
    return dict(duration_s=4.,notes=notes,timing_map=timing_from_beats(np.arange(0,4,.5)),bpm=120.,difficulty=2.,threshold=0.,candidates=1600)


def note(time,lane,**kw):return dict(time_s=time,lane=lane,kind='tap',probability=.8,**kw)


def test_chords_and_syncopated_positions_shift_together():
    c=chart([note(.512,0),note(.512,1),note(.762,2)])
    original=copy.deepcopy(c)
    r=align_chart(c,onsets=(np.array([.5,.75]),np.ones(2)))
    assert r['notes'][0]['time_s']==r['notes'][1]['time_s']==pytest.approx(.5)
    assert r['notes'][2]['time_s']==pytest.approx(.75)
    assert c==original
    assert r['alignment']['moved_groups']==2
    validate_chart(r,placement=True)


def test_holds_and_release_boundaries_stay_legal():
    hold=dict(time_s=.762,lane=0,kind='hold',end_s=1.262,probability=.8)
    c=chart([hold,note(1.262,0),note(1.27,1)])
    r=align_chart(c,onsets=(np.array([.75,1.25]),np.ones(2)))
    assert len(r['notes'])==3
    assert r['notes'][0]['end_s']-r['notes'][0]['time_s']==pytest.approx(.5)
    assert r['notes'][1]['time_s']>=r['notes'][0]['end_s']
    assert all(abs(a['time_s']-b['time_s'])<=.025+1e-9 for a,b in zip(c['notes'],r['notes']))
    validate_chart(r,placement=True)
    assert align_chart(c,onsets=(np.array([.75,1.25]),np.ones(2)))==r


def test_alignment_is_bounded_and_reports_provenance():
    c=chart([note(.1,0),note(3.99,1)])
    r=align_chart(c,max_shift_ms=10)
    for before,after in zip(c['notes'],r['notes']):
        assert abs(after['time_s']-before['time_s'])<=.010000001
        assert 0<=after['time_s']<4
    assert len(r['alignment']['prior_sha256'])==64
    probabilities,_=load_prior()
    assert probabilities.sum()==pytest.approx(1.)
    with pytest.raises(ValueError):align_chart(c,max_shift_ms=100)
    with pytest.raises(ValueError):align_chart(dict(c,timing_map=[]))
    validate_chart(r,placement=True)
