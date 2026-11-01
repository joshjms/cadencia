import numpy as np
import pytest
from scripts.profile_beat_offsets import timing_observations,summarize


def test_original_timestamps_chords_and_syncopation_are_preserved():
    trace=dict(timing_map=[dict(start_s=0.,beat_position=0.,bpm=120.)],lanes={
        '0':[dict(start_s=.512,lane=0,kind='hit')],
        '1':[dict(start_s=.512,lane=1,kind='hit')],
        '2':[dict(start_s=1/6,lane=2,kind='hit')],
        '3':[dict(start_s=1.,end_s=2.,lane=3,kind='hold')]})
    values=timing_observations(trace)
    np.testing.assert_allclose(values['nearest_beat_offset_ms'][1:3],[12.,12.])
    assert values['subdivision'][0]=='1/3 beat'
    assert values['common_grid_offset_ms'][0]==pytest.approx(0.)
    assert values['onset_group'].sum()==3
    assert (values['kind']=='hold_start').sum()==1
    result=summarize(values)
    assert result['heads']==4
    assert sum(result['phase_within_beat']['probability'])==pytest.approx(1.)
    assert sum(result['nearest_beat_offset_ms']['counts'])==4
