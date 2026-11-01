"""Measured difficulty targeting must preserve audio timing and valid hold intervals."""
import pytest
pytest.importorskip('rosu_pp_py')
from scripts.convert_cadencia import calibrate_chart, calculated_stars, osu_text


def chart():
    notes=[]
    for i in range(500):
        n=dict(time_s=i*.12,lane=i%4,kind='tap',probability=.8)
        if i%25==0:
            n.update(kind='hold',end_s=i*.12+.24)
        notes.append(n)
    return dict(notes=notes,bpm=125.,beat_offset_s=0.,difficulty=1.5,duration_s=61.,title='Fixture')


def test_measured_target_keeps_original_notes_and_endpoints():
    original=chart()
    before=[dict(n) for n in original['notes']]
    target=calculated_stars(original)*.75
    result,stars=calibrate_chart(original,target,tolerance=.04)
    assert abs(stars-target)<=.04
    assert calculated_stars(result)==pytest.approx(stars)
    assert len(result['notes'])<len(before)
    assert result['notes'][0]==before[0] and result['notes'][-1]==before[-1]
    assert all(n in before for n in result['notes'])
    assert original['notes']==before
    text=osu_text(result)
    assert 'Mode: 3' in text and 'CircleSize:4' in text
    assert len(text.split('[HitObjects]\n')[1].splitlines())==len(result['notes'])


def test_calibration_rejects_invalid_and_harder_targets():
    value=chart();rating=calculated_stars(value)
    with pytest.raises(ValueError):calibrate_chart(value,float('nan'))
    with pytest.raises(ValueError):calibrate_chart(value,1.5,tolerance=0)
    with pytest.raises(ValueError,match='harder source'):calibrate_chart(value,rating+1)
    result,actual=calibrate_chart(value,rating)
    assert result['notes']==value['notes'] and actual==rating
