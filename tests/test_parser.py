import pytest

from cadencia.osumania import Event, Hit, Hold
from cadencia.osumania.parser import parse_chart
from conftest import beatmap


def test_four_lanes_chords_and_explicit_hold_times():
    result = parse_chart(beatmap("0,192,1000,1,0\n128,192,1000,1,0\n256,192,1500,128,0,2200:0:0:0:0\n512,192,2500,1,0"), 4)
    lanes = result["lanes"]
    assert set(lanes) == {0, 1, 2, 3}
    assert [lanes[i][0].lane for i in range(4)] == [0, 1, 2, 3]
    assert lanes[0][0].start_s == lanes[1][0].start_s == 1
    assert isinstance(lanes[0][0], Hit)
    hold = lanes[2][0]
    assert isinstance(hold, (Hold, Event))
    assert hold.start_s == 1.5
    assert hold.end_s == 2.2
    assert hold.duration_s == pytest.approx(0.7)
    assert result["metadata"]["simultaneous_counts"]["1.0"] == 2
    assert result["metadata"]["coverage_status"] == "complete"
    assert len({event.id for events in lanes.values() for event in events}) == 4


def test_scroll_points_do_not_change_note_time_or_tempo():
    timing = "1000,500,4,2,1,100,1,0\n1500,-25,4,2,1,100,0,0\n2000,250,3,2,1,100,1,8"
    result = parse_chart(beatmap("64,192,2250,1,0", timing=timing), 4)
    assert result["lanes"][0][0].start_s == 2.25
    assert [point["bpm"] for point in result["timing_map"]] == [120, 240]
    assert [point["beat_position"] for point in result["timing_map"]] == [0, 2]
    scroll = next(point for point in result["raw_events"] if point["category"] == "scroll_control")
    assert scroll["scroll_multiplier"] == 4
    grid = {point["time_s"]: point for point in result["beat_grid"]}
    assert grid[0]["beat_position"] == -2
    assert not grid[2]["is_barline"]
    assert grid[2.75]["is_barline"]
    assert grid[2.75]["meter"] == 3


def test_invalid_objects_are_retained_without_fabricating_events():
    result = parse_chart(beatmap("64,192,500,128,0,400:0:0:0:0\n64,192,600,2,0\n64,192,700,129,0,900:0:0:0:0\n64,192,800,1,0\nmalformed"), 4)
    assert len(result["lanes"][0]) == 1
    assert result["lanes"][0][0].start_s == 0.8
    assert len([raw for raw in result["raw_events"] if raw["category"] == "unsupported_object"]) == 4
    assert len(result["diagnostics"]) == 4
    assert result["metadata"]["coverage_status"] == "partial"


def test_outside_audio_event_keeps_original_timestamp():
    result = parse_chart(beatmap("64,192,5000,1,0"), 4)
    event = result["lanes"][0][0]
    assert event.start_s == 5
    assert not event.analysis_eligible
    assert result["diagnostics"][0]["code"] == "outside_audio"


@pytest.mark.parametrize("kwargs", [{"keys": 7}, {"mode": 0}])
def test_rejects_other_modes_and_key_counts(kwargs):
    with pytest.raises(ValueError):
        parse_chart(beatmap("64,192,1000,1,0", **kwargs), 4)


def test_event_invariants():
    arguments = dict(start_s=1, lane=0, source_line=1, source_type=128, source_x=64, id="test")
    with pytest.raises(ValueError, match="after"):
        Hold(**arguments, end_s=1)
    with pytest.raises(ValueError, match="lanes"):
        Hit(**{**arguments, "lane": 4})
