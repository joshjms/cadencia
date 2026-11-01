import copy
import json
import urllib.request
from dataclasses import replace

import numpy as np
import pytest

from cadencia.dataset import FEATURE_COLUMNS, _assign_splits, build_dataset, chart_samples, load_dataset
from scripts.main import beat_dataset


def test_windows_labels_history_and_legacy_dataframe_agree(downloaded_chart):
    chart, _, _ = downloaded_chart
    x, y, times = chart_samples(chart)
    assert x.shape == (4, 100, 20)
    assert x.dtype == np.float32
    np.testing.assert_array_equal(times, [2, 2.5, 3, 3.5])
    np.testing.assert_array_equal(y, [1, 0, 0, 0])
    # Simultaneous prior taps count twice; the target tap at 2s is excluded.
    assert x[0, :, -1].sum() == 2
    assert x[1, :, -1].sum() == 3
    np.testing.assert_allclose(x[0, :, 1], chart.onset_strength.values[:100], rtol=1e-6)
    frame = beat_dataset(chart)
    np.testing.assert_array_equal(frame.time_s, times)
    np.testing.assert_array_equal(np.asarray(frame.hit.tolist())[:, 1], y)
    np.testing.assert_array_equal(np.asarray(frame.hit_history.tolist()), x[:, :, -1])
    np.testing.assert_array_equal(np.asarray(frame.chroma.tolist()), x[:, :, 5:17])


def test_off_grid_taps_are_candidates_but_hold_starts_are_not_positive(downloaded_chart):
    chart, _, _ = downloaded_chart
    lanes = dict(chart.lanes)
    lanes[0] = [replace(lanes[0][0], start_s=2.123)]
    lanes[2] = [replace(lanes[2][0], start_s=2.5, end_s=3)]
    modified = replace(chart, lanes=lanes)
    _, y, times = chart_samples(modified)
    assert y[times == 2.123].tolist() == [1]
    assert y[times == 2.5].tolist() == [0]


def test_splits_keep_sets_and_shared_audio_together_transitively():
    entries = [dict(chart_id=i + 1, song_id=song, audio_sha256=audio)
               for i, (song, audio) in enumerate([(10, "a"), (10, "b"), (20, "b"), (30, "c"), (40, "d"), (50, "e")])]
    again = copy.deepcopy(entries)
    assert _assign_splits(entries, 42) == 4
    _assign_splits(again, 42)
    assert entries == again
    assert len({entry["group_id"] for entry in entries[:3]}) == 1
    assert len({entry["split"] for entry in entries[:3]}) == 1
    assert {entry["split"] for entry in entries} == {"train", "validation", "test"}
    for field in ["song_id", "audio_sha256", "group_id"]:
        assignments = {}
        for entry in entries:
            assignments.setdefault(entry[field], set()).add(entry["split"])
        assert all(len(splits) == 1 for splits in assignments.values())


def test_offline_build_load_failure_reporting_and_integrity(downloaded_chart, tmp_path, monkeypatch):
    chart, root, _ = downloaded_chart
    bad = root / "osumania/charts/999999"
    bad.mkdir()
    (bad / "trace.json").write_text("invalid JSON")
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("Dataset attempted network access"))
    output = tmp_path / "dataset"
    manifest = build_dataset(output, data_dir=root)
    assert manifest["charts"] == 1
    assert manifest["rows"] == 4
    assert manifest["positives"] == 1
    assert manifest["groups"] == 1
    assert manifest["splits"]["train"]["charts"] == 1
    assert manifest["feature_columns"] == FEATURE_COLUMNS
    assert manifest["failures"][0]["error_type"] == "JSONDecodeError"
    assert (output / "manifest.json").is_file()
    frame, x, y = load_dataset(output)
    expected_x, expected_y, expected_times = chart_samples(chart)
    np.testing.assert_array_equal(x, expected_x)
    np.testing.assert_array_equal(y, expected_y)
    np.testing.assert_array_equal(frame.time_s, expected_times)
    assert frame.chart_id.unique().tolist() == [chart.id]
    assert frame.split.unique().tolist() == ["train"]
    with pytest.raises(FileExistsError):
        build_dataset(output, data_dir=root)
    shard = output / manifest["shards"][0]["file"]
    shard.write_bytes(shard.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(output)


def test_no_cached_charts_do_not_create_dataset(tmp_path):
    output = tmp_path / "dataset"
    with pytest.raises(FileNotFoundError, match="No preprocessed"):
        build_dataset(output, data_dir=tmp_path / "missing")
    assert not output.exists()


@pytest.mark.parametrize("window,rate", [(0, 50), (2, 0), (float("nan"), 50), (0.03, 50)])
def test_invalid_sampling_settings_fail_before_audio_access(window, rate):
    with pytest.raises(ValueError):
        chart_samples(None, window_s=window, sampling_rate=rate)
