import copy
import json
import urllib.request
from dataclasses import replace

import numpy as np
import pytest

from cadencia.dataset import chart_samples
from main import beat_dataset


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



@pytest.mark.parametrize("window,rate", [(0, 50), (2, 0), (float("nan"), 50), (0.03, 50)])
def test_invalid_sampling_settings_fail_before_audio_access(window, rate):
    with pytest.raises(ValueError):
        chart_samples(None, window_s=window, sampling_rate=rate)
