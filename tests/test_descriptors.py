from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

import cadencia

CONTINUOUS = [
    "onset_strength", "rms_energy", "spectral_centroid",
    "harmonic_energy_fraction", "chroma",
]
GETTERS = [
    getattr(cadencia, f"get_{name}")
    for name in CONTINUOUS + ["waveform", "candidate_onsets", "estimated_beats"]
]


def test_descriptors_share_a_uniform_time_grid(downloaded_chart):
    chart, _, _ = downloaded_chart
    for rate in [25, 50, 100, 12.5]:
        length = int(4 * rate)
        for name in CONTINUOUS:
            result = getattr(cadencia, f"get_{name}")(chart, sampling_rate=rate)
            assert isinstance(result, np.ndarray)
            assert result.shape == ((length, 12) if name == "chroma" else (length,))
            assert np.isfinite(result).all()
            if rate == 50:
                np.testing.assert_allclose(result, getattr(chart, name).values[:length])
        for name in ["candidate_onsets", "estimated_beats"]:
            result = getattr(cadencia, f"get_{name}")(chart, sampling_rate=rate)
            assert result.shape == (length,)
            assert np.issubdtype(result.dtype, np.integer)
            events = chart.candidate_onsets.times if name == "candidate_onsets" else chart.estimated_beats
            assert result.sum() == len(events)


def test_linear_interpolation_and_exclusive_duration():
    chart = SimpleNamespace(
        song=SimpleNamespace(duration_s=0.28),
        onset_strength=cadencia.Series([0, 0.1, 0.2, 0.3], [0, 10, 20, 30], "flux"),
    )
    result = cadencia.get_onset_strength(chart, sampling_rate=100)
    assert len(result) == 28
    np.testing.assert_allclose(result, np.arange(28))


def test_missing_values_are_not_filled_across_silence():
    chart = SimpleNamespace(
        song=SimpleNamespace(duration_s=0.1),
        spectral_centroid=cadencia.Series([0, 0.02, 0.04, 0.06, 0.08, 0.1], [440, 440, None, None, 440, 440], "Hz"),
    )
    result = cadencia.get_spectral_centroid(chart, sampling_rate=100)
    assert np.isnan(result[3:8]).all()
    np.testing.assert_allclose(result[[0, 1, 2, 8, 9]], 440)


def test_sparse_events_use_half_open_bins_and_preserve_collisions():
    events = [-0.1, 0, 0.01, 0.099, 0.1, 0.299, 0.3, 0.999, 1, 2, float("nan")]
    chart = SimpleNamespace(
        song=SimpleNamespace(duration_s=1),
        candidate_onsets=cadencia.Series(events, [1] * len(events), "flux"),
        estimated_beats=events,
    )
    for getter in [cadencia.get_candidate_onsets, cadencia.get_estimated_beats]:
        np.testing.assert_array_equal(getter(chart, sampling_rate=10), [3, 1, 1, 1, 0, 0, 0, 0, 0, 1])
        np.testing.assert_array_equal(getter(chart, sampling_rate=1), [7])
    chart.estimated_beats = []
    np.testing.assert_array_equal(cadencia.get_estimated_beats(chart, sampling_rate=2), [0, 0])


def test_waveform_is_mono_audio_and_resampling_preserves_pitch(downloaded_chart, tmp_path):
    chart, _, _ = downloaded_chart
    source, source_rate = sf.read(chart.song.path, dtype="float32")
    waveform = cadencia.get_waveform(chart, sampling_rate=source_rate)
    np.testing.assert_array_equal(waveform, source)
    result = cadencia.get_waveform(chart, sampling_rate=8000)
    assert result.shape == (32000,)
    section = result[5600:7600]
    frequencies = np.fft.rfftfreq(len(section), d=1 / 8000)
    assert frequencies[np.argmax(abs(np.fft.rfft(section)))] == pytest.approx(440)
    path = tmp_path / "stereo.wav"
    sf.write(path, np.column_stack([source, -source]), source_rate, subtype="FLOAT")
    stereo = replace(chart, song=replace(chart.song, path=path))
    np.testing.assert_array_equal(cadencia.get_waveform(stereo), np.zeros(len(source)))


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf"), True, "100", None])
def test_all_getters_reject_invalid_rates_before_reading_audio(rate):
    chart = SimpleNamespace(
        **{name: None for name in CONTINUOUS},
        candidate_onsets=cadencia.Series([], [], "flux"),
        estimated_beats=[],
    )
    for getter in GETTERS:
        with pytest.raises(ValueError, match="sampling_rate"):
            getter(chart, sampling_rate=rate)
