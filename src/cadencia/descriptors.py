from numbers import Real

import librosa
import numpy as np

from .audio import read_audio
from .chart import Chart


def _validate_rate(sampling_rate):
    if isinstance(sampling_rate, bool) or not isinstance(sampling_rate, Real):
        raise ValueError("sampling_rate must be a positive, finite number")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate must be a positive, finite number")
    return float(sampling_rate)


def _sample_count(chart, rate):
    count = int(np.ceil(chart.song.duration_s * rate))
    if count and (count - 1) / rate >= chart.song.duration_s:
        count -= 1
    return count


def _times(chart, sampling_rate):
    rate = _validate_rate(sampling_rate)
    return np.arange(_sample_count(chart, rate)) / rate


def _interpolate(chart, series, sampling_rate):
    times = _times(chart, sampling_rate)
    values = np.asarray(series.values, dtype=float)
    if values.ndim == 1:
        return np.interp(times, series.times, values)
    return np.column_stack([
        np.interp(times, series.times, column) for column in values.T
    ])


def _counts(chart, event_times, sampling_rate):
    times = _times(chart, sampling_rate)
    events = np.asarray(event_times, dtype=float)
    events = events[(events >= 0) & (events < chart.song.duration_s)]
    indices = np.searchsorted(times, events, side="right") - 1
    return np.bincount(indices, minlength=len(times))


def get_waveform(chart: Chart, sampling_rate: float = 22050) -> np.ndarray:
    rate = _validate_rate(sampling_rate)
    samples, source_rate, _ = read_audio(chart.song.path)
    waveform = librosa.resample(
        samples.mean(axis=1), orig_sr=source_rate, target_sr=rate, res_type="soxr_hq",
    )
    return waveform[:_sample_count(chart, rate)]


def get_onset_strength(chart: Chart, sampling_rate: float = 50) -> np.ndarray:
    return _interpolate(chart, chart.onset_strength, sampling_rate)


def get_rms_energy(chart: Chart, sampling_rate: float = 50) -> np.ndarray:
    return _interpolate(chart, chart.rms_energy, sampling_rate)


def get_spectral_centroid(chart: Chart, sampling_rate: float = 50) -> np.ndarray:
    return _interpolate(chart, chart.spectral_centroid, sampling_rate)


def get_harmonic_energy_fraction(chart: Chart, sampling_rate: float = 50) -> np.ndarray:
    return _interpolate(chart, chart.harmonic_energy_fraction, sampling_rate)


def get_chroma(chart: Chart, sampling_rate: float = 50) -> np.ndarray:
    return _interpolate(chart, chart.chroma, sampling_rate)


def get_candidate_onsets(chart: Chart, sampling_rate: float = 50) -> np.ndarray:
    return _counts(chart, chart.candidate_onsets.times, sampling_rate)


def get_estimated_beats(chart: Chart, sampling_rate: float = 50) -> np.ndarray:
    return _counts(chart, chart.estimated_beats, sampling_rate)
