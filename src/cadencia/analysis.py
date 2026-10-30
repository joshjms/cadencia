import hashlib
from importlib.metadata import version
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .chart import Series, Song, Waveform
from .audio import read_audio

SAMPLE_RATE = 22050
FRAME_LENGTH = 2048
HOP_LENGTH = 441


def analyze_song(path: Path, title: str, artist: str) -> dict:
    samples, source_rate, decoder = read_audio(path)

    y = librosa.resample(samples.mean(axis=1), orig_sr=source_rate, target_sr=SAMPLE_RATE, res_type="soxr_hq")
    spectrum = abs(librosa.stft(y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH, center=True, pad_mode="constant"))
    mel = librosa.feature.melspectrogram(S=spectrum**2, sr=SAMPLE_RATE, n_mels=128)
    flux = librosa.onset.onset_strength(
        S=librosa.power_to_db(mel), sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
        n_fft=FRAME_LENGTH, center=True,
    )
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, center=True)[0]
    centroid = librosa.feature.spectral_centroid(S=spectrum, sr=SAMPLE_RATE)[0]
    harmonic, _ = librosa.decompose.hpss(spectrum)
    power = (spectrum**2).sum(axis=0)
    harmonic_fraction = (harmonic**2).sum(axis=0) / np.maximum(power, 1e-12)
    chroma = librosa.feature.chroma_stft(S=harmonic**2, sr=SAMPLE_RATE, n_fft=FRAME_LENGTH, tuning=0)

    scale = max(float(np.percentile(flux, 99)), 1e-12)
    detector_input = np.minimum(flux / scale, 1)
    peaks = librosa.onset.onset_detect(
        onset_envelope=detector_input, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
        normalize=False, backtrack=False, pre_max=3, post_max=3,
        pre_avg=10, post_avg=10, delta=0.07, wait=3,
    )
    _, beats = librosa.beat.beat_track(onset_envelope=flux, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    times = (np.arange(len(rms)) * HOP_LENGTH / SAMPLE_RATE).tolist()
    duration = len(samples) / source_rate
    peaks = peaks[peaks * HOP_LENGTH / SAMPLE_RATE < duration]
    beats = beats[beats * HOP_LENGTH / SAMPLE_RATE < duration]
    bins = [y[start:start + HOP_LENGTH] for start in range(0, len(y), HOP_LENGTH)]

    return dict(
        song=Song(title, artist, path, duration, source_rate, samples.shape[1], hashlib.sha256(path.read_bytes()).hexdigest()),
        waveform=Waveform(
            (np.arange(len(bins)) * HOP_LENGTH / SAMPLE_RATE).tolist(),
            [float(window.min()) for window in bins],
            [float(window.max()) for window in bins],
            HOP_LENGTH / SAMPLE_RATE,
        ),
        onset_strength=Series(times, flux.tolist(), "spectral flux"),
        candidate_onsets=Series((peaks * HOP_LENGTH / SAMPLE_RATE).tolist(), flux[peaks].tolist(), "spectral flux"),
        rms_energy=Series(times, rms.tolist(), "linear amplitude"),
        spectral_centroid=Series(times, [float(v) if p > 1e-12 else None for v, p in zip(centroid, power)], "Hz"),
        harmonic_energy_fraction=Series(times, [float(v) if p > 1e-12 else None for v, p in zip(harmonic_fraction, power)], "ratio"),
        chroma=Series(times, chroma.T.tolist(), "normalized pitch-class energy", ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]),
        estimated_beats=(beats * HOP_LENGTH / SAMPLE_RATE).tolist(),
        analysis_metadata=dict(
            version=1, audio_decoder=decoder, sample_rate=SAMPLE_RATE, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH,
            stft_window="Hann", rms_window="rectangular", center=True, padding="constant zero",
            channel_mix="mean", resampling="soxr_hq",
            feature_timestamps="Frame centers: frame index * hop / sample rate",
            waveform_timestamps="Bin left edges; bin width = hop / sample rate",
            onset_detector="Raw flux / per-song 99th percentile, clipped to [0,1]; no backtracking",
            onset_scale=scale, onset_peak_parameters=dict(pre_max=3, post_max=3, pre_avg=10, post_avg=10, delta=0.07, wait=3),
            harmonic_fraction="Squared HPSS harmonic magnitude / squared total magnitude; mixture proxy",
            chroma="HPSS harmonic power, STFT chroma, tuning 0, L-infinity frame normalization",
            versions={name: version(name) for name in ["librosa", "numpy", "scipy", "soundfile", "soxr"]},
            libsndfile=sf.__libsndfile_version__,
        ),
    )
