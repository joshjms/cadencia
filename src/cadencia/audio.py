"""Audio decoding shared by preprocessing and waveform descriptors."""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def read_audio(path: Path) -> tuple[np.ndarray, int, str]:
    """Read native-rate float32 audio, falling back to FFmpeg on decoder errors.

    The fallback keeps the original channels and sample rate and leaves the
    source file untouched. The final value identifies the decoder for provenance.
    """
    path = Path(path)
    decoder = "soundfile"
    try:
        samples, source_rate = sf.read(path, dtype="float32", always_2d=True)
    except sf.LibsndfileError as soundfile_error:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                f"SoundFile could not decode {path}: {soundfile_error}. "
                "Install FFmpeg to enable the alternative decoder."
            ) from soundfile_error
        logger.warning("SoundFile could not decode %s; trying FFmpeg: %s", path, soundfile_error)
        with tempfile.TemporaryDirectory(prefix="cadencia-audio-") as temporary:
            decoded = Path(temporary) / "decoded.wav"
            try:
                result = subprocess.run(
                    [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
                     "-i", str(path.resolve()), "-map", "0:a:0", "-vn",
                     "-c:a", "pcm_f32le", str(decoded)],
                    capture_output=True, text=True, timeout=180,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise RuntimeError(f"FFmpeg could not decode {path}: {error}") from error
            if result.returncode:
                raise RuntimeError(
                    f"Neither SoundFile nor FFmpeg could decode {path}. "
                    f"SoundFile: {soundfile_error}; FFmpeg: {result.stderr.strip()[-2000:]}"
                ) from soundfile_error
            samples, source_rate = sf.read(decoded, dtype="float32", always_2d=True)
        decoder = "ffmpeg"
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError("The song must contain finite audio samples")
    return samples, source_rate, decoder
