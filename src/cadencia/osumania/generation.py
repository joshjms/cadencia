"""Reusable transformer inference and portable generated chart artifacts."""

import hashlib
from pathlib import Path
import tempfile

import numpy as np

from ..analysis import analyze_song
from ..audio import read_audio
from .chart import Chart
from .artifact import write_artifact


def candidate_times(analysis):
    """Audio-only candidates: estimated beats, their midpoints, and detected onsets."""
    beats = np.asarray(analysis["estimated_beats"], dtype=float)
    midpoints = (beats[:-1] + beats[1:]) / 2
    times = np.unique(np.round(np.concatenate([
        beats, midpoints, analysis["candidate_onsets"].times,
    ]), 3))
    duration = analysis["song"].duration_s
    return times[(times >= 2) & (times < duration)]


class ManiaModel:
    """A loaded CPU transformer with the preprocessing saved in its checkpoint."""

    def __init__(self, path, *, checkpoint=None):
        from ..dataset import FEATURES

        try:
            import torch
            from ..model import HitTransformer
        except ImportError as error:
            raise RuntimeError("Install inference dependencies with uv sync --extra training") from error

        checkpoint = checkpoint if checkpoint is not None else torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("features") != FEATURES:
            raise ValueError("Checkpoint feature order does not match the model")
        mean = np.asarray(checkpoint["mean"], dtype=np.float32)
        scale = np.asarray(checkpoint["scale"], dtype=np.float32)
        threshold = float(checkpoint["threshold"])
        if (mean.shape != (20,) or scale.shape != (20,) or not np.isfinite(mean).all()
                or not np.isfinite(scale).all() or (scale <= 0).any()
                or not 0 <= threshold <= 1):
            raise ValueError("Invalid checkpoint normalization or threshold")
        model = HitTransformer()
        model.load_state_dict(checkpoint["model"])
        model.eval()
        self.model, self.mean, self.scale, self.threshold = model, mean, scale, threshold
        self.checkpoint_sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def predict(self, path):
        """Generate first-lane taps from audio with autoregressive history."""
        from ..dataset import chart_samples
        import torch

        model, mean, scale, threshold = self.model, self.mean, self.scale, self.threshold
        analysis = analyze_song(Path(path), Path(path).stem, "Unknown artist")
        candidates = candidate_times(analysis)
        if not len(candidates):
            raise ValueError("No audio candidates after the two-second history window; try a longer song")
        metadata = analysis.pop("analysis_metadata")
        chart = Chart(metadata={"analysis": metadata}, id=0, game="osumania", lanes={i: [] for i in range(4)},
                      beat_grid=[{"time_s": float(t)} for t in candidates], **analysis)
        inputs, _, times = chart_samples(chart)
        history, notes = [], []
        with torch.inference_mode():
            for features, time_s in zip(inputs, times):
                edges = time_s - 2 + np.linspace(0, 2, 101)
                edges[-1] = time_s
                features[:, -1] = np.diff(np.searchsorted(history, edges, side="left"))
                normalized = ((np.where(np.isfinite(features), features, mean) - mean) / scale).astype(np.float32)
                probability = model(torch.from_numpy(normalized[None])).sigmoid().item()
                # Avoid nearly simultaneous taps on the single active lane.
                if probability >= threshold and (not history or time_s - history[-1] >= 0.08):
                    history.append(float(time_s))
                    notes.append(dict(time_s=float(time_s), lane=0, probability=probability))
        return dict(notes=notes, duration_s=chart.song.duration_s, threshold=threshold,
                    candidates=len(times), candidate_policy="Audio beats, beat midpoints and onsets; predicted tap history",
                    title=Path(path).stem, model_sha256=self.checkpoint_sha256)

    def create_chart(self, mp3_file, *, out):
        """Save a self-contained chart ZIP containing notes and decoded audio.

        Returns the output Path. Existing output files are never overwritten.
        """
        import soundfile as sf

        out = Path(out)
        if out.exists():
            raise FileExistsError(f"Chart already exists: {out}")
        result = self.predict(mp3_file)
        with tempfile.TemporaryDirectory(prefix="cadencia-chart-") as temporary:
            playback = Path(temporary) / "audio.wav"
            samples, rate, _ = read_audio(Path(mp3_file))
            sf.write(playback, samples, rate, subtype="PCM_16")
            write_artifact(out, result, playback)
        return out


def load_model(path="v1", *, device="cpu"):
    """Load a Cadencia transformer checkpoint once for repeated chart generation."""
    import torch

    from ..model_versions import is_model_version, resolve_model_version
    if is_model_version(str(path)):
        path = resolve_model_version(str(path))
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") in ("cadencia.placement.v1", "cadencia.placement.v2", "cadencia.placement.intervals.v3", "cadencia.placement.planned.v4", "cadencia.placement.planned.v5"):
        from .placement_generation import PlacementModel

        return PlacementModel(path, checkpoint=checkpoint, device=device)
    if device != "cpu":
        raise ValueError("Legacy tap checkpoints support CPU inference only")
    return ManiaModel(path, checkpoint=checkpoint)
