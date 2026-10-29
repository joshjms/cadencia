import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Generic, TypeVar

E = TypeVar("E")
SCHEMA_VERSION = "cadencia.trace.v1"


@dataclass
class Song:
    title: str
    artist: str
    path: Path
    duration_s: float
    sample_rate: int
    channels: int
    sha256: str


@dataclass
class Series:
    times: list[float]
    values: list
    unit: str
    labels: list[str] | None = None


@dataclass
class Waveform:
    times: list[float]
    minimum: list[float]
    maximum: list[float]
    bin_width_s: float


@dataclass
class Chart(Generic[E]):
    id: int
    game: str
    song: Song
    lanes: dict[int, list[E]]
    waveform: Waveform
    onset_strength: Series
    candidate_onsets: Series
    rms_energy: Series
    spectral_centroid: Series
    harmonic_energy_fraction: Series
    chroma: Series
    estimated_beats: list[float]
    metadata: dict = field(default_factory=dict)
    timing_map: list[dict] = field(default_factory=list)
    beat_grid: list[dict] = field(default_factory=list)
    raw_events: list[dict] = field(default_factory=list)
    diagnostics: list[dict] = field(default_factory=list)

    @property
    def events(self) -> list[E]:
        return sorted(
            (event for lane in self.lanes.values() for event in lane),
            key=lambda event: (event.start_s, event.lane, event.source_line),
        )

    @property
    def trace_path(self) -> Path:
        return self.song.path.parent / "trace.json"

    def to_dict(self) -> dict:
        result = asdict(self)
        result["schema_version"] = SCHEMA_VERSION
        result["song"]["path"] = self.song.path.name
        return result

    def save(self) -> Path:
        temporary = self.trace_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(self.trace_path)
        return self.trace_path
