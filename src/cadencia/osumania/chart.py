import json
from pathlib import Path

from ..chart import SCHEMA_VERSION, Chart as BaseChart, Series, Song, Waveform
from .events import Event, Hit, Hold


class Chart(BaseChart[Event]):
    @classmethod
    def read(cls, path: str | Path) -> "Chart":
        path = Path(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.pop("schema_version", None) != SCHEMA_VERSION or value.get("game") != "osumania":
            raise ValueError(f"Unsupported trace format: {path}")
        song_path = Path(value["song"]["path"])
        if song_path.name != str(song_path):
            raise ValueError("The cached song must be stored beside trace.json")
        value["song"]["path"] = path.parent / song_path
        if not value["song"]["path"].is_file():
            raise FileNotFoundError(f"Cached song is missing: {value['song']['path']}")
        value["song"] = Song(**value["song"])
        lanes = {}
        for lane, events in value["lanes"].items():
            lanes[int(lane)] = []
            for event in events:
                kind = event.pop("kind")
                event_class = {"hit": Hit, "hold": Hold}.get(kind)
                if event_class is None:
                    raise ValueError(f"Unsupported cached event kind: {kind}")
                if event["lane"] != int(lane):
                    raise ValueError("Cached event lane does not match its lane group")
                lanes[int(lane)].append(event_class(**event))
        if set(lanes) != set(range(4)):
            raise ValueError("A four-key trace must contain lanes 0 through 3")
        value["lanes"] = lanes
        value["waveform"] = Waveform(**value["waveform"])
        for name in ["onset_strength", "candidate_onsets", "rms_energy", "spectral_centroid", "harmonic_energy_fraction", "chroma"]:
            value[name] = Series(**value[name])
        return cls(**value)
