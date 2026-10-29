import bisect
import csv
import hashlib
import math
from collections import Counter

from .events import Hit, Hold


def sections(payload: bytes) -> tuple[dict, dict]:
    headers, records = {}, {}
    section = ""
    text = payload.decode("utf-8-sig")
    if not text.lstrip().startswith("osu file format v"):
        raise ValueError("Missing .osu format header")

    for line_number, original in enumerate(text.splitlines(), 1):
        line = original.strip()
        if not line or line.startswith("//") or line.startswith("osu file format"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
        elif section in {"TimingPoints", "HitObjects", "Events"}:
            records.setdefault(section, []).append((line_number, original))
        elif ":" in line:
            key, value = line.split(":", 1)
            headers.setdefault(section, {})[key.strip()] = value.strip()

    return headers, records


def validate_headers(headers: dict):
    if headers.get("General", {}).get("Mode", "0") != "3":
        raise ValueError("Expected an osu!mania chart with Mode: 3")
    if float(headers.get("Difficulty", {}).get("CircleSize", "0")) != 4:
        raise ValueError("Cadencia currently supports four-key osu!mania charts")


def parse_chart(payload: bytes, duration_s: float) -> dict:
    headers, records = sections(payload)
    validate_headers(headers)

    digest = hashlib.sha256(payload).hexdigest()
    lanes = {lane: [] for lane in range(4)}
    diagnostics, raw, timing = [], [], []

    def diagnose(code, message, source_line):
        diagnostics.append(dict(code=code, message=message, source_line=source_line))

    for number, line in records.get("TimingPoints", []):
        event = dict(id=f"{digest}:L{number}", source_line=number, source_text=line, start_s=None, category="timing")
        raw.append(event)
        try:
            values = next(csv.reader([line]))
            offset, length = float(values[0]) / 1000, float(values[1])
            meter = int(values[2]) if len(values) > 2 else 4
            inherited = len(values) > 6 and values[6].strip() == "0"
            effects = int(values[7]) if len(values) > 7 else 0
            if not math.isfinite(offset) or not math.isfinite(length) or meter <= 0:
                raise ValueError()
            event.update(start_s=offset, beat_length_ms=length, meter=meter, effects=effects,
                         category="scroll_control" if inherited else "tempo")
            if inherited:
                event["scroll_multiplier"] = -100 / length if length < 0 else None
            elif length > 0:
                timing.append(event)
            else:
                raise ValueError()
        except (ValueError, IndexError, csv.Error):
            diagnose("invalid_timing_point", "Timing point retained without a usable beat grid", number)

    timing.sort(key=lambda point: (point["start_s"], point["source_line"]))
    segments = []
    for point in timing:
        beat = 0.0
        if segments:
            previous = segments[-1]
            beat = previous["beat_position"] + (point["start_s"] - previous["start_s"]) * 1000 / previous["beat_length_ms"]
            if point["start_s"] == previous["start_s"]:
                segments.pop()
        segments.append(dict(start_s=point["start_s"], beat_length_ms=point["beat_length_ms"],
                             bpm=60000 / point["beat_length_ms"], meter=point["meter"],
                             beat_position=beat, source_id=point["id"], omit_first_barline=bool(point["effects"] & 8)))

    segment_times = [segment["start_s"] for segment in segments]
    grid = []
    for index, segment in enumerate(segments):
        step = segment["beat_length_ms"] / 1000
        end = segments[index + 1]["start_s"] if index + 1 < len(segments) else duration_s
        tick = math.ceil(-segment["start_s"] / step) if index == 0 else 0
        while segment["start_s"] + tick * step < min(end, duration_s):
            if len(grid) >= 100000:
                raise ValueError("Unreasonably large chart beat grid")
            time_s = segment["start_s"] + tick * step
            if time_s >= 0:
                grid.append(dict(time_s=time_s, beat_position=segment["beat_position"] + tick,
                                 meter=segment["meter"], is_barline=tick % segment["meter"] == 0
                                 and not (tick == 0 and segment["omit_first_barline"])))
            tick += 1

    decoded = 0
    for number, line in records.get("HitObjects", []):
        source = dict(id=f"{digest}:L{number}", source_line=number, source_text=line, category="unsupported_object", start_s=None)
        raw.append(source)
        try:
            values = next(csv.reader([line]))
            x, _, start_ms, flags = map(int, values[:4])
            start = start_ms / 1000
            lane = max(0, min(3, math.floor(x * 4 / 512)))
            source.update(start_s=start, lane=lane)
            is_hold, is_hit = bool(flags & 128), bool(flags & 1)
            if flags & (2 | 8) or is_hold == is_hit:
                raise ValueError("Unsupported hit-object flags")
            end = int(values[5].split(":", 1)[0]) / 1000 if is_hold else start
            eligible = start >= 0 and end <= duration_s
            if not eligible:
                diagnose("outside_audio", "Event retained outside the recording bounds", number)
            arguments = dict(start_s=start, lane=lane, source_line=number, source_type=flags,
                             source_x=x, id=source["id"], analysis_eligible=eligible)
            event = Hold(**arguments, end_s=end) if is_hold else Hit(**arguments)
            lanes[lane].append(event)
            decoded += 1
            source["category"] = event.kind
            if segments:
                segment = segments[max(0, bisect.bisect_right(segment_times, start) - 1)]
                source["beat_position"] = segment["beat_position"] + (start - segment["start_s"]) * 1000 / segment["beat_length_ms"]
        except (ValueError, IndexError, csv.Error) as error:
            diagnose("invalid_object", f"Source object retained without inventing a hit/hold: {error}", number)

    for lane in lanes.values():
        lane.sort(key=lambda event: (event.start_s, event.source_line))

    for number, line in records.get("Events", []):
        event = dict(id=f"{digest}:L{number}", source_line=number, source_text=line, category="media_or_storyboard", start_s=None)
        raw.append(event)
        try:
            values = next(csv.reader([line], skipinitialspace=True))
            if values[0] in {"2", "Break"}:
                event.update(category="break", start_s=int(values[1]) / 1000, end_s=int(values[2]) / 1000)
        except (ValueError, IndexError, csv.Error):
            diagnose("unparsed_context", "Context retained without interpreting its timing", number)

    total = len(records.get("HitObjects", []))
    return dict(lanes=lanes, timing_map=segments, beat_grid=grid, raw_events=raw, diagnostics=diagnostics,
                metadata=dict(headers=headers, chart_sha256=digest, column_count=4,
                              difficulty=headers.get("Metadata", {}).get("Version", "Unnamed"),
                              charter=headers.get("Metadata", {}).get("Creator", "Unknown"),
                              source_object_count=total, decoded_event_count=decoded,
                              coverage_status="complete" if total == decoded else "partial",
                              simultaneous_counts=dict(Counter(str(event.start_s) for lane in lanes.values() for event in lane)),
                              time_origin="Source milliseconds / 1000, relative to audio zero. No lead-in or calibration offset added.",
                              beat_origin="First red point defines beat zero; first tempo extends backwards; red points reset their local grid"))
