import json
import mimetypes
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

LAYERS = [
    "lane0", "lane1", "lane2", "lane3", "waveform", "onset_strength",
    "candidate_onsets", "rms_energy", "spectral_centroid",
    "harmonic_energy_fraction", "chroma", "estimated_beats", "timing",
]


def selected_layers(include):
    if include is None:
        return LAYERS.copy()
    if isinstance(include, str):
        include = [include]
    selected = {name.strip() for item in include for name in item.split(",") if name.strip()}
    unknown = selected.difference(LAYERS)
    if unknown:
        raise ValueError(f"Unknown timeline layers: {', '.join(sorted(unknown))}")
    return [name for name in LAYERS if name in selected]


def manifest(data_dir):
    songs = {}
    for path in sorted((data_dir / "osumania" / "charts").glob("*/trace.json")):
        trace = json.loads(path.read_text(encoding="utf-8"))
        if trace.get("game") != "osumania":
            continue
        song, metadata = trace["song"], trace["metadata"]
        group = songs.setdefault(song["sha256"], dict(
            song_id=song["sha256"], title=song["title"], artist=song["artist"],
            duration_s=song["duration_s"], charts=[],
            audio_path=str((path.parent / song["path"]).relative_to(data_dir)),
        ))
        group["charts"].append(dict(
            chart_id=str(trace["id"]), difficulty_label=metadata["difficulty"], column_count=4,
            charter=metadata["charter"], beatmap_id=trace["id"], source_url=metadata["source_url"],
            processed_path=str(path.relative_to(data_dir)),
        ))
    return dict(schema_version="cadencia.library.v1", songs=list(songs.values()))


def make_handler(data_dir, include):
    assets = Path(__file__).with_name("_viewer")

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.send_asset(head=True)

        def do_GET(self):
            self.send_asset()

        def log_message(self, *_):
            pass

        def send_asset(self, head=False):
            url = unquote(urlsplit(self.path).path)
            if url in {"/api/manifest", "/api/config"}:
                try:
                    value = manifest(data_dir) if url.endswith("manifest") else dict(include=include)
                except (ValueError, OSError, KeyError) as error:
                    self.send_error(500, f"Could not read chart cache: {error}")
                    return
                payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                if not head:
                    self.wfile.write(payload)
                return

            root, relative = (data_dir, url[6:]) if url.startswith("/data/") else (assets, url.lstrip("/") or "index.html")
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                self.send_error(404, "Chart or asset not found")
                return
            size = path.stat().st_size
            start, end, partial = 0, size - 1, False
            if value := self.headers.get("Range"):
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
                if not match or not any(match.groups()):
                    self.send_error(416)
                    return
                start = int(match[1]) if match[1] else max(0, size - int(match[2]))
                end = min(end, int(match[2])) if match[1] and match[2] else end
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                partial = True
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", mimetypes.guess_type(path)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(max(0, end - start + 1)))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if not head:
                try:
                    with path.open("rb") as stream:
                        stream.seek(start)
                        remaining = end - start + 1
                        while remaining > 0 and (chunk := stream.read(min(65536, remaining))):
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass

    return Handler


def visualize(data_dir, port="8080", include=None, block=True):
    include = selected_layers(include)
    server = ThreadingHTTPServer(("127.0.0.1", int(port)), make_handler(data_dir, include))
    print(f"Cadencia: http://127.0.0.1:{server.server_port}", flush=True)
    if not block:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
