"""Local browser simulator and audio-only transformer inference."""

from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import secrets
import tempfile
import threading
import webbrowser


from ..audio import read_audio


@dataclass(frozen=True)
class SimulatorConfig:
    """Browser KeyboardEvent.code bindings and tile travel speed in pixels/second."""

    keybinds: tuple[str, str, str, str] = ("KeyD", "KeyF", "KeyJ", "KeyK")
    tile_speed: float = 420.0

    def __post_init__(self):
        if len(self.keybinds) != 4 or len(set(self.keybinds)) != 4 or any(
            not isinstance(key, str) or not key.strip() for key in self.keybinds
        ):
            raise ValueError("Provide four distinct nonempty KeyboardEvent.code keybinds")
        if isinstance(self.tile_speed, bool) or not math.isfinite(self.tile_speed) or self.tile_speed <= 0:
            raise ValueError("tile_speed must be positive and finite")


# Retained for callers of the initial simulator inference helpers.
from .generation import candidate_times, load_model


def generate_chart(path, checkpoint_path, *, difficulty=None, bpm=None):
    import soundfile as sf

    model = load_model(checkpoint_path)
    if hasattr(model, 'config'):
        if difficulty is None:
            raise ValueError('Four-lane generation requires a difficulty star-rating target')
        result = model.predict(path, difficulty=difficulty, bpm=bpm)
    else:
        result = model.predict(path)
    samples, rate, _ = read_audio(path)
    sf.write(path.with_name("playback.wav"), samples, rate, subtype="PCM_16")
    return result

class Simulator:
    """Start a loopback browser game. Use block=False for notebooks and close() to stop."""

    def __init__(self, chart_file=None, *, port=8080, config=None, checkpoint=None, data_dir=None,
                 block=True, open_browser=True):
        self.config = config or SimulatorConfig()
        if not isinstance(self.config, SimulatorConfig):
            raise TypeError("config must be a SimulatorConfig")
        from ..model_versions import resolve_model_version
        self.checkpoint = Path(checkpoint) if checkpoint else resolve_model_version()
        self._temporary = tempfile.TemporaryDirectory(prefix="cadencia-simulator-")
        self._root = Path(self._temporary.name)
        self._lock = threading.Lock()
        self._state = {"status": "idle"}
        self._worker = None
        self._token = secrets.token_urlsafe(32)
        self._closed = False
        try:
            if chart_file is not None:
                from .artifact import read_artifact

                result, audio = read_artifact(chart_file)
                (self._root / "playback.wav").write_bytes(audio)
                self._state = dict(status="ready", **result)
            self.server = ThreadingHTTPServer(("127.0.0.1", int(port)), self._handler())
        except Exception:
            self._temporary.cleanup()
            raise
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()
        print(f"Cadencia Simulator: {self.url}", flush=True)
        if open_browser:
            webbrowser.open(self.url)
        if block:
            try:
                self._thread.join()
            except KeyboardInterrupt:
                pass
            finally:
                self.close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.server.shutdown()
        self.server.server_close()
        self._thread.join()
        if self._worker:
            self._worker.join()
        self._temporary.cleanup()

    def _generate(self, difficulty=None, bpm=None):
        try:
            options = dict(difficulty=difficulty, bpm=bpm) if difficulty is not None else {}
            result = generate_chart(self._root / "upload.mp3", self.checkpoint, **options)
            state = dict(status="ready", **result)
        except Exception as error:
            state = dict(status="error", error=f"{type(error).__name__}: {error}")
        with self._lock:
            self._state = state

    def _handler(self):
        simulator = self
        assets = Path(__file__).parents[1] / "_simulator"

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def reply(self, status, value, content_type="application/json"):
                payload = json.dumps(value).encode() if content_type == "application/json" else value
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                if self.path == "/api/config":
                    self.reply(200, dict(**asdict(simulator.config), token=simulator._token))
                elif self.path == "/api/status":
                    with simulator._lock:
                        state = dict(simulator._state)
                    self.reply(200, state)
                elif self.path == "/audio.wav":
                    with simulator._lock:
                        if simulator._state["status"] != "ready":
                            self.reply(404, {"error": "No prepared audio"})
                            return
                        self.reply(200, (simulator._root / "playback.wav").read_bytes(), "audio/wav")
                elif self.path in ("/", "/app.js", "/style.css"):
                    filename, mime = {"/": ("index.html", "text/html; charset=utf-8"),
                                      "/app.js": ("app.js", "text/javascript"),
                                      "/style.css": ("style.css", "text/css")}[self.path]
                    self.reply(200, (assets / filename).read_bytes(), mime)
                else:
                    self.reply(404, {"error": "Not found"})

            def do_POST(self):
                if self.path != "/api/song":
                    self.reply(404, {"error": "Not found"})
                    return
                if self.headers.get("X-Cadencia-Token") != simulator._token:
                    self.reply(403, {"error": "Invalid upload token"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = 0
                if not 0 < length <= 100 * 1024 * 1024:
                    self.reply(413, {"error": "Upload a song between 1 byte and 100 MiB"})
                    return
                if not simulator.checkpoint.is_file():
                    self.reply(400, {"error": f"Checkpoint not found: {simulator.checkpoint}"})
                    return
                try:
                    difficulty = self.headers.get('X-Cadencia-Difficulty')
                    difficulty = float(difficulty) if difficulty is not None else None
                    bpm = self.headers.get('X-Cadencia-Bpm')
                    bpm = float(bpm) if bpm else None
                    if difficulty is not None and (not math.isfinite(difficulty) or difficulty <= 0):
                        raise ValueError('Difficulty must be a positive star rating')
                    if bpm is not None and (not math.isfinite(bpm) or bpm <= 0):
                        raise ValueError('BPM must be positive')
                except ValueError as error:
                    self.reply(400, {'error': str(error)})
                    return
                with simulator._lock:
                    if simulator._state["status"] == "processing":
                        self.reply(409, {"error": "A song is already processing"})
                        return
                    simulator._state = {"status": "processing"}
                try:
                    self.connection.settimeout(60)
                    payload = self.rfile.read(length)
                    if len(payload) != length:
                        raise ValueError("Incomplete upload")
                    (simulator._root / "upload.mp3").write_bytes(payload)
                    simulator._worker = threading.Thread(target=simulator._generate, args=(difficulty, bpm), daemon=True)
                    simulator._worker.start()
                except Exception as error:
                    with simulator._lock:
                        simulator._state = {"status": "error", "error": str(error)}
                    self.reply(400, {"error": str(error)})
                    return
                self.reply(202, {"status": "processing"})

        return Handler
