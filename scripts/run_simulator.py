"""Generate Tunggu Apa Lagi with the latest placement checkpoint and play on port 6767."""

import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile

from cadencia.osumania import Simulator, SimulatorConfig, load_model


def main():
    data = Path(__file__).resolve().parents[1] / 'data'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--song', type=Path, help='Input audio/video file; defaults to Tunggu Apa Lagi')
    parser.add_argument('--chart-out', type=Path, help='Explicit output .cadencia artifact')
    models = parser.add_mutually_exclusive_group()
    models.add_argument('--checkpoint', type=Path)
    models.add_argument('--model-version', help='Model release tag (default v1)')
    parser.add_argument('--difficulty', type=float, default=3., help='Target osu!mania stars')
    parser.add_argument('--bpm', type=float, help='Optional BPM override; otherwise estimate')
    parser.add_argument('--port', type=int, default=6767)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--decoding', choices=['sample','argmax'], help='Joint models default to seeded sampling')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--beat-alignment', action=argparse.BooleanOptionalAction,default=None,help='Enabled by default for high-resolution checkpoints')
    parser.add_argument('--alignment-max-shift-ms',type=float,default=25.)
    parser.add_argument('--repeat-sections', action='store_true', help='Reuse patterns in audio-matched repeated eight-bar sections')
    parser.add_argument('--repeat-variation', type=float, default=.2, help='Fraction of repeated bars retaining their independently generated patterns')
    parser.add_argument('--generate-only', action='store_true')
    args = parser.parse_args()
    import torch

    torch.set_num_threads(2)
    from cadencia.model_versions import resolve_model_version, DEFAULT_MODEL_VERSION
    args.checkpoint = args.checkpoint or resolve_model_version(args.model_version or DEFAULT_MODEL_VERSION)
    song = args.song or data / 'Nyoman Paul - Tunggu Apa Lagi (Official Music Video).mp3'
    # Snapshot the atomically replaced training checkpoint so the generated
    # artifact's provenance matches the exact weights read during this run.
    with tempfile.TemporaryDirectory(prefix='cadencia-model-') as temporary:
        snapshot = Path(temporary) / 'placement.pt'
        shutil.copyfile(args.checkpoint, snapshot)
        digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        bpm_label = f'{args.bpm:g}' if args.bpm is not None else 'auto'
        chart = args.chart_out or data / 'generated' / f'{song.stem}.placement-{digest[:12]}-{args.difficulty:g}stars-{bpm_label}bpm-{args.decoding or 'auto'}-seed{args.seed}-T{args.temperature:g}-align{args.beat_alignment}-{args.alignment_max_shift_ms:g}ms-repeat{args.repeat_sections}-var{args.repeat_variation:g}.cadencia'
        if not chart.exists():
            print(f'Generating four-lane chart from {song.name} ({args.difficulty:g} stars, {args.device})…', flush=True)
            model = load_model(snapshot, device=args.device)
            model.create_chart(song, out=chart, difficulty=args.difficulty, bpm=args.bpm,
                               cache_dir=data / 'mert_cache', decoding=args.decoding, seed=args.seed, temperature=args.temperature,
                               align_beats=args.beat_alignment,alignment_max_shift_ms=args.alignment_max_shift_ms,
                               repeat_sections=args.repeat_sections,repeat_variation=args.repeat_variation)
            print(f'Saved {chart}', flush=True)
        else:
            print(f'Using saved chart: {chart}', flush=True)
    if not args.generate_only:
        Simulator(chart, port=args.port, config=SimulatorConfig(), open_browser=False)


if __name__ == '__main__':
    main()
