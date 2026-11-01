"""Generate an osu!mania .osz from local audio/video using the existing scripts."""
import argparse
from pathlib import Path
import subprocess
import sys

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Local music video or audio file (MP4, MP3, WAV, etc.)')
    parser.add_argument('--out', type=Path, help='Output .osz; defaults to data/exports/INPUT.osz')
    models = parser.add_mutually_exclusive_group()
    models.add_argument('--checkpoint', type=Path, help='Explicit checkpoint instead of a versioned release')
    models.add_argument('--model-version', default=None, help='Model release tag (default v1)')
    parser.add_argument('--difficulty', type=float, default=2., help='Model conditioning target, not guaranteed star rating')
    parser.add_argument('--target-stars', type=float, help='Optional calculator-guided simplification; may differ from in-game rating')
    parser.add_argument('--tolerance', type=float, default=.03)
    parser.add_argument('--bpm', type=float)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--decoding', choices=['sample','argmax'], default='sample')
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--beat-alignment',action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--alignment-max-shift-ms',type=float,default=25.)
    parser.add_argument('--repeat-sections', action='store_true', help='Reuse patterns across matching audio sections')
    parser.add_argument('--repeat-variation', type=float, default=.2)
    parser.add_argument('--title')
    parser.add_argument('--artist', default='Unknown artist')
    args = parser.parse_args()
    source = args.input.resolve()
    from cadencia.model_versions import resolve_model_version, DEFAULT_MODEL_VERSION
    checkpoint = args.checkpoint.resolve() if args.checkpoint else resolve_model_version(args.model_version or DEFAULT_MODEL_VERSION)
    out = (args.out or ROOT/'data/exports'/f'{source.stem}.osz').resolve()
    artifact = out.with_suffix('.cadencia')
    if not source.is_file(): parser.error(f'Input does not exist: {source}')
    if not checkpoint.is_file(): parser.error(f'Checkpoint does not exist: {checkpoint}')
    if out.suffix.lower() != '.osz': parser.error('--out must end in .osz')
    for path in [out,artifact]:
        if path.exists(): parser.error(f'Output already exists: {path}; choose another --out')
    # read_audio's FFmpeg fallback handles video inputs; the artifact embeds the
    # decoded soundtrack. Generation snapshots the selected checkpoint itself.
    generate = [sys.executable,str(SCRIPTS/'run_simulator.py'),'--song',str(source),
                '--chart-out',str(artifact),'--checkpoint',str(checkpoint),
                '--difficulty',str(args.difficulty),'--device',args.device,
                '--seed',str(args.seed),'--decoding',args.decoding,
                '--temperature',str(args.temperature),'--generate-only']
    if args.beat_alignment is not None:
        generate += ['--beat-alignment' if args.beat_alignment else '--no-beat-alignment']
    generate += ['--alignment-max-shift-ms',str(args.alignment_max_shift_ms)]
    if args.repeat_sections:
        generate += ['--repeat-sections','--repeat-variation',str(args.repeat_variation)]
    if args.bpm is not None: generate += ['--bpm',str(args.bpm)]
    subprocess.run(generate,check=True)
    convert = [sys.executable,str(SCRIPTS/'convert_cadencia.py'),str(artifact),
               '--out',str(out),'--title',args.title or source.stem,'--artist',args.artist]
    if args.target_stars is not None:
        convert += ['--target-stars',str(args.target_stars),'--tolerance',str(args.tolerance),
                    '--chart-out',str(out.with_name(out.stem+'.calibrated.cadencia'))]
    subprocess.run(convert,check=True)
    print(f'Import into osu!: {out}',flush=True)


if __name__ == '__main__':
    main()
