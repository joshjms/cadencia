"""Generate Tunggu Apa Lagi's chart and play it at http://127.0.0.1:6767."""

from pathlib import Path

from cadencia.osumania import Simulator, SimulatorConfig, load_model


def main():
    data = Path(__file__).resolve().parent / "data"
    song = data / "Nyoman Paul - Tunggu Apa Lagi (Official Music Video).mp3"
    chart = data / "generated" / f"{song.stem}.cadencia"
    if not chart.exists():
        print(f"Generating chart from {song.name}…", flush=True)
        model = load_model(data / "ranked_hit_transformer.pt")
        model.create_chart(song, out=chart)
        print(f"Saved {chart}", flush=True)
    else:
        print(f"Using saved chart: {chart}", flush=True)
    Simulator(chart, port=6767, config=SimulatorConfig(), open_browser=False)


if __name__ == "__main__":
    main()
