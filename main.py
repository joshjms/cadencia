import cadencia


def main():
    cadencia.osumania.load_chart(
        "https://osu.ppy.sh/beatmapsets/2043161#mania/4855704"
    )
    cadencia.osumania.visualize(port="8080")


if __name__ == "__main__":
    main()
