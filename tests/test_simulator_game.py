from pathlib import Path
import shutil
import subprocess

import pytest


def test_browser_game_chords_and_holds():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is required for browser game logic tests')
    subprocess.run([node, 'tests/simulator_game.cjs'], cwd=Path(__file__).parents[1], check=True)
