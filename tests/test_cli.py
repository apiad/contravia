import subprocess
import sys


def test_help_lists_the_three_commands():
    out = subprocess.run([sys.executable, "-m", "havana.cli", "--help"],
                         capture_output=True, text=True, check=True).stdout
    for cmd in ("prepare", "play", "bench"):
        assert cmd in out
