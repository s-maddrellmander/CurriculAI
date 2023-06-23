import subprocess


def test_black_formatting():
    result = subprocess.run(["black", "--check", "."], capture_output=True)
    assert result.returncode == 0, result.stderr
