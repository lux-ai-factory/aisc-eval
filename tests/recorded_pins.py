"""The pins the engine records for a catalogue plugin and hands each run: its wheel's sha256 and a hashed lock
that names that wheel. A run of a catalogue plugin without them is refused (test_run_pins.py)."""

WHEEL_SHA256 = "a" * 64


def recorded_pins(package: str = "pkg", version: str = "1.0") -> dict:
    return {
        "wheel_sha256": WHEEL_SHA256,
        "dependency_lock": f"{package}=={version} \\\n    --hash=sha256:{WHEEL_SHA256}\n",
    }
