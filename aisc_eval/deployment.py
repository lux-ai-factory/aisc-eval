"""Where this worker runs: on its own, or inside the Sandbox Configurator.

One switch, AISC_DEPLOYMENT, read once at import into MODE. Every behaviour that
differs between the two modes asks this module; nothing else reads the variable.

This is the worker's own copy of the backend's aisc_backend.deployment contract
(mode(), check_environment(), is_configurator(), is_standalone()): the worker does
not import Django settings, and it has no .env loader (aisc_eval.utils.env reads
os.environ directly, so there is nothing to read this after).
"""
import os
from collections.abc import Mapping

STANDALONE = "standalone"
CONFIGURATOR = "configurator"
MODES = (STANDALONE, CONFIGURATOR)


def mode(env: Mapping[str, str] = os.environ) -> str:
    """AISC_DEPLOYMENT, forgiving case and spaces. Absent means standalone; present
    and not one of the two values stops the worker (an empty variable is a mistake
    in a compose file, not a request for the default)."""
    if "AISC_DEPLOYMENT" not in env:
        return STANDALONE
    value = env["AISC_DEPLOYMENT"].strip().lower()
    if value not in MODES:
        raise SystemExit(
            f"AISC_DEPLOYMENT must be {STANDALONE} or {CONFIGURATOR}, not {env['AISC_DEPLOYMENT']!r}"
        )
    return value


def check_environment(env: Mapping[str, str] = os.environ) -> None:
    """The worker has no database of its own (I7.3): unlike the backend, there is no
    Postgres/project-databases constraint to check here. Kept for the same contract;
    it only validates AISC_DEPLOYMENT itself, same as mode()."""
    mode(env)


def is_configurator() -> bool:
    return MODE == CONFIGURATOR


def is_standalone() -> bool:
    return not is_configurator()


MODE = mode()
