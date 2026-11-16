"""I7 — no secret leaves the point of use.

The scrub helper is what turns a leak into a non-issue; every outbound path in
the plugin run (stdout, stderr, the execution log, an artifact, a failure
message) runs through it.
"""

from aisc_eval.utils.scrub import scrub_bytes, scrub_secrets


def test_scrub_replaces_every_occurrence():
    assert (
        scrub_secrets("a secret, then the secret again", ["secret"])
        == "a ***, then the *** again"
    )


def test_scrub_masks_a_longer_secret_whole():
    # "sec" is a substring of "secret"; the longer value must win.
    assert scrub_secrets("secret", ["sec", "secret"]) == "***"


def test_scrub_ignores_falsy_values():
    assert scrub_secrets("unchanged", ["", None]) == "unchanged"


def test_scrub_bytes_masks_text():
    assert scrub_bytes(b"token=abc123", ["abc123"]) == b"token=***"


def test_scrub_bytes_leaves_binary_untouched():
    payload = bytes([0xFF, 0xFE, 0x00, 0x01])
    assert scrub_bytes(payload, ["abc123"]) == payload
