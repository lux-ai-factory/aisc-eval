"""The files a plugin leaves are uploaded without the values the worker handed it: each project secret, the
run's pass and each secret-named configuration value is replaced, byte for byte, in the file's content and
in its name. Only those exact values: a file is evidence, so nothing merely shaped like a key is rewritten.
The run is faked as test_plugin_process.py fakes it."""

import pytest

from aisc_eval import celery_tasks, scrub

from test_plugin_process import (  # noqa: F401  (fixtures)
    TARGET,
    _Answer,
    _resource,
    platform,
    worker,
)
from test_scrubbed_failures import JUDGE, SECRET_SETTING, with_secret  # noqa: F401

RUN_PASS = "the-pass-0123456789"
CONFIG_KEY = "cfg-key-12345678"


def run(worker, files, config=None, inputs=(), settings=SECRET_SETTING):
    def leave(output):
        (output / "measures.json").write_text("[]")
        for name, content in files.items():
            (output / name).write_bytes(content)

    worker["leave"] = leave
    celery_tasks.run_plugin.run(
        "pkg", "Plugin", "1.0", config or {}, list(inputs), settings, "eval", "ep"
    )
    return worker["uploads"]


def test_a_secret_in_a_csv_report_is_replaced(worker, with_secret):
    uploads = run(
        worker, {"report.csv": b"prompt,answer\nq,the key is judge-secret-value-1\n"}
    )
    assert uploads["report.csv"] == b"prompt,answer\nq,the key is [redacted]\n"


def test_a_secret_in_a_binary_file_is_replaced_and_the_rest_kept(worker, with_secret):
    content = b"\x89PNG\r\n\x1a\n" + JUDGE.encode() + b"\x00\xff"
    uploads = run(worker, {"chart.png": content})
    assert uploads["chart.png"] == b"\x89PNG\r\n\x1a\n[redacted]\x00\xff"


def test_a_key_shape_that_is_no_known_value_is_left_as_it_is(worker, with_secret):
    content = b"sk-proj-abcdefghijklmnopqrstu"
    uploads = run(worker, {"evidence.txt": content})
    assert uploads["evidence.txt"] == content


def test_the_runs_pass_is_replaced(worker, with_secret, platform):
    platform["mint"] = _Answer(201, {"pass": RUN_PASS, "expires_at": "2026-10-19T12:00:00Z"})
    uploads = run(
        worker,
        {"trace.txt": f"called with {RUN_PASS}\n".encode()},
        inputs=[_resource("target", TARGET)],
    )
    assert uploads["trace.txt"] == b"called with [redacted]\n"


def test_a_secret_named_configuration_value_is_replaced(worker, with_secret):
    uploads = run(
        worker,
        {"config-echo.json": f'{{"api_key": "{CONFIG_KEY}"}}'.encode()},
        config={"api_key": CONFIG_KEY},
    )
    assert uploads["config-echo.json"] == b'{"api_key": "[redacted]"}'


def test_a_secret_in_a_file_name_is_replaced(worker, with_secret):
    uploads = run(worker, {f"{JUDGE}.txt": b"nothing secret"})
    assert "[redacted].txt" in uploads
    assert not any(JUDGE in name for name in uploads)
    assert uploads["[redacted].txt"] == b"nothing secret"


@pytest.mark.parametrize(
    "content, known, expected",
    [
        (b"abc", ["short"], b"abc"),
        (b"a short value", ["short"], b"a short value"),
        (b"x judge-secret-value-1 y", [JUDGE], b"x [redacted] y"),
        (
            b"long-secret-value and long-secret",
            ["long-secret", "long-secret-value"],
            b"[redacted] and [redacted]",
        ),
    ],
)
def test_without_known_bytes_replaces_exact_values_of_eight_characters_or_more(
    content, known, expected
):
    assert scrub.without_known_bytes(content, known) == expected
