"""I2 — a plugin reaches a system through a door, not a credential.

A plugin is untrusted catalogue code. The worker must not hand it the worker's
own environment (database, registry or signing credentials); it gets a small
allowlist plus the run's own secrets, added by the caller.
"""

from aisc_eval.celery_tasks import (
    _PLUGIN_ENV_ALLOWLIST,
    plugin_child_env,
    plugin_run_env,
)

WORKER_ONLY = (
    "DJANGO_SECRET_KEY",
    "DATABASE_URL",
    "REDIS_URL",
    "PACKAGE_REGISTRY_PASSWORD",
    "KEYCLOAK_CLIENT_SECRET",
)


def test_plugin_environment_excludes_worker_credentials(monkeypatch):
    for name in WORKER_ONLY:
        monkeypatch.setenv(name, "worker-only-value")
    monkeypatch.setenv("PATH", "/usr/bin")

    env = plugin_child_env()

    for name in WORKER_ONLY:
        assert name not in env, f"{name} leaks into the plugin subprocess"
    assert env["PATH"] == "/usr/bin"


def test_plugin_environment_is_a_closed_allowlist(monkeypatch):
    monkeypatch.setenv("AISC_SOMETHING_UNEXPECTED", "value")

    env = plugin_child_env()

    assert "AISC_SOMETHING_UNEXPECTED" not in env


def test_plugin_environment_holds_nothing_but_the_allowlist(monkeypatch):
    """The child environment is exactly the allowlist — a name is present only
    if it is on it (SEC-01 #104)."""
    for name in ("PATH", "HOME", "LANG", "DJANGO_SECRET_KEY", "AISC_ANYTHING"):
        monkeypatch.setenv(name, "value")

    env = plugin_child_env()

    assert set(env) <= set(_PLUGIN_ENV_ALLOWLIST)


def test_plugin_environment_withholds_aws_and_service_secrets(monkeypatch):
    """A fake cloud key, service token and DB password in the worker's
    environment never reach the child (SEC-01 #104)."""
    worker_only = (
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID",
        "AISC_SERVICE_TOKEN",
        "POSTGRES_PASSWORD",
    )
    for name in worker_only:
        monkeypatch.setenv(name, "worker-only-secret")
    monkeypatch.setenv("PATH", "/usr/bin")

    env = plugin_child_env()

    assert env["PATH"] == "/usr/bin"
    assert not (set(worker_only) & set(env))


def test_run_local_door_withholds_the_targets_credentials(monkeypatch):
    """A run local to the assessed target must not carry that target's
    credentials, while a normal run keeps the existing behaviour (EXEC-01 #120)."""
    monkeypatch.setenv("PATH", "/usr/bin")
    # the target's run credentials are in the worker environment and the
    # dispatcher hands the child AISC_SECRET_ * entries under the SEC-01 rule.
    monkeypatch.setenv("AISC_SECRET_CONNECTION_TARGET_KEY", "target-secret")
    run_secrets = {"AISC_SECRET_CONNECTION_TARGET_KEY": "target-secret"}

    normal = plugin_run_env(run_secrets)
    assert normal["AISC_SECRET_CONNECTION_TARGET_KEY"] == "target-secret"
    assert normal["PATH"] == "/usr/bin"

    local = plugin_run_env(run_secrets, run_local=True)
    assert "AISC_SECRET_CONNECTION_TARGET_KEY" not in local
    assert local["PATH"] == "/usr/bin"
