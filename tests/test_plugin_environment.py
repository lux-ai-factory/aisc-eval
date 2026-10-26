"""A plugin is code from a catalogue, run by the worker. It gets the variables it needs
(system basics, proxies, the platform's address, its run's pass, its project's secrets) and none
of the worker's own credentials: with the broker URL and the internal key a plugin could read
other runs' queued tasks, lift their run tickets and call the backend as another project."""

from aisc_eval import celery_tasks

from test_plugin_process import worker  # noqa: F401  (the fixture)

WORKER_ENV = {
    "PATH": "/usr/bin",
    "HOME": "/root",
    "LANG": "C.UTF-8",
    "HTTPS_PROXY": "http://proxy:3128",
    "no_proxy": "localhost",
    "PLATFORM_URL": "http://platform:8000",
    "PLATFORM_CONNECTIONS_TOKEN": "conn-token",
    "CONNECTIONS_ALLOWED_HOSTS": "example.org",
    "CELERY_BROKER_URL": "amqp://user:pw@rabbitmq:5672//",
    "REDIS_BACKEND_URL": "redis://:pw@redis:6379/0",
    "INTERNAL_API_KEY": "internal",
    "DJANGO_SECRET_KEY": "django",
    "PACKAGE_REGISTRY_PASSWORD": "devpi",
    "PACKAGE_REGISTRY_USER": "root",
    "API_URL": "http://aisc-backend:8000",
    "VIRTUAL_ENV": "/app/.venv",
}


def test_the_worker_credentials_never_reach_a_plugin():
    env = celery_tasks.plugin_environment(WORKER_ENV, {})
    for name in (
        "CELERY_BROKER_URL",
        "REDIS_BACKEND_URL",
        "INTERNAL_API_KEY",
        "DJANGO_SECRET_KEY",
        "PACKAGE_REGISTRY_PASSWORD",
        "PACKAGE_REGISTRY_USER",
        "API_URL",
        "VIRTUAL_ENV",
    ):
        assert name not in env, name


def test_a_plugin_keeps_what_it_needs():
    env = celery_tasks.plugin_environment(WORKER_ENV, {})
    for name in (
        "PATH",
        "HOME",
        "LANG",
        "HTTPS_PROXY",
        "no_proxy",
        "PLATFORM_URL",
        "CONNECTIONS_ALLOWED_HOSTS",
    ):
        assert env[name] == WORKER_ENV[name], name


def test_the_stack_connections_token_never_reaches_a_plugin():
    # it opens every project's connections; a plugin gets its run's pass instead
    assert "PLATFORM_CONNECTIONS_TOKEN" not in celery_tasks.plugin_environment(
        WORKER_ENV, {}
    )
    env = celery_tasks.plugin_environment(WORKER_ENV, {}, run_pass="the-pass")
    assert "PLATFORM_CONNECTIONS_TOKEN" not in env
    assert env["PLATFORM_RUN_PASS"] == "the-pass"


def test_a_run_without_a_pass_gets_none_even_if_the_worker_has_one():
    env = celery_tasks.plugin_environment(
        {**WORKER_ENV, "PLATFORM_RUN_PASS": "stale"}, {}
    )
    assert "PLATFORM_RUN_PASS" not in env


def test_project_secrets_are_added_and_the_openai_placeholder_kept():
    secret = celery_tasks.SECRET_ENV_PREFIX + "TARGET_KEY"
    env = celery_tasks.plugin_environment(WORKER_ENV, {secret: "s"})
    assert env[secret] == "s"
    assert env["API_KEY_OPENAI"]  # LangBiTe builds an OpenAI client at init


def test_a_real_openai_key_from_the_project_wins_over_the_placeholder():
    env = celery_tasks.plugin_environment(WORKER_ENV, {"API_KEY_OPENAI": "sk-real"})
    assert env["API_KEY_OPENAI"] == "sk-real"


WORKER_KEYS = {
    "AISC_TARGET_API_KEY": "sk-target-worker-0123456789",
    "OPENAI_API_KEY": "sk-openai-worker-0123456789",
    "API_KEY_OPENAI": "sk-openai2-worker-0123456789",
}


def test_the_workers_own_model_keys_never_reach_a_plugin():
    env = celery_tasks.plugin_environment({**WORKER_ENV, **WORKER_KEYS}, {})
    assert "AISC_TARGET_API_KEY" not in env
    assert "OPENAI_API_KEY" not in env
    assert env["API_KEY_OPENAI"] == celery_tasks.OPENAI_PLACEHOLDER
    for value in WORKER_KEYS.values():
        assert value not in env.values()


def test_the_model_addresses_still_reach_a_plugin():
    env = celery_tasks.plugin_environment(
        {
            **WORKER_ENV,
            "AISC_TARGET_BASE_URL": "http://t/v1",
            "OPENAI_BASE_URL": "http://o/v1",
        },
        {},
    )
    assert env["AISC_TARGET_BASE_URL"] == "http://t/v1"
    assert env["OPENAI_BASE_URL"] == "http://o/v1"


def test_no_model_key_name_is_passed_through():
    for name in ("AISC_TARGET_API_KEY", "OPENAI_API_KEY", "API_KEY_OPENAI"):
        assert name not in celery_tasks.PLUGIN_ENV_NAMES, name


def test_a_run_hands_its_plugin_no_openai_key_of_the_worker(worker, monkeypatch):  # noqa: F811
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-worker-0123456789")
    celery_tasks.run_plugin.run("pkg", "Plugin", "1.0", {}, [], [], "eval", "ep")
    [(_argv, kwargs)] = worker["popen"]
    assert "OPENAI_API_KEY" not in kwargs["env"]
    assert "sk-openai-worker-0123456789" not in kwargs["env"].values()
