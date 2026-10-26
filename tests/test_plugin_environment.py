"""A plugin is code from a catalogue, run by the worker. It gets the variables it needs
(system basics, proxies, the platform's address, its run's pass, its project's secrets) and none
of the worker's own credentials: with the broker URL and the internal key a plugin could read
other runs' queued tasks, lift their run tickets and call the backend as another project."""

from aisc_eval import celery_tasks

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
