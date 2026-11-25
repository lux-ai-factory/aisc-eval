"""The worker decrypts plugin settings with the engine's own key, ENGINE_SECRETS_KEY, and still reads values
stored under the key derived from DJANGO_SECRET_KEY. A plugin never gets either key.
"""

import base64
import hashlib

import pytest
from cryptography.fernet import Fernet, InvalidToken

from aisc_eval import celery_tasks
from aisc_eval.utils import encryption

NEW = Fernet.generate_key().decode()


def _legacy(secret):
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest()))


def test_the_worker_decrypts_what_the_backend_encrypts_with_the_engine_key(monkeypatch):
    monkeypatch.setenv("ENGINE_SECRETS_KEY", NEW)
    monkeypatch.setenv("DJANGO_SECRET_KEY", "django-key-for-the-test")
    token = Fernet(NEW.encode()).encrypt(b"value-for-the-test").decode()
    try:
        assert encryption.decrypt_value(token) == "value-for-the-test"
    except InvalidToken:
        pytest.fail("the worker cannot read a value encrypted with ENGINE_SECRETS_KEY")


def test_the_worker_still_reads_a_value_stored_under_the_django_key(monkeypatch):
    monkeypatch.setenv("ENGINE_SECRETS_KEY", NEW)
    monkeypatch.setenv("DJANGO_SECRET_KEY", "django-key-for-the-test")
    token = _legacy("django-key-for-the-test").encrypt(b"old").decode()
    assert encryption.decrypt_value(token) == "old"


def test_a_plugin_never_gets_the_engine_secrets_key():
    got = celery_tasks.plugin_environment(
        {"ENGINE_SECRETS_KEY": NEW, "PATH": "/bin"}, {}
    )
    assert "ENGINE_SECRETS_KEY" not in got
