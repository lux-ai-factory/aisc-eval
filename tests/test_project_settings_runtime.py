from aisc_eval.celery_tasks import build_project_settings, build_secret_environment
from aisc_plugin_interface import BaseEvaluationPlugin, SettingCategory, SettingDefinition
from pydantic import BaseModel


class RuntimeConfig(BaseModel):
    value: int = 1


class RuntimePlugin(BaseEvaluationPlugin[RuntimeConfig]):
    def evaluate(self, config_data):
        return config_data


def test_build_project_settings_excludes_secrets():
    settings = build_project_settings([
        {"key": "threshold", "category": "general", "json_value": {"value": 0.8}},
        {"key": "shape", "category": "datashape", "json_value": {"features": []}},
        {"key": "token", "category": "secrets", "encrypted_value": "ciphertext"},
    ])

    assert settings == {"threshold": 0.8, "shape": {"features": []}}


def test_plugin_accessors_use_separate_channels(monkeypatch):
    plugin = RuntimePlugin()
    plugin._set_project_settings({"threshold": {"value": 1}})
    monkeypatch.setenv("AISC_SECRET_TOKEN", "secret-value")

    assert plugin.get_project_setting("threshold") == {"value": 1}
    assert plugin.require_project_setting("threshold") == {"value": 1}
    assert plugin.require_secret("token") == "secret-value"
    assert plugin.get_project_setting("token") is None


def test_build_secret_environment_uses_plugin_key_namespace(monkeypatch):
    monkeypatch.setattr("aisc_eval.celery_tasks.decrypt_value", lambda value: f"decrypted:{value}")

    definition = SettingDefinition(
        key="openai",
        name="OpenAI API Key",
        category=SettingCategory.SECRETS,
    )
    assert build_secret_environment([
        {"key": "open_ai_key", "category": "secrets", "encrypted_value": "ciphertext"},
    ], [definition]) == {"AISC_SECRET_OPENAI": "decrypted:ciphertext"}
