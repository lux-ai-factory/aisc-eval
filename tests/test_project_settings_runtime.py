import json

from aisc_eval.celery_tasks import build_project_settings, build_secret_environment
from aisc_plugin_interface import BaseEvaluationPlugin
from pydantic import BaseModel


FIRST_SETTING_PID = "5f0c2d1e-1111-4a6b-9c3d-000000000001"
SECOND_SETTING_PID = "5f0c2d1e-2222-4a6b-9c3d-000000000002"
THIRD_SETTING_PID = "5f0c2d1e-3333-4a6b-9c3d-000000000003"


class RuntimeConfig(BaseModel):
    value: int = 1


class RuntimePlugin(BaseEvaluationPlugin[RuntimeConfig]):
    def evaluate(self, config_data):
        return config_data


def test_plugin_description_defaults_to_empty_and_reads_class_attribute():
    assert RuntimePlugin().description == ""

    class DescribedPlugin(BaseEvaluationPlugin[RuntimeConfig]):
        description = "## Purpose\n\nEvaluates examples."

        def evaluate(self, config_data):
            return config_data

    assert DescribedPlugin().description == "## Purpose\n\nEvaluates examples."


def test_plugin_help_text_dedents_indented_triple_quoted_description():
    class IndentedPlugin(BaseEvaluationPlugin[RuntimeConfig]):
        description = """
        Reuses the same **project configs** as the Project Config plugin.

        - **API Key** - secret
        - **Base URL** - shared base URL
        """

        def evaluate(self, config_data):
            return config_data

    assert IndentedPlugin().help_text == (
        "Reuses the same **project configs** as the Project Config plugin.\n\n"
        "- **API Key** - secret\n"
        "- **Base URL** - shared base URL"
    )


def test_build_project_settings_excludes_secrets():
    settings = build_project_settings(
        [
            {
                "pid": FIRST_SETTING_PID,
                "key": "threshold",
                "plugin_config_key": "threshold",
                "category": "variables",
                "encrypted_value": "",
                "json_value": {"value": 0.8},
            },
            {
                "pid": SECOND_SETTING_PID,
                "key": "shape",
                "plugin_config_key": "shape",
                "category": "datashape",
                "encrypted_value": "",
                "json_value": {"features": []},
            },
            {
                "pid": THIRD_SETTING_PID,
                "key": "token",
                "plugin_config_key": "token",
                "category": "secrets",
                "encrypted_value": "ciphertext",
                "json_value": {},
            },
        ]
    )

    assert settings == {"threshold": 0.8, "shape": {"features": []}}
    assert "token" not in settings


def test_build_project_settings_keys_by_plugin_config_key_not_project_key():
    settings = build_project_settings(
        [
            {
                "pid": FIRST_SETTING_PID,
                "key": "project_threshold",
                "plugin_config_key": "threshold",
                "category": "variables",
                "encrypted_value": "",
                "json_value": {"value": 3},
            },
        ]
    )

    assert settings == {"threshold": 3}


def test_build_project_settings_excludes_secret_carrying_json_value():
    settings = build_project_settings(
        [
            {
                "pid": FIRST_SETTING_PID,
                "key": "api_key",
                "plugin_config_key": "api_key",
                "category": "secrets",
                "encrypted_value": "ciphertext",
                "json_value": {"value": "sk-not-real"},
            },
        ]
    )

    assert settings == {}
    assert "sk-not-real" not in json.dumps(settings)


def test_plugin_accessors_use_separate_channels(monkeypatch):
    plugin = RuntimePlugin()
    plugin._set_project_settings({"threshold": {"value": 1}})
    monkeypatch.setenv("AISC_SECRET_TOKEN", "secret-value")

    assert plugin.get_project_setting("threshold") == {"value": 1}
    assert plugin.require_project_setting("threshold") == {"value": 1}
    assert plugin.require_secret("token") == "secret-value"
    assert plugin.get_project_setting("token") is None


def test_build_secret_environment_uses_plugin_key_namespace(monkeypatch):
    monkeypatch.setattr(
        "aisc_eval.celery_tasks.decrypt_value", lambda value: f"decrypted:{value}"
    )

    assert build_secret_environment(
        [
            {
                "key": "open_ai_key",
                "category": "secrets",
                "encrypted_value": "ciphertext",
            },
        ]
    ) == {"AISC_SECRET_OPEN_AI_KEY": "decrypted:ciphertext"}
