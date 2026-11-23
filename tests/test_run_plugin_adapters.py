"""The run door's wiring inside run_plugin (ADR 0009, I2).

The full task creates a venv and installs, which a unit test cannot do; these
tests cover the parts that decide what the plugin is handed: the door spec, the
endpoint input file, the adapter install specs, and the data-adapter resolution
the runtime does in the venv.
"""
from unittest import mock

from aisc_eval import celery_tasks, plugin_runtime


def _endpoint_component(**overrides):
    base = {
        "name": "target",
        "component_type": "endpoint",
        "adapter": "openai",
        "adapter_version": "1",
        "adapter_config": {"model": "gpt-4o"},
        "url": "https://api.example/v1",
        "protocol": "openai-chat",
        "face": "openai-chat",
        "secret_encrypted_value": "cipher",
        "secret_headers": {"X-Api-Key": "cipher2"},
        "install_target": None,
        "allowed_hosts": ["10.0.0.5:11434"],
        "floor_hosts": [],
        "stack_hosts": ["postgres"],
        "config_hash": "abc",
        "value": {},
    }
    base.update(overrides)
    return base


def test_door_slots_decrypts_the_key_and_mints_a_token():
    with mock.patch.object(celery_tasks, "decrypt_value", side_effect=lambda v: f"plain:{v}"):
        slots, tokens = celery_tasks._door_slots([_endpoint_component()])
    assert set(slots) == {"target"}
    slot = slots["target"]
    assert slot["secret"] == "plain:cipher"
    assert slot["secret_headers"] == {"X-Api-Key": "plain:cipher2"}
    assert slot["allowed"] == ["10.0.0.5:11434"]
    assert slot["stack"] == ["postgres"]
    assert slot["config_hash"] == "abc"
    # the token is minted here and handed to the plugin under the name
    # `get_secret(llm.secret_key)` looks up (ADR 0009 D-16)
    assert set(tokens) == {"AISC_SECRET_DOOR_TARGET"}
    assert slot["token"] == tokens["AISC_SECRET_DOOR_TARGET"]
    assert len(tokens["AISC_SECRET_DOOR_TARGET"]) > 16


def test_the_door_token_is_reachable_by_get_secret():
    from aisc_plugin_interface.base_evaluation_plugin import BaseEvaluationPlugin

    _, tokens = celery_tasks._door_slots([_endpoint_component()])
    env_name = BaseEvaluationPlugin._secret_environment_name("door_target")
    assert env_name in tokens


def test_the_door_spec_carries_no_token_for_a_data_input():
    slots, tokens = celery_tasks._door_slots([
        {"name": "data", "component_type": "dataset", "adapter": "csv", "install_target": None},
    ])
    assert slots == {}
    assert tokens == {}


def test_endpoint_input_file_is_an_llm_config_with_the_door_url_and_token():
    payload = celery_tasks._endpoint_input_file(
        _endpoint_component(), "http://127.0.0.1:9/target", "door_target"
    )
    assert payload == {
        "endpoint_url": "http://127.0.0.1:9/target",
        "secret_key": "door_target",
        "model": "gpt-4o",
    }


def test_endpoint_input_file_uses_the_endpoint_shape_for_a_non_llm_face():
    payload = celery_tasks._endpoint_input_file(
        _endpoint_component(face="agent", protocol="agent"),
        "http://127.0.0.1:9/target", "door_target",
    )
    assert payload == {
        "url": "http://127.0.0.1:9/target",
        "protocol": "agent",
        "model": "gpt-4o",
        "token_key": "door_target",
    }


def test_adapter_install_specs_are_deduplicated():
    components = [
        {"install_target": {"package": "aisc-plugin-x", "version": "1.0", "source": "registry"}},
        {"install_target": {"package": "aisc-plugin-x", "version": "1.0", "source": "registry"}},
        {"install_target": {"package": "aisc-plugin-y", "version": "2.0", "source": "registry"}},
        {"install_target": None},
    ]
    loader = mock.Mock(discovered_adapters={})
    specs = celery_tasks._adapter_install_specs(components, loader)
    assert specs == ["aisc-plugin-x==1.0", "aisc-plugin-y==2.0"]


class _Slot:
    def __init__(self, name, accepts):
        self.name = name
        self.accepts = accepts


class _FakePlugin:
    _input_definitions = [_Slot("data", ["records"])]


def test_resolve_data_adapter_picks_the_produce_the_slot_accepts():
    resolved = plugin_runtime.resolve_data_adapter(
        _FakePlugin(), "data", {"adapter": "csv", "config": {}, "module": "", "class": ""}
    )
    assert resolved is not None
    adapter, produce = resolved
    assert adapter.id == "csv"
    assert produce == "records"


def test_resolve_data_adapter_returns_none_for_an_unknown_adapter():
    assert plugin_runtime.resolve_data_adapter(
        _FakePlugin(), "data", {"adapter": "nope", "config": {}, "module": "", "class": ""}
    ) is None
