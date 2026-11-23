#!/usr/bin/env python3
"""
Runtime executor for BaseEvaluationPlugin instances.

This script is called by run_plugin in celery_tasks.py via UV to execute plugins in an isolated environment.

Expected directory structure:
  working-dir/
    ├── config.json      # Plugin configuration
    ├── input/           # Input files
    └── output/          # Output files (created if doesn't exist)
"""

import importlib
import importlib.util
import json
import sys
from functools import partial
from pathlib import Path

from aisc_plugin_interface import TaskProgress
from aisc_plugin_interface.base_evaluation_plugin import BaseEvaluationPlugin


def write_artifact_file(name: str, content: bytes, output_dir: Path):
    """Callback that writes artifact files to the output folder."""
    artifact_path = output_dir / name
    artifact_path.write_bytes(content)
    print(f"📄 Artifact saved: {artifact_path}")


def print_progress(task_progress: TaskProgress):
    """
    Callback that streams progress updates to the parent Celery process.
    It serializes the Pydantic model to JSON and prints it to stdout.
    The parent process intercepts this JSON string in real-time.
    """
    progress_json = task_progress.model_dump_json()
    print(progress_json, flush=True)


def load_plugin(plugin_spec: str):
    """Load a BaseEvaluationPlugin subclass from an installed package.

    Args:
        plugin_spec: Format 'package-name:PluginClass'
    """
    parts = plugin_spec.split(":")
    if len(parts) != 2:
        raise ValueError(
            f"Registry plugin must be in format 'package-name:PluginClass', got: {plugin_spec}"
        )

    module_name, class_name = parts
    module_name = module_name.replace("-", "_")

    module = importlib.import_module(module_name)
    plugin_class = getattr(module, class_name)

    if not issubclass(plugin_class, BaseEvaluationPlugin):
        raise ValueError(f"{class_name} is not a subclass of BaseEvaluationPlugin")

    return plugin_class


def resolve_data_adapter(plugin, name: str, spec: dict):
    """The registry data adapter for a slot, and the value type to produce.

    A builtin adapter is resolved by id; an adapter a package ships is imported
    from its module. The produce type is the first the adapter makes that the slot
    accepts (ADR 0009). Returns ``(adapter, produce)`` or ``None``.
    """
    from aisc_plugin_interface.adapters import BUILTIN_ADAPTERS, DataAdapter

    module_name = spec.get("module") or ""
    if module_name:
        cls = getattr(importlib.import_module(module_name), spec["class"])
    else:
        cls = next(
            (c for c in BUILTIN_ADAPTERS if getattr(c, "id", "") == spec.get("adapter")),
            None,
        )
    if cls is None or not issubclass(cls, DataAdapter):
        return None
    adapter = cls(spec.get("config") or {})
    slot = next((d for d in plugin._input_definitions if d.name == name), None)
    accepts = set(getattr(slot, "accepts", None) or [])
    produces = tuple(getattr(adapter, "produces", ()) or ())
    produce = next((p for p in produces if p in accepts), produces[0] if produces else "")
    return adapter, produce


def main():
    working_dir = Path.cwd()

    try:
        config_path = working_dir / "config.json"
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)

        output_dir = working_dir / "output"

        with open(config_path) as f:
            config = json.load(f)

        plugin_source = config.get("plugin_source")
        if not plugin_source:
            print("❌ Missing 'plugin_source' in config.json", file=sys.stderr)
            sys.exit(1)

        print(f"📦 Loading plugin: {plugin_source}")
        plugin_class = load_plugin(plugin_source)

        print(f"🔌 Instantiating plugin: {plugin_class.__name__}")
        plugin = plugin_class()

        plugin._set_artifact_callback(
            partial(write_artifact_file, output_dir=output_dir)
        )

        plugin._set_progress_callback(print_progress)
        if hasattr(plugin, "_set_project_settings"):
            plugin._set_project_settings(config.get("project_settings", {}))

        input_mapping = config.get("input_mapping", {})
        input_adapters = config.get("input_adapters", {})
        input_kinds = config.get("input_kinds", {})
        for name, path in input_mapping.items():
            # Resolve input paths relative to working directory
            file_path = Path(path)
            if not file_path.is_absolute():
                file_path = working_dir / "input" / file_path

            if not file_path.exists():
                print(f"⚠️  Input file not found: {file_path}", file=sys.stderr)
                continue

            print(f"📥 Loading input '{name}' from {path}")
            raw = file_path.read_bytes()
            spec = input_adapters.get(name)
            resolved = resolve_data_adapter(plugin, name, spec) if spec else None
            if resolved is not None:
                # A data adapter (standard, or one the plugin ships) produces the
                # value the slot declared it accepts (ADR 0009 D-22).
                adapter, produce = resolved
                plugin.set_input_content(name, adapter.load(raw, produce))
            elif input_kinds.get(name) in ("endpoint", "datashape", "resource"):
                # The runtime wrote a JSON document (a config, a datashape, an
                # endpoint); hand the plugin the parsed value.
                plugin.set_input_content(name, json.loads(raw))
            else:
                plugin.set_input_content(name, raw)

        print("🚀 Running evaluation...")
        plugin_config = config.get("plugin_config", {})
        output = plugin.evaluate(plugin_config)

        print("📊 Exporting metrics...")
        metrics = [m.model_dump(mode="json") for m in plugin.export_metrics(output)]

        measures_file = output_dir / "measures.json"
        with open(measures_file, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"✅ Metrics saved to: {measures_file}")
        print(f"📈 Generated {len(metrics)} measures")

    except Exception as e:
        print(f"❌ Plugin execution failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
