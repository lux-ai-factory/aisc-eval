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

import argparse
import importlib
import importlib.util
import json
import sys
from functools import partial
from pathlib import Path

from vera_plugin_interface.base_evaluation_plugin import BaseEvaluationPlugin


def artifact_callback(name: str, content: bytes, output_dir: Path):
    """Callback that writes artifact files to the output folder."""
    artifact_path = output_dir / name
    artifact_path.write_bytes(content)
    print(f"📄 Artifact saved: {artifact_path}")


def load_plugin_from_registry(plugin_spec: str):
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
        raise ValueError(
            f"{class_name} is not a subclass of BaseEvaluationPlugin"
        )

    return plugin_class


def load_plugin_from_module(plugin_name: str):
    """Load a BaseEvaluationPlugin subclass from an installed module.

    This scans all installed packages to find a plugin class with the matching display_name.
    Works for both local plugins (installed via uv in the workspace) and registry plugins.

    Args:
        plugin_name: The display name of the plugin class
    """
    import pkgutil
    import inspect as insp

    # Scan all available packages
    # UV has installed the plugin package in the venv's site-packages
    for importer, modname, ispkg in pkgutil.iter_modules():
        # Skip obviously non-plugin packages
        if modname.startswith('_') or modname in ('pip', 'setuptools', 'pkg_resources'):
            continue

        try:
            module = importlib.import_module(modname)

            # Check if this module exports plugin classes in __all__
            if hasattr(module, '__all__'):
                for class_name in module.__all__:
                    try:
                        obj = getattr(module, class_name)
                        if (
                            isinstance(obj, type)
                            and issubclass(obj, BaseEvaluationPlugin)
                            and obj != BaseEvaluationPlugin
                            and obj.display_name == plugin_name
                        ):
                            return obj
                    except Exception:
                        continue

            # Also check all class members
            for name, obj in insp.getmembers(module, insp.isclass):
                try:
                    if (
                        issubclass(obj, BaseEvaluationPlugin)
                        and obj != BaseEvaluationPlugin
                        and obj.display_name == plugin_name
                    ):
                        return obj
                except Exception:
                    continue

        except Exception:
            continue

    raise ValueError(f"Could not find plugin class with display_name '{plugin_name}'")


def main():
    parser = argparse.ArgumentParser(
        description="Execute a BaseEvaluationPlugin instance"
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        help="Working directory containing config.json, input/, and output/ folders (defaults to cwd)"
    )

    args = parser.parse_args()
    working_dir = args.working_dir.resolve() if args.working_dir else Path.cwd()

    try:
        config_path = working_dir / "config.json"
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)

        output_dir = working_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(config_path) as f:
            config = json.load(f)

        plugin_source = config.get("plugin_source")
        if not plugin_source:
            print("❌ Missing 'plugin_source' in config.json", file=sys.stderr)
            sys.exit(1)

        is_registry = config.get("is_registry", False)

        if is_registry:
            print(f"📦 Loading plugin from registry: {plugin_source}")
            plugin_class = load_plugin_from_registry(plugin_source)
        else:
            print(f"📦 Loading local plugin: {plugin_source}")
            plugin_class = load_plugin_from_module(plugin_source)

        print(f"🔌 Instantiating plugin: {plugin_class.__name__}")
        plugin = plugin_class()

        plugin._set_artifact_callback(partial(artifact_callback, output_dir=output_dir))

        input_mapping = config.get("input_mapping", {})
        for name, path in input_mapping.items():
            # Resolve input paths relative to working directory
            file_path = Path(path)
            if not file_path.is_absolute():
                file_path = working_dir / "input" / file_path

            if file_path.exists():
                print(f"📥 Loading input '{name}' from {path}")
                plugin.set_input_content(name, file_path.read_bytes())
            else:
                print(f"⚠️  Input file not found: {file_path}", file=sys.stderr)

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
