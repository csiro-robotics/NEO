"""Safe serialization helpers for experiment dataclasses."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml


ConfigT = TypeVar("ConfigT")


def _to_primitive(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _to_primitive(getattr(value, field.name))
            for field in fields(value)
            if not field.name.startswith("_")
        }
    if isinstance(value, dict):
        return {str(key): _to_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_primitive(item) for item in value]
    if isinstance(value, type) or callable(value):
        return None
    return value


def save_experiment_config(config: Any, path: Path) -> None:
    """Write a dataclass config as portable, plain YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_to_primitive(config), sort_keys=False), encoding="utf8")


def _update_dataclass(instance: Any, values: dict[str, Any], prefix: str = "") -> None:
    known_fields = {field.name for field in fields(instance)}
    unknown = set(values) - known_fields
    if unknown:
        location = prefix or type(instance).__name__
        raise ValueError(f"Unknown configuration keys in {location}: {sorted(unknown)}")

    for name, value in values.items():
        current = getattr(instance, name)
        location = f"{prefix}.{name}" if prefix else name
        if is_dataclass(current) and isinstance(value, dict):
            _update_dataclass(current, value, location)
        elif isinstance(current, Path) or name.endswith(("_root", "_path")) and isinstance(value, str):
            setattr(instance, name, Path(value))
        else:
            setattr(instance, name, value)


def load_experiment_config(config: ConfigT, path: Path) -> ConfigT:
    """Load portable YAML values into an existing dataclass config."""
    values = yaml.safe_load(path.read_text(encoding="utf8"))
    if not isinstance(values, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    _update_dataclass(config, values)
    return config
