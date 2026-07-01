"""Local JSON-backed settings memory for the desktop app."""

from __future__ import annotations

import json
from pathlib import Path


MEMORY_SCHEMA_VERSION = 2
MEMORY_PATH = Path.home() / ".chromatopy_gui_memory.json"


def default_settings_memory() -> dict:
    return {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "compound_histories": [],
        "integration_configuration": {},
        "theme": "light",
    }


def load_settings_memory() -> dict:
    if not MEMORY_PATH.exists():
        return default_settings_memory()
    try:
        memory = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return default_settings_memory()
    defaults = default_settings_memory()
    if not isinstance(memory, dict):
        return defaults
    merged = defaults | memory
    merged["compound_histories"] = memory.get("compound_histories", [])
    merged["integration_configuration"] = memory.get("integration_configuration", {})
    return merged


def save_settings_memory(memory: dict) -> None:
    current = load_settings_memory()
    current.update(memory)
    current["schema_version"] = MEMORY_SCHEMA_VERSION
    MEMORY_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")

def load_theme() -> str:
    memory = load_settings_memory()
    theme = memory.get("theme", "light")
    return theme if theme in {"light", "dark"} else "light"

def save_theme(theme: str) -> None:
    if theme not in {"light", "dark"}:
        theme = "light"
    save_settings_memory({"theme": theme})

def list_compound_histories() -> list[list[str]]:
    memory = load_settings_memory()
    histories = memory.get("compound_histories", [])
    return [entry for entry in histories if isinstance(entry, list) and entry]


def remember_compound_history(compounds: list[str]) -> None:
    cleaned = [compound.strip() for compound in compounds if compound.strip()]
    if not cleaned:
        return
    histories = list_compound_histories()
    histories = [entry for entry in histories if entry != cleaned]
    histories.insert(0, cleaned)
    save_settings_memory({"compound_histories": histories[:20]})


def delete_compound_history(compounds: list[str]) -> None:
    cleaned = [compound.strip() for compound in compounds if compound.strip()]
    histories = [entry for entry in list_compound_histories() if entry != cleaned]
    save_settings_memory({"compound_histories": histories})


def load_integration_configuration_memory() -> dict:
    memory = load_settings_memory()
    config = memory.get("integration_configuration", {})
    return config if isinstance(config, dict) else {}


def save_integration_configuration_memory(config: dict) -> None:
    save_settings_memory({"integration_configuration": config})
