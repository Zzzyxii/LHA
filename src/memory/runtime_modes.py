from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemoryMode(str, Enum):
    NONE = "none"
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    HYBRID = "hybrid"


_MODE_ALIASES = {
    "off": MemoryMode.NONE,
    "disabled": MemoryMode.NONE,
    "false": MemoryMode.NONE,
    "0": MemoryMode.NONE,
    "on": MemoryMode.EXPLICIT,
    "enabled": MemoryMode.EXPLICIT,
    "true": MemoryMode.EXPLICIT,
    "1": MemoryMode.EXPLICIT,
    "both": MemoryMode.HYBRID,
    "all": MemoryMode.HYBRID,
}


def normalize_memory_mode(value: object, *, default: MemoryMode = MemoryMode.EXPLICIT) -> MemoryMode:
    if isinstance(value, MemoryMode):
        return value
    normalized = str(value or "").strip().lower()
    if not normalized:
        return default
    alias = _MODE_ALIASES.get(normalized)
    if alias is not None:
        return alias
    try:
        return MemoryMode(normalized)
    except ValueError:
        return default


def mode_enables_explicit(mode: MemoryMode | str) -> bool:
    normalized = normalize_memory_mode(mode)
    return normalized in {MemoryMode.EXPLICIT, MemoryMode.HYBRID}


def mode_enables_implicit(mode: MemoryMode | str) -> bool:
    normalized = normalize_memory_mode(mode)
    return normalized in {MemoryMode.IMPLICIT, MemoryMode.HYBRID}


@dataclass(frozen=True)
class ModuleMemoryConfig:
    module_name: str
    mode: MemoryMode
    explicit_enabled: bool
    implicit_enabled: bool
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeMemoryConfig:
    mode: MemoryMode
    critic: ModuleMemoryConfig
    edit: ModuleMemoryConfig
    raw: dict[str, Any] = field(default_factory=dict)


def _module_mode(value: object, *, parent_mode: MemoryMode) -> MemoryMode:
    normalized = str(value or "").strip()
    if not normalized:
        return parent_mode
    return normalize_memory_mode(normalized, default=parent_mode)


def runtime_memory_config_from_policy(policy: dict | None = None) -> RuntimeMemoryConfig:
    payload = dict(policy or {})
    runtime = dict(payload.get("runtime", {}) or {})
    parent_mode = normalize_memory_mode(runtime.get("mode"), default=MemoryMode.EXPLICIT)
    critic_mode = _module_mode(runtime.get("critic_mode"), parent_mode=parent_mode)
    edit_mode = _module_mode(runtime.get("edit_mode"), parent_mode=parent_mode)
    return RuntimeMemoryConfig(
        mode=parent_mode,
        critic=ModuleMemoryConfig(
            module_name="critic",
            mode=critic_mode,
            explicit_enabled=mode_enables_explicit(critic_mode),
            implicit_enabled=mode_enables_implicit(critic_mode),
            raw=runtime,
        ),
        edit=ModuleMemoryConfig(
            module_name="edit",
            mode=edit_mode,
            explicit_enabled=mode_enables_explicit(edit_mode),
            implicit_enabled=mode_enables_implicit(edit_mode),
            raw=runtime,
        ),
        raw=runtime,
    )


def get_module_memory_config(
    runtime_config: RuntimeMemoryConfig | dict | None,
    *,
    module_name: str,
) -> ModuleMemoryConfig:
    if isinstance(runtime_config, dict) or runtime_config is None:
        runtime_config = runtime_memory_config_from_policy(runtime_config)
    normalized_name = str(module_name or "").strip().lower()
    if normalized_name == "critic":
        return runtime_config.critic
    if normalized_name == "edit":
        return runtime_config.edit
    return ModuleMemoryConfig(
        module_name=normalized_name or "global",
        mode=runtime_config.mode,
        explicit_enabled=mode_enables_explicit(runtime_config.mode),
        implicit_enabled=mode_enables_implicit(runtime_config.mode),
        raw=runtime_config.raw,
    )


def resolve_memory_mode(
    runtime_config: RuntimeMemoryConfig | dict | None,
    *,
    module_name: str = "",
    default: MemoryMode = MemoryMode.EXPLICIT,
) -> MemoryMode:
    if runtime_config is None:
        return default
    return get_module_memory_config(runtime_config, module_name=module_name).mode
