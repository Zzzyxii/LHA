from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MemoryPrototype:
    prototype_id: str = ""
    scope: str = "local"
    polarity: str = "positive"
    scale: str = "short"
    vector: list[float] = field(default_factory=list)
    weight: float = 0.0
    confidence: float = 0.0
    sample_id: str = ""
    attempt_index: int = 0
    variant: str = ""
    style_family: str = ""
    archetype_id: str = ""
    carriers: list[str] = field(default_factory=list)
    anchors: list[str] = field(default_factory=list)
    failure_tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict | None) -> "MemoryPrototype":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            prototype_id=str(payload.get("prototype_id", "") or ""),
            scope=str(payload.get("scope", "local") or "local"),
            polarity=str(payload.get("polarity", "positive") or "positive"),
            scale=str(payload.get("scale", "short") or "short"),
            vector=[float(value) for value in (payload.get("vector") or []) if isinstance(value, (int, float))],
            weight=float(payload.get("weight", 0.0) or 0.0),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            sample_id=str(payload.get("sample_id", "") or ""),
            attempt_index=int(payload.get("attempt_index", 0) or 0),
            variant=str(payload.get("variant", "") or ""),
            style_family=str(payload.get("style_family", "") or ""),
            archetype_id=str(payload.get("archetype_id", "") or ""),
            carriers=[str(value) for value in (payload.get("carriers") or []) if str(value).strip()],
            anchors=[str(value) for value in (payload.get("anchors") or []) if str(value).strip()],
            failure_tags=[str(value) for value in (payload.get("failure_tags") or []) if str(value).strip()],
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass
class PrototypeScaleState:
    short: list[MemoryPrototype] = field(default_factory=list)
    mid: list[MemoryPrototype] = field(default_factory=list)
    long: list[MemoryPrototype] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "short": [item.to_dict() for item in self.short],
            "mid": [item.to_dict() for item in self.mid],
            "long": [item.to_dict() for item in self.long],
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "PrototypeScaleState":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            short=[MemoryPrototype.from_dict(item) for item in (payload.get("short") or []) if isinstance(item, dict)],
            mid=[MemoryPrototype.from_dict(item) for item in (payload.get("mid") or []) if isinstance(item, dict)],
            long=[MemoryPrototype.from_dict(item) for item in (payload.get("long") or []) if isinstance(item, dict)],
        )


@dataclass
class LocalImplicitState:
    positive: PrototypeScaleState = field(default_factory=PrototypeScaleState)
    negative: PrototypeScaleState = field(default_factory=PrototypeScaleState)

    def to_dict(self) -> dict[str, Any]:
        return {
            "positive": self.positive.to_dict(),
            "negative": self.negative.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "LocalImplicitState":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            positive=PrototypeScaleState.from_dict(payload.get("positive", {})),
            negative=PrototypeScaleState.from_dict(payload.get("negative", {})),
        )


@dataclass
class GlobalImplicitState:
    positive: PrototypeScaleState = field(default_factory=PrototypeScaleState)
    negative: PrototypeScaleState = field(default_factory=PrototypeScaleState)

    def to_dict(self) -> dict[str, Any]:
        return {
            "positive": self.positive.to_dict(),
            "negative": self.negative.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "GlobalImplicitState":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            positive=PrototypeScaleState.from_dict(payload.get("positive", {})),
            negative=PrototypeScaleState.from_dict(payload.get("negative", {})),
        )


@dataclass
class ImplicitStateSnapshot:
    local: LocalImplicitState = field(default_factory=LocalImplicitState)
    global_state: GlobalImplicitState = field(default_factory=GlobalImplicitState)
    visual_references: list[dict[str, Any]] = field(default_factory=list)
    routing_trace: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "local": self.local.to_dict(),
            "global": self.global_state.to_dict(),
            "visual_references": list(self.visual_references),
            "routing_trace": dict(self.routing_trace),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "ImplicitStateSnapshot":
        if not isinstance(payload, dict):
            return cls()
        visual_references = [dict(item) for item in (payload.get("visual_references") or []) if isinstance(item, dict)]
        return cls(
            local=LocalImplicitState.from_dict(payload.get("local", {})),
            global_state=GlobalImplicitState.from_dict(payload.get("global", {})),
            visual_references=visual_references,
            routing_trace=dict(payload.get("routing_trace", {}) or {}),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass
class MemoryQuery:
    scope: str = "local"
    sample_id: str = ""
    variant: str = ""
    style_family: str = ""
    archetype_id: str = ""
    carriers: list[str] = field(default_factory=list)
    anchors: list[str] = field(default_factory=list)
    failure_tags: list[str] = field(default_factory=list)
    low_score_dimensions: list[str] = field(default_factory=list)
    region_edits: list[str] = field(default_factory=list)
    vector: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict | None) -> "MemoryQuery":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            scope=str(payload.get("scope", "local") or "local"),
            sample_id=str(payload.get("sample_id", "") or ""),
            variant=str(payload.get("variant", "") or ""),
            style_family=str(payload.get("style_family", "") or ""),
            archetype_id=str(payload.get("archetype_id", "") or ""),
            carriers=[str(value) for value in (payload.get("carriers") or []) if str(value).strip()],
            anchors=[str(value) for value in (payload.get("anchors") or []) if str(value).strip()],
            failure_tags=[str(value) for value in (payload.get("failure_tags") or []) if str(value).strip()],
            low_score_dimensions=[str(value) for value in (payload.get("low_score_dimensions") or []) if str(value).strip()],
            region_edits=[str(value) for value in (payload.get("region_edits") or []) if str(value).strip()],
            vector=[float(value) for value in (payload.get("vector") or []) if isinstance(value, (int, float))],
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass
class ImplicitBias:
    scope: str = "local"
    preferred_skills: dict[str, float] = field(default_factory=dict)
    penalized_skills: dict[str, float] = field(default_factory=dict)
    preferred_carriers: dict[str, float] = field(default_factory=dict)
    penalized_carriers: dict[str, float] = field(default_factory=dict)
    candidate_bias: dict[str, float] = field(default_factory=dict)
    variant_bias: dict[str, float] = field(default_factory=dict)
    edit_controls: dict[str, Any] = field(default_factory=dict)
    routing_bias: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "preferred_skills": dict(self.preferred_skills),
            "penalized_skills": dict(self.penalized_skills),
            "preferred_carriers": dict(self.preferred_carriers),
            "penalized_carriers": dict(self.penalized_carriers),
            "candidate_bias": dict(self.candidate_bias),
            "variant_bias": dict(self.variant_bias),
            "edit_controls": dict(self.edit_controls),
            "routing_bias": dict(self.routing_bias),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "ImplicitBias":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            scope=str(payload.get("scope", "local") or "local"),
            preferred_skills={str(key): float(value) for key, value in dict(payload.get("preferred_skills", {}) or {}).items()},
            penalized_skills={str(key): float(value) for key, value in dict(payload.get("penalized_skills", {}) or {}).items()},
            preferred_carriers={str(key): float(value) for key, value in dict(payload.get("preferred_carriers", {}) or {}).items()},
            penalized_carriers={str(key): float(value) for key, value in dict(payload.get("penalized_carriers", {}) or {}).items()},
            candidate_bias={str(key): float(value) for key, value in dict(payload.get("candidate_bias", {}) or {}).items()},
            variant_bias={str(key): float(value) for key, value in dict(payload.get("variant_bias", {}) or {}).items()},
            edit_controls=dict(payload.get("edit_controls", {}) or {}),
            routing_bias={str(key): float(value) for key, value in dict(payload.get("routing_bias", {}) or {}).items()},
            metadata=dict(payload.get("metadata", {}) or {}),
        )
