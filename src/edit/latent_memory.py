from __future__ import annotations

from copy import deepcopy
import math
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clean_vector(values: Any, *, max_dim: int = 0) -> list[float]:
    if not isinstance(values, list):
        return []
    out: list[float] = []
    for value in values:
        if not isinstance(value, (int, float)):
            continue
        number = float(value)
        if not math.isfinite(number):
            continue
        out.append(number)
        if max_dim > 0 and len(out) >= max_dim:
            break
    return out


def _vector_norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = _vector_norm(left)
    right_norm = _vector_norm(right)
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    dot = sum(left[i] * right[i] for i in range(len(left)))
    return dot / (left_norm * right_norm)


def _average_vectors(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    if dim <= 0:
        return []
    total = [0.0 for _ in range(dim)]
    count = 0
    for vector in vectors:
        if len(vector) != dim:
            continue
        count += 1
        for index, value in enumerate(vector):
            total[index] += value
    if count <= 0:
        return []
    return [value / count for value in total]


def _ema_vector(previous: list[float], current: list[float], *, decay: float) -> list[float]:
    if not previous:
        return list(current)
    if len(previous) != len(current):
        return list(current)
    alpha = max(0.0, min(1.0, float(decay)))
    return [alpha * previous[index] + (1.0 - alpha) * current[index] for index in range(len(previous))]


def _pool_hidden_vector(hidden_tensor) -> list[float]:
    import torch

    if not torch.is_tensor(hidden_tensor):
        return []
    hidden = hidden_tensor.detach()
    if hidden.ndim < 2:
        return []
    reduce_dims = tuple(range(hidden.ndim - 1))
    pooled = hidden.float().mean(dim=reduce_dims)
    if pooled.ndim != 1:
        pooled = pooled.reshape(-1)
    return pooled.cpu().tolist()


def _clean_entry(entry: dict | None, *, max_vector_dim: int = 0) -> dict:
    payload = dict(entry or {})
    vector = _clean_vector(payload.get("vector"), max_dim=max_vector_dim)
    if not vector:
        return {}
    return {
        "vector": vector,
        "weight": max(0.0, _safe_float(payload.get("weight", 1.0), 1.0)),
        "scope": str(payload.get("scope", "") or ""),
        "scale": str(payload.get("scale", "") or ""),
        "attempt_index": _safe_int(payload.get("attempt_index", 0), 0),
    }


class EditLatentMemorySession:
    def __init__(
        self,
        *,
        pipe: Any,
        controls: dict | None,
        backend: str,
        variant_id: str,
    ) -> None:
        self._pipe = pipe
        self._backend = str(backend or "")
        self._variant_id = str(variant_id or "")
        self._controls = dict(controls or {})
        self._latent = dict(self._controls.get("latent_memory") or {})
        self._handles: list[Any] = []
        self._capture_state: dict[tuple[str, int], dict[str, Any]] = {}
        self._retrieval_trace: list[dict[str, Any]] = []
        self._active_blocks: dict[str, list[int]] = {}
        self._installed = False
        self._applied_events = 0

        self._enabled = bool(self._latent.get("enabled", False))
        self._positive_bank = [
            _clean_entry(item, max_vector_dim=_safe_int(self._latent.get("max_vector_dim", 0), 0))
            for item in list(dict(self._latent.get("banks", {}) or {}).get("positive", []) or [])
            if isinstance(item, dict)
        ]
        self._negative_bank = [
            _clean_entry(item, max_vector_dim=_safe_int(self._latent.get("max_vector_dim", 0), 0))
            for item in list(dict(self._latent.get("banks", {}) or {}).get("negative", []) or [])
            if isinstance(item, dict)
        ]
        self._positive_bank = [item for item in self._positive_bank if item]
        self._negative_bank = [item for item in self._negative_bank if item]

        self._residual_scale = max(0.0, _safe_float(self._latent.get("residual_scale", 0.0), 0.0))
        self._positive_weight = max(0.0, _safe_float(self._latent.get("positive_weight", 1.0), 1.0))
        self._negative_weight = max(0.0, _safe_float(self._latent.get("negative_weight", 1.0), 1.0))
        self._style_weight = max(0.0, _safe_float(self._latent.get("style_weight", 1.0), 1.0))
        self._similarity_floor = max(-1.0, min(1.0, _safe_float(self._latent.get("similarity_floor", 0.05), 0.05)))
        self._positive_topk = max(1, _safe_int(self._latent.get("positive_topk", 2), 2))
        self._negative_topk = max(1, _safe_int(self._latent.get("negative_topk", 2), 2))
        self._capture_decay = max(0.0, min(1.0, _safe_float(self._latent.get("capture_ema_decay", 0.8), 0.8)))

        self._capture_indices: set[int] = set()
        capture_index = _safe_int(self._latent.get("capture_block_index", -1), -1)
        if capture_index >= 0:
            self._capture_indices.add(capture_index)
        for item in list(self._latent.get("capture_block_indices", []) or []):
            parsed = _safe_int(item, -1)
            if parsed >= 0:
                self._capture_indices.add(parsed)

        if self._enabled:
            self._install()

    @property
    def enabled(self) -> bool:
        return bool(self._enabled)

    def _resolve_default_indices(self, num_blocks: int) -> list[int]:
        if num_blocks <= 0:
            return []
        anchors = [num_blocks // 4, num_blocks // 2, (num_blocks * 3) // 4]
        indices = []
        for value in anchors:
            clipped = max(0, min(num_blocks - 1, int(value)))
            if clipped not in indices:
                indices.append(clipped)
        return indices

    def _resolve_block_indices(self, num_blocks: int) -> list[int]:
        requested = [_safe_int(item, -1) for item in list(self._latent.get("block_indices", []) or [])]
        indices: list[int] = []
        for item in requested:
            if 0 <= item < num_blocks and item not in indices:
                indices.append(item)
        if not indices:
            indices = self._resolve_default_indices(num_blocks)
        return indices

    def _resolve_capture_indices(self, selected_indices: list[int]) -> set[int]:
        if self._capture_indices:
            return {index for index in self._capture_indices if index in set(selected_indices)}
        if selected_indices:
            return {selected_indices[-1]}
        return set()

    def _install(self) -> None:
        model_specs = []
        for model_name in ("dit", "dit2"):
            model = getattr(self._pipe, model_name, None)
            blocks = getattr(model, "blocks", None)
            if model is None or blocks is None:
                continue
            model_specs.append((model_name, model, blocks))
        if not model_specs:
            self._enabled = False
            return

        for model_name, _model, blocks in model_specs:
            num_blocks = len(blocks)
            selected_indices = self._resolve_block_indices(num_blocks)
            capture_indices = self._resolve_capture_indices(selected_indices)
            self._active_blocks[model_name] = list(selected_indices)
            for block_index in selected_indices:
                hook = self._make_block_hook(
                    model_name=model_name,
                    block_index=block_index,
                    capture=(block_index in capture_indices),
                )
                handle = blocks[block_index].register_forward_hook(hook)
                self._handles.append(handle)
        self._installed = bool(self._handles)

    def _retrieve_from_bank(
        self,
        *,
        query_vector: list[float],
        bank: list[dict],
        top_k: int,
    ) -> tuple[list[float], list[dict]]:
        scored: list[tuple[float, dict]] = []
        for entry in bank:
            vector = _clean_vector(entry.get("vector"), max_dim=0)
            if not vector or len(vector) != len(query_vector):
                continue
            similarity = _cosine_similarity(query_vector, vector)
            if similarity < self._similarity_floor:
                continue
            weight = max(0.0, _safe_float(entry.get("weight", 1.0), 1.0))
            score = similarity * weight
            scored.append((score, {"score": score, "similarity": similarity, "weight": weight, "vector": vector}))
        if not scored:
            return [], []
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = scored[:max(1, top_k)]
        vectors = [payload["vector"] for _, payload in selected]
        average_vector = _average_vectors(vectors)
        trace = [{key: value for key, value in payload.items() if key != "vector"} for _, payload in selected]
        return average_vector, trace

    def _make_block_hook(self, *, model_name: str, block_index: int, capture: bool):
        def _hook(_module, _inputs, output):
            import torch

            hidden = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(hidden):
                return output
            query_vector = _pool_hidden_vector(hidden)
            if not query_vector:
                return output

            if capture:
                key = (model_name, block_index)
                state = self._capture_state.get(key)
                if state is None:
                    self._capture_state[key] = {"count": 1, "vector": list(query_vector)}
                else:
                    state["count"] = int(state.get("count", 0)) + 1
                    state["vector"] = _ema_vector(state.get("vector", []), query_vector, decay=self._capture_decay)

            positive_vector, positive_trace = self._retrieve_from_bank(
                query_vector=query_vector,
                bank=self._positive_bank,
                top_k=self._positive_topk,
            )
            negative_vector, negative_trace = self._retrieve_from_bank(
                query_vector=query_vector,
                bank=self._negative_bank,
                top_k=self._negative_topk,
            )
            self._retrieval_trace.append(
                {
                    "model": model_name,
                    "block_index": int(block_index),
                    "positive_hits": len(positive_trace),
                    "negative_hits": len(negative_trace),
                }
            )
            self._retrieval_trace = self._retrieval_trace[-24:]

            if not positive_vector and not negative_vector:
                return output
            hidden_dim = hidden.shape[-1]
            if hidden_dim <= 0:
                return output

            delta = [0.0 for _ in range(hidden_dim)]
            if positive_vector and len(positive_vector) == hidden_dim:
                for index, value in enumerate(positive_vector):
                    delta[index] += self._positive_weight * value
            if negative_vector and len(negative_vector) == hidden_dim:
                for index, value in enumerate(negative_vector):
                    delta[index] -= self._negative_weight * value
            if not any(abs(value) > 1e-10 for value in delta):
                return output

            scaled_delta = [value * self._residual_scale * self._style_weight for value in delta]
            delta_tensor = torch.tensor(scaled_delta, device=hidden.device, dtype=hidden.dtype).view(
                *((1 for _ in range(hidden.ndim - 1)), hidden_dim)
            )
            updated = hidden + delta_tensor
            self._applied_events += 1
            if isinstance(output, tuple):
                payload = list(output)
                payload[0] = updated
                return tuple(payload)
            return updated

        return _hook

    def close(self) -> None:
        while self._handles:
            handle = self._handles.pop()
            try:
                handle.remove()
            except Exception:
                pass

    def report(self) -> dict:
        capture_items: list[dict] = []
        vectors: list[list[float]] = []
        for (model_name, block_index), payload in sorted(self._capture_state.items(), key=lambda item: (item[0][0], item[0][1])):
            vector = _clean_vector(payload.get("vector"), max_dim=0)
            if not vector:
                continue
            vectors.append(vector)
            capture_items.append(
                {
                    "model": model_name,
                    "block_index": int(block_index),
                    "count": _safe_int(payload.get("count", 0), 0),
                    "vector": vector,
                }
            )
        summary_vector = _average_vectors(vectors)
        return {
            "enabled": bool(self._enabled),
            "installed": bool(self._installed),
            "backend": self._backend,
            "variant": self._variant_id,
            "blocks": deepcopy(self._active_blocks),
            "applied_events": int(self._applied_events),
            "positive_bank_size": len(self._positive_bank),
            "negative_bank_size": len(self._negative_bank),
            "captures": capture_items,
            "summary_vector": summary_vector,
            "retrieval_trace": list(self._retrieval_trace),
        }


def build_edit_latent_memory_session(
    *,
    pipe: Any,
    controls: dict | None,
    backend: str,
    variant_id: str,
) -> EditLatentMemorySession:
    return EditLatentMemorySession(
        pipe=pipe,
        controls=controls,
        backend=backend,
        variant_id=variant_id,
    )
