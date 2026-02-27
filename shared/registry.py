"""Generic registry for generators, metrics, exporters, and any other extensible component."""

from typing import Any, Callable, Dict, List, Tuple, Type


class Registry:
    """A name → (class, metadata) mapping with decorator support."""

    def __init__(self, name: str):
        self.name = name
        self._entries: Dict[str, Tuple[Type, Dict[str, Any]]] = {}

    def register(self, key: str, **meta: Any) -> Callable:
        """Use as a class decorator: @registry.register("sft", format="instruction")"""
        def decorator(cls: Type) -> Type:
            self._entries[key] = (cls, meta)
            return cls
        return decorator

    def add(self, key: str, cls: Type, **meta: Any) -> None:
        """Programmatic registration (non-decorator)."""
        self._entries[key] = (cls, meta)

    def get(self, key: str) -> Tuple[Type, Dict[str, Any]]:
        if key not in self._entries:
            raise KeyError(
                f"[{self.name}] '{key}' not registered. "
                f"Available: {list(self._entries.keys())}"
            )
        return self._entries[key]

    def list_keys(self) -> List[str]:
        return list(self._entries.keys())

    def list_all(self) -> Dict[str, Dict[str, Any]]:
        """Return {key: metadata} for all entries."""
        return {k: meta for k, (_, meta) in self._entries.items()}

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __repr__(self) -> str:
        return f"Registry('{self.name}', keys={self.list_keys()})"


# Pre-instantiated registries for cross-pipeline use
generator_registry = Registry("generators")
metric_registry = Registry("metrics")
exporter_registry = Registry("exporters")
judge_registry = Registry("judges")
