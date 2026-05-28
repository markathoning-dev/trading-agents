from __future__ import annotations

from typing import Protocol, runtime_checkable, Optional, Any

from trading_agent.core.state import AgentState


@runtime_checkable
class StrategyNode(Protocol):
    name: str
    position: str  # "pre_trade" | "post_trade" | "reward"

    def __call__(self, state: AgentState) -> AgentState: ...


NODE_REGISTRY: dict[str, type] = {}


def register_node(cls: type) -> type:
    """Decorator to register a strategy node class."""
    if not hasattr(cls, "name"):
        raise ValueError(f"Node class {cls.__name__} must have a 'name' attribute")
    if not hasattr(cls, "position"):
        raise ValueError(f"Node class {cls.__name__} must have a 'position' attribute")
    if cls.position not in ("pre_trade", "post_trade", "reward"):
        raise ValueError(f"Node {cls.name} position must be 'pre_trade', 'post_trade', or 'reward'")
    NODE_REGISTRY[cls.name] = cls
    return cls


def get_node(name: str) -> Optional[type]:
    """Get a node class by name."""
    return NODE_REGISTRY.get(name)


def list_nodes() -> list[dict[str, Any]]:
    """List all registered nodes with metadata."""
    return [
        {
            "name": cls.name,
            "position": cls.position,
            "class": cls.__name__,
        }
        for cls in NODE_REGISTRY.values()
    ]
