"""Identity-memoized iterative traversal helpers for expression DAGs."""

from ..errors import GraphCycleError
from .nodes.base import RandomVariable


def _unique_nodes(
    root: RandomVariable,
) -> tuple[tuple[RandomVariable, ...], tuple[RandomVariable, ...]]:
    preorder: list[RandomVariable] = []
    postorder: list[RandomVariable] = []
    states: dict[int, int] = {}
    pending: list[tuple[RandomVariable, bool]] = [(root, False)]

    while pending:
        node, expanded = pending.pop()
        identity = id(node)
        state = states.get(identity, 0)
        if expanded:
            states[identity] = 2
            postorder.append(node)
            continue
        if state == 1:
            raise GraphCycleError("cycle detected while traversing expression graph")
        if state == 2:
            continue

        states[identity] = 1
        preorder.append(node)
        pending.append((node, True))
        pending.extend(
            (dependency.var, False) for dependency in reversed(node._dependency_slots)
        )

    return tuple(preorder), tuple(postorder)


def unique_nodes_preorder(root: RandomVariable) -> tuple[RandomVariable, ...]:
    """Return each reachable node once, before its dependencies."""

    return _unique_nodes(root)[0]


def unique_nodes_postorder(root: RandomVariable) -> tuple[RandomVariable, ...]:
    """Return each reachable node once, after its dependencies."""

    return _unique_nodes(root)[1]


__all__ = ["unique_nodes_postorder", "unique_nodes_preorder"]
