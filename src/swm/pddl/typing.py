from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TypedSymbol:
    name: str
    type_name: str
    explicit: bool


def parse_typed_symbols(
    items: Iterable[object],
    *,
    context: str,
    default_type: str = "object",
) -> list[TypedSymbol]:
    """Parse a PDDL grouped typed list such as ``a b - block c``."""
    tokens = list(items)
    if any(not isinstance(token, str) for token in tokens):
        raise ValueError(f"{context}: typed declarations may contain only symbols")

    result: list[TypedSymbol] = []
    pending: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token != "-":
            pending.append(token)
            index += 1
            continue
        if not pending or index + 1 >= len(tokens) or tokens[index + 1] == "-":
            raise ValueError(f"{context}: malformed type annotation")
        type_name = tokens[index + 1]
        result.extend(TypedSymbol(name, type_name, True) for name in pending)
        pending = []
        index += 2

    result.extend(TypedSymbol(name, default_type, False) for name in pending)
    return result


def typed_symbol_map(
    items: Iterable[object],
    *,
    context: str,
    default_type: str = "object",
) -> tuple[list[str], dict[str, str], set[str]]:
    parsed = parse_typed_symbols(
        items,
        context=context,
        default_type=default_type,
    )
    order: list[str] = []
    types: dict[str, str] = {}
    explicit: set[str] = set()
    for item in parsed:
        if item.name in types:
            if types[item.name] != item.type_name:
                raise ValueError(
                    f"{context}: conflicting types for '{item.name}': "
                    f"{types[item.name]} and {item.type_name}"
                )
            raise ValueError(f"{context}: duplicate declaration '{item.name}'")
        order.append(item.name)
        types[item.name] = item.type_name
        if item.explicit:
            explicit.add(item.name)
    return order, types, explicit


@dataclass(frozen=True)
class TypeHierarchy:
    parents: dict[str, str]

    @classmethod
    def object_only(cls) -> "TypeHierarchy":
        return cls({})

    @classmethod
    def from_declaration(cls, items: Iterable[object]) -> "TypeHierarchy":
        order, parents, _ = typed_symbol_map(
            items,
            context=":types",
            default_type="object",
        )
        if "object" in parents:
            raise ValueError(":types: 'object' is the built-in root type")
        declared = set(order)
        for type_name, parent in parents.items():
            if parent != "object" and parent not in declared:
                raise ValueError(
                    f":types: type '{type_name}' has undeclared parent '{parent}'"
                )
            if type_name == parent:
                raise ValueError(f":types: type '{type_name}' cannot inherit itself")

        hierarchy = cls(parents)
        for type_name in order:
            hierarchy.ancestors(type_name)
        return hierarchy

    @property
    def declared(self) -> set[str]:
        return set(self.parents)

    def require(self, type_name: str, context: str) -> None:
        if type_name != "object" and type_name not in self.parents:
            raise ValueError(f"{context}: undeclared type '{type_name}'")

    def ancestors(self, type_name: str) -> tuple[str, ...]:
        self.require(type_name, ":types")
        result = [type_name]
        seen = {type_name}
        current = type_name
        while current != "object":
            current = self.parents[current]
            if current in seen:
                cycle = " -> ".join([*result, current])
                raise ValueError(f":types: inheritance cycle: {cycle}")
            seen.add(current)
            result.append(current)
        return tuple(result)

    def is_subtype(self, actual: str, expected: str) -> bool:
        self.require(actual, "type compatibility")
        self.require(expected, "type compatibility")
        return expected in self.ancestors(actual)

    def least_common_supertype(self, types: Iterable[str]) -> str:
        values = list(types)
        if not values:
            return "object"
        common = set(self.ancestors(values[0]))
        for value in values[1:]:
            common.intersection_update(self.ancestors(value))
        for candidate in self.ancestors(values[0]):
            if candidate in common:
                return candidate
        return "object"


def render_typed_symbols(
    order: Iterable[str],
    types: dict[str, str],
    *,
    include_object: bool = False,
) -> list[str]:
    """Render a stable grouped typed list while preserving first-seen order."""
    order = list(order)
    include_object = include_object or any(types[name] != "object" for name in order)
    groups: list[tuple[str, list[str]]] = []
    for name in order:
        type_name = types[name]
        if type_name == "object" and not include_object:
            type_name = ""
        if groups and groups[-1][0] == type_name:
            groups[-1][1].append(name)
        else:
            groups.append((type_name, [name]))

    tokens: list[str] = []
    for type_name, names in groups:
        tokens.extend(names)
        if type_name:
            tokens.extend(["-", type_name])
    return tokens
