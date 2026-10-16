"""Shared PDDL identifier and schema checks.

PDDL uses ``-`` both as a type-list separator and as a legal character in
identifiers.  The project deliberately uses the latter only as a legacy
spelling; canonical output uses underscores while preserving the standalone
type separator and PDDL keywords.
"""

from __future__ import annotations

import re
from collections.abc import Iterable


_TOKEN_RE = re.compile(r";[^\n]*|\(|\)|[^\s();]+")
_HISTORY_PARAMETER_MARKERS = (
    "context",
    "history",
    "phase",
    "stage",
    "record",
    "sequence",
    "step",
    "marker",
)
_HISTORY_PREDICATE_NAMES = {
    "closed_after",
    "closed_after_storage",
    "inserted_into",
    "opened_after",
    "rinsed_after",
    "toasted_in",
}
_HISTORY_PREDICATE_PREFIXES = (
    "picked_",
    "placed_",
    "removed_",
    "returned_",
    "stored_",
)
_IDENTIFIER_ALIASES = {
    # Stable spelling for the misspelled object family that otherwise creates
    # separate predicates, types, and operators for the same lamp.
    "repelent": "repellent",
    # Pose predicates name the physical state directly; ``pose_of`` already
    # carries the pose role for pose objects.
    "upright_pose": "upright",
    "horizontal_pose": "horizontal",
}


def canonicalize_identifier(token: str) -> str:
    """Return the stable identifier spelling used by generated PDDL."""
    if token == "-":
        return token
    if token.startswith(":"):
        return token.lower()
    canonical = token.lower().replace("-", "_")
    for old, new in _IDENTIFIER_ALIASES.items():
        canonical = canonical.replace(old, new)
    return canonical


def canonicalize_pddl_text(text: str) -> str:
    """Canonicalize names without changing PDDL syntax or comments."""

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.startswith(";") or token in {"(", ")"}:
            return token
        return canonicalize_identifier(token)

    return _TOKEN_RE.sub(replace, text)


def parse_sexp(text: str):
    """Parse one parenthesized PDDL expression after removing comments."""
    tokens = [
        token
        for token in _TOKEN_RE.findall(text)
        if not token.startswith(";")
    ]
    root: list[object] = []
    stack = [root]
    for token in tokens:
        if token == "(":
            expression: list[object] = []
            stack[-1].append(expression)
            stack.append(expression)
        elif token == ")":
            if len(stack) == 1:
                raise ValueError("PDDL expression has an unmatched ')'")
            stack.pop()
        else:
            stack[-1].append(canonicalize_identifier(token))
    if len(stack) != 1 or len(root) != 1:
        raise ValueError("PDDL expression is malformed")
    return root[0]


def _variables(node: object) -> set[str]:
    result: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, str):
            if value.startswith("?"):
                result.add(value.lower())
            return
        if isinstance(value, list):
            for child in value:
                visit(child)

    visit(node)
    return result


def _history_parameter(variable: str) -> bool:
    name = variable.lower().lstrip("?")
    return any(marker in name for marker in _HISTORY_PARAMETER_MARKERS)


def has_fixed_hand_equality(expression_text: str) -> bool:
    """Detect a precondition that hard-codes a demonstrated hand identity."""
    root = parse_sexp(expression_text)

    def visit(node: object) -> bool:
        if not isinstance(node, list):
            return False
        if (
            len(node) == 3
            and node[0] == "="
            and any(
                isinstance(argument, str)
                and argument in {"left_hand", "right_hand"}
                for argument in node[1:]
            )
        ):
            return True
        return any(visit(child) for child in node)

    return visit(root)


def unbound_effect_parameters(
    parameters: Iterable[str],
    precondition_text: str,
    effect_text: str,
) -> set[str]:
    """Find action parameters used by effects but absent from preconditions."""
    declared = {parameter.lower() for parameter in parameters}
    precondition = _variables(parse_sexp(precondition_text))
    effect = _variables(parse_sexp(effect_text))
    return (effect - precondition) & declared


def validate_effect_bindings(
    action_name: str,
    parameters: Iterable[str],
    precondition_text: str,
    effect_text: str,
) -> list[str]:
    """Return structural errors for causally unbound effect parameters.

    Typed placement targets may legitimately occur only in an effect (the
    target is still a typed action participant).  History/context parameters
    and non-transfer effect-only parameters have no physical interpretation
    when absent from every precondition and let an action write arbitrary
    facts. Those schemas are rejected at generation/merge boundaries so a
    task-specific repair is not silently propagated.
    """
    precondition = parse_sexp(precondition_text)
    effect = parse_sexp(effect_text)
    return validate_effect_binding_literals(
        action_name,
        parameters,
        _literal_atoms(precondition),
        _literal_atoms(effect),
    )


def _literal_atoms(node: object) -> list[tuple[str, ...]]:
    """Flatten a STRIPS expression into atomic literals, ignoring polarity."""
    if not isinstance(node, list) or not node:
        return []
    if node[0] == "and":
        return [atom for child in node[1:] for atom in _literal_atoms(child)]
    if node[0] == "not" and len(node) == 2:
        return _literal_atoms(node[1])
    if not isinstance(node[0], str) or any(
        not isinstance(argument, str) for argument in node[1:]
    ):
        return []
    return [tuple(str(item) for item in node)]


def _history_predicate(name: str) -> bool:
    return name in _HISTORY_PREDICATE_NAMES or name.startswith(
        _HISTORY_PREDICATE_PREFIXES
    )


def validate_effect_binding_literals(
    action_name: str,
    parameters: Iterable[str],
    preconditions: Iterable[tuple[str, ...]],
    effects: Iterable[tuple[str, ...]],
) -> list[str]:
    """Validate effect-only parameters from parsed STRIPS literals.

    A typed destination or value may first occur in an effect when a physical
    relation ties it to an already-bound participant, for example ``on(x, t)``
    or ``temperature_set(ac, value)``.  History relations do not establish
    that causal tie by themselves.
    """
    declared = {parameter.lower() for parameter in parameters}
    precondition_variables = {
        argument.lower()
        for literal in preconditions
        for argument in literal[1:]
        if argument.lower() in declared
    }
    effect_literals = [tuple(item.lower() for item in literal) for literal in effects]
    effect_variables = {
        argument
        for literal in effect_literals
        for argument in literal[1:]
        if argument in declared
    }
    issues = []
    for variable in sorted(effect_variables - precondition_variables):
        if _history_parameter(variable):
            issues.append(
                f"{action_name}: effect-only history/context parameter {variable}"
            )
            continue
        physically_linked = any(
            variable in literal[1:]
            and not _history_predicate(literal[0])
            and any(
                argument in precondition_variables
                for argument in literal[1:]
                if argument != variable
            )
            for literal in effect_literals
        )
        if not physically_linked:
            issues.append(
                f"{action_name}: effect-only parameter {variable} is not tied "
                "to a causal participant"
            )
    return issues


def validate_effect_binding_variables(
    action_name: str,
    unbound_parameters: Iterable[str],
) -> list[str]:
    """Validate already-parsed effect-only parameters."""
    # Compatibility helper for callers that do not retain literal structure.
    # Such callers cannot prove a relational causal tie, so remain strict.
    issues = []
    for variable in sorted(unbound_parameters):
        if _history_parameter(variable):
            issues.append(
                f"{action_name}: effect-only history/context parameter {variable}"
            )
        else:
            issues.append(
                f"{action_name}: effect-only parameter {variable} is not tied "
                "to a causal participant"
            )
    return issues
