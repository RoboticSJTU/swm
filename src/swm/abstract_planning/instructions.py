"""Names-only operation lists shared by generation, migration and audits."""
import re

from .model import parse_sexpr


def operation_names(domain_text):
    return [a[1] for a in parse_sexpr(domain_text)
            if isinstance(a, list) and a[:1] == [":action"]]


def operation_list(domain_text):
    return "Allowed operations: " + ", ".join(operation_names(domain_text)) + ".\n"


def names_only(text, domain_text):
    """Preserve rules/goals/symbols; replace only interface hints and lists."""
    text = text.replace("In parameters (tile, from, to), from is the tile cell and to the blank cell before the swap. ", "")
    text = text.replace("Parameters from and to are the tile cell and blank cell before the swap. ", "")
    text = text.replace(" For movement/interactions, from/to are player/target cells and d the heading; turns take old/new headings.", "")
    text = text.replace("The from/source_after and target_before/to pairs are adjacent slot heights before and after the move. ", "")
    text = text.replace(" entry is the adjacent cell; to is the final landing cell.", "")
    text = text.replace("Boarding consumes one free slot and departing restores one.", "For a finite-capacity lift, boarding consumes one free slot and departing restores one.")
    text = re.sub(r"(?:Available operation names and parameter order:|Available operations and parameter order:|Operations \(parameter order\):|Actions:)\n(?:[\w-]+\([^\n]*\)\n)+",
                  lambda _: operation_list(domain_text), text)
    text = text.replace("IDs and these interfaces", "IDs and operation names").replace("IDs and interfaces", "IDs and operation names")
    check_operation_list(text, domain_text)
    return text


def check_operation_list(text, domain_text):
    matches = re.findall(r"^Allowed operations: ([a-z0-9_, -]+)\.$", text, re.M)
    assert len(matches) == 1, "expected one names-only operation list"
    names = matches[0].split(", ")
    assert len(names) == len(set(names)) and set(names) == set(operation_names(domain_text)), "instruction/domain action names differ"
    assert not re.search(r"parameter order|In parameters|Parameters from|from/source_after|turns take old/new|\b(?:" + "|".join(map(re.escape, names)) + r")\(", text), "action signature leaked into instruction"
