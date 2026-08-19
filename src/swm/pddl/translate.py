import re
from pathlib import Path


def translate_pddl_plan(domain_path: Path, plan_path: Path) -> None:
    """Translate a symbolic plan and write plan_nl.txt beside it."""
    plan_nl_path = plan_path.parent / "plan_nl.txt"
    domain = domain_path.read_text(encoding="utf-8", errors="ignore")
    plan = plan_path.read_text(encoding="utf-8", errors="ignore")

    def clean_template(comment_line: str, unary_preds: set[str]) -> str:
        s = comment_line.lstrip(";").strip()
        s = re.sub(r"\bobject\s+(\?\w+)\b", r"\1", s, flags=re.IGNORECASE)
        if unary_preds:
            pat = (
                r"\b("
                + "|".join(map(re.escape, sorted(unary_preds, key=str.lower)))
                + r")\s+(\?\w+)\b"
            )
            s = re.sub(pat, r"\2", s, flags=re.IGNORECASE)

        s = re.sub(r"(?<=[A-Za-z0-9_])\?", " ?", s)
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"\s+([.,;:!?])", r"\1", s)
        return s

    action2info: dict[str, tuple[str, list[str]]] = {}
    lines = domain.splitlines()

    i = 0
    while i < len(lines):
        if "(:action" not in lines[i]:
            i += 1
            continue

        m = re.search(r"\(:action\s+([^\s()]+)", lines[i])
        if not m:
            i += 1
            continue
        act = m.group(1)

        j = i - 1
        while j >= 0 and lines[j].strip() == "":
            j -= 1
        comment = (
            lines[j].strip() if j >= 0 and lines[j].lstrip().startswith(";") else ""
        )

        bal, block_lines = 0, []
        k = i
        while k < len(lines):
            block_lines.append(lines[k])
            bal += lines[k].count("(") - lines[k].count(")")
            if k > i and bal <= 0:
                break
            k += 1
        block = "\n".join(block_lines)

        params_m = re.search(
            r":parameters\s*\((.*?)\)", block, flags=re.DOTALL | re.IGNORECASE
        )
        params = re.findall(r"\?\w+", params_m.group(1) if params_m else "")

        pre_m = re.search(
            r":precondition\s*\((.*?)\)\s*:effect",
            block,
            flags=re.DOTALL | re.IGNORECASE,
        )
        pre = pre_m.group(1) if pre_m else ""
        unary_preds = {
            pred
            for pred, _v in re.findall(r"\(\s*([A-Za-z_][\w\-]*)\s+(\?\w+)\s*\)", pre)
            if pred.lower() not in {"and", "or", "not", "=", "imply"}
        }

        tpl = clean_template(comment, unary_preds) if comment else ""
        action2info[act] = (tpl, params)

        i = k + 1

    out = []
    for raw in plan.splitlines():
        line = raw.strip()
        if not line or line.startswith((";", "#")):
            continue
        line = re.sub(r";.*$", "", line).strip()
        line = re.sub(r"^\s*\d+\s*:\s*", "", line)
        line = re.sub(r"\s*\[\s*[\d\.]+\s*\]\s*$", "", line)
        line = line.strip().strip("()").strip()
        if not line:
            continue

        toks = line.split()
        act, args = toks[0], toks[1:]
        tpl, params = action2info.get(act, ("", []))

        if not tpl:
            out.append(act + (" " + " ".join(args) if args else ""))
            continue

        mapping = {params[t]: args[t] for t in range(min(len(params), len(args)))}
        s = re.sub(r"(?<=[A-Za-z0-9_])\?", " ?", tpl)
        s = re.sub(
            r"\?\w+",
            lambda match, mapping=mapping: mapping.get(match.group(), match.group()),
            s,
        )
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"\s+([.,;:!?])", r"\1", s)
        out.append(s)

    plan_nl_path.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
