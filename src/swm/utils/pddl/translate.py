import re
from pathlib import Path

def replace_vars_exact(template: str, mapping: dict[str, str]) -> str:
    """只替换完整变量 token，如 ?s、?sh，不做子串误替换"""
    def repl(m):
        var = m.group(0)
        return mapping.get(var, var)
    return re.sub(r"\?\w+", repl, template)

def translate_pddl_plan(domain_path: Path, plan_path: Path) -> None:
    """会自动写入plan_nl.txt
    """
    plan_nl_path = plan_path.parent / "plan_nl.txt"
    domain = domain_path.read_text(encoding="utf-8", errors="ignore")
    plan = plan_path.read_text(encoding="utf-8", errors="ignore")

    # ---------- helpers ----------
    def space_before_var(s: str) -> str:
        # "from?t" -> "from ?t"（不会把 ?tr 拆开）
        return re.sub(r"(?<=[A-Za-z0-9_])\?", " ?", s)

    def clean_template(comment_line: str, unary_preds: set[str]) -> str:
        s = comment_line.lstrip(";").strip()

        # # 翻译中移除手 drop hand phrase (both variants)
        # s = re.sub(r"\s*\bwith\s+hand\s+\?\w+\b", "", s, flags=re.IGNORECASE)
        # s = re.sub(r"\s*\busing\s+hand\s+\?\w+\b", "", s, flags=re.IGNORECASE)
        # s = re.sub(r"\s*\bfrom\s+hand\s+\?\w+\b", "", s, flags=re.IGNORECASE)

        # drop "object ?x" -> "?x"
        s = re.sub(r"\bobject\s+(\?\w+)\b", r"\1", s, flags=re.IGNORECASE)

        # drop unary predicate names: "table ?t" -> "?t"
        if unary_preds:
            pat = r"\b(" + "|".join(map(re.escape, sorted(unary_preds, key=str.lower))) + r")\s+(\?\w+)\b"
            s = re.sub(pat, r"\2", s, flags=re.IGNORECASE)

        s = space_before_var(s)
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"\s+([.,;:!?])", r"\1", s)
        return s

    # ---------- parse domain: action -> (template, params) ----------
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

        # nearest previous non-empty comment line
        j = i - 1
        while j >= 0 and lines[j].strip() == "":
            j -= 1
        comment = lines[j].strip() if j >= 0 and lines[j].lstrip().startswith(";") else ""

        # grab action block by parentheses balance
        bal, block_lines = 0, []
        k = i
        while k < len(lines):
            block_lines.append(lines[k])
            bal += lines[k].count("(") - lines[k].count(")")
            if k > i and bal <= 0:
                break
            k += 1
        block = "\n".join(block_lines)

        params_m = re.search(r":parameters\s*\((.*?)\)", block, flags=re.DOTALL | re.IGNORECASE)
        params = re.findall(r"\?\w+", params_m.group(1) if params_m else "")

        pre_m = re.search(r":precondition\s*\((.*?)\)\s*:effect", block, flags=re.DOTALL | re.IGNORECASE)
        pre = pre_m.group(1) if pre_m else ""
        unary_preds = {
            pred for pred, _v in re.findall(r"\(\s*([A-Za-z_][\w\-]*)\s+(\?\w+)\s*\)", pre)
            if pred.lower() not in {"and", "or", "not", "=", "imply"}
        }

        tpl = clean_template(comment, unary_preds) if comment else ""
        action2info[act] = (tpl, params)

        i = k + 1

    # ---------- translate plan ----------
    out = []
    for raw in plan.splitlines():
        line = raw.strip()
        if not line or line.startswith(";") or line.startswith("#"):
            continue
        line = re.sub(r";.*$", "", line).strip()                 # trailing comments
        line = re.sub(r"^\s*\d+\s*:\s*", "", line)               # "0: ..."
        line = re.sub(r"\s*\[\s*[\d\.]+\s*\]\s*$", "", line)     # "[1.23]"
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
        s = space_before_var(tpl)
        s = replace_vars_exact(s, mapping)

        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"\s+([.,;:!?])", r"\1", s)
        out.append(s)

    plan_nl_path.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")

if __name__ == "__main__":
    domain_path = Path("/inspire/hdd/project/robot-decision/xiaoyunxiao-240108120113/swm/eval_results/test/droid/episode_15/round1/domain.pddl") 
    plan_path = Path("/inspire/hdd/project/robot-decision/xiaoyunxiao-240108120113/swm/eval_results/test/droid/episode_15/round1/plan.txt") 
    translate_pddl_plan(domain_path, plan_path)