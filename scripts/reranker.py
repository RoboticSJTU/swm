import os, json, re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# 0) 配置：只改这里
# =========================
BIG_DOMAIN_PDDL = "/inspire/hdd/project/robot-decision/xiaoyunxiao-240108120113/swm/domain/unified_domain.pddl"
STEPS_PATH      = "/inspire/hdd/project/robot-decision/xiaoyunxiao-240108120113/swm/eval_results/test/droid/episode_6/steps.txt"
MODEL_DIR       = "/inspire/hdd/project/robot-decision/xiaoyunxiao-240108120113/swm/Qwen3-Reranker-0.6B"   

OUT_DIR     = "/inspire/hdd/project/robot-decision/xiaoyunxiao-240108120113/swm/out"
TOPK        = 1
BATCH_SIZE  = 32

QUERY_INSTRUCT = "Select the single robot action operator whose semantic meaning best matches the given natural language step."
EXPORT_TEMPLATE_DOMAIN = True


# =========================
# 1) 小工具：读文件
# =========================
def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().replace("\r\n", "\n").replace("\r", "\n")

def read_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


# =========================
# 2) 从 domain.pddl 自动生成 mapping（operator -> 精简注释）
#    若同目录已有 unified_domain_mapping.json 则直接读
# =========================
def clean_action_comment(s: str) -> str:
    if not s:
        return ""
    s = s.lstrip(";").strip()
    s = re.sub(r"\?[A-Za-z0-9_\-]+", "", s)  # 去变量
    s = re.sub(r"\bwith\s+(the\s+)?(right\s+|left\s+)?hand\b", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+\.", ".", s)
    return s.strip()

def comment_before(text: str, idx: int):
    # 只看紧贴上一行（不跨空行）
    line_start = text.rfind("\n", 0, idx) + 1
    prev_end = line_start - 1
    if prev_end < 0:
        return "", -1
    prev_start = text.rfind("\n", 0, prev_end) + 1
    prev_line = text[prev_start:prev_end].rstrip("\r")
    if prev_line.strip() and prev_line.lstrip().startswith(";"):
        if not re.match(r"\s*;;\s*source\s*:", prev_line, flags=re.I):
            return prev_line.strip(), prev_start
    return "", -1

def build_or_load_mapping_from_domain(domain_pddl_path: str):
    domain_dir = os.path.dirname(domain_pddl_path)
    mapping_path = os.path.join(domain_dir, "unified_domain_mapping.json")

    if os.path.exists(mapping_path):
        mapping = json.load(open(mapping_path, "r", encoding="utf-8"))
        mapping = {k: v for k, v in mapping.items() if isinstance(v, str) and v.strip()}
        if not mapping:
            raise RuntimeError(f"{mapping_path} 存在但为空/无有效文本，请删除后重试。")
        return mapping_path, mapping

    text = read_text(domain_pddl_path)
    acts = list(re.finditer(r"\(\s*:action\s+([^\s\)]+)", text))
    if not acts:
        raise RuntimeError("在 domain.pddl 中没有找到任何 (:action ...)")

    mapping = {}
    for m in acts:
        op = m.group(1)
        raw_cmt, _ = comment_before(text, m.start())
        c = clean_action_comment(raw_cmt) or op.replace("_", " ").strip()
        mapping[op] = c

    json.dump(mapping, open(mapping_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[OK] Created mapping: {mapping_path} (actions={len(mapping)})")
    return mapping_path, mapping


# =========================
# 3) Qwen3-Reranker：全库打分（yes/no logits）
#    模板来自 vLLM 官方示例/社区讨论 :contentReference[oaicite:1]{index=1}
# =========================
class Qwen3Reranker:
    def __init__(self, model_dir: str, max_len: int = 2048):
        self.tok = AutoTokenizer.from_pretrained(model_dir, padding_side="left", trust_remote_code=True)
        self.mdl = AutoModelForCausalLM.from_pretrained(
            model_dir, device_map="auto", torch_dtype="auto", trust_remote_code=True
        ).eval()

        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        self.max_len = max_len
        self.yes_id = self.tok.convert_tokens_to_ids("yes")
        self.no_id  = self.tok.convert_tokens_to_ids("no")

        self.prefix = (
            '<|im_start|>system\n'
            'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
            'Note that the answer can only be "yes" or "no".<|im_end|>\n'
            '<|im_start|>user\n'
        )
        self.suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        self.prefix_ids = self.tok.encode(self.prefix, add_special_tokens=False)
        self.suffix_ids = self.tok.encode(self.suffix, add_special_tokens=False)

    def _format(self, instruct: str, query: str, doc: str) -> str:
        return f"<Instruct>: {instruct}\n<Query>: {query}\n<Document>: {doc}"

    @torch.no_grad()
    def score_docs(self, instruct: str, query: str, docs, batch_size: int = 32):
        # 返回每个 doc 的 p(yes)
        scores = []
        for s in range(0, len(docs), batch_size):
            batch = [self._format(instruct, query, d) for d in docs[s:s+batch_size]]
            enc = self.tok(
                batch,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=self.max_len - len(self.prefix_ids) - len(self.suffix_ids),
            )
            for i, ids in enumerate(enc["input_ids"]):
                enc["input_ids"][i] = self.prefix_ids + ids + self.suffix_ids
            enc = self.tok.pad(enc, padding=True, return_tensors="pt", max_length=self.max_len).to(self.mdl.device)

            logits = self.mdl(**enc).logits[:, -1, :]  # last token
            two = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=1)
            p_yes = torch.softmax(two, dim=1)[:, 1]
            scores.append(p_yes.float().cpu())
        return torch.cat(scores, dim=0)


# =========================
# 4) （可选）导出 template_domain.pddl：局部 predicates + 保留 action 注释
#    （沿用你的实现，已包含“注释行属于 action span”的修复）
# =========================
def sexpr_span(text: str, lparen_idx: int):
    depth = 0
    for i in range(lparen_idx, len(text)):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return lparen_idx, i + 1
    raise RuntimeError("Unbalanced parentheses while scanning s-expression.")

def extract_predicates(text: str):
    m = re.search(r"\(\s*:predicates\b", text)
    if not m:
        raise RuntimeError("No (:predicates ...) block found in domain.")
    ps, pe = sexpr_span(text, m.start())
    block = text[ps:pe]

    pred_map, depth, i = {}, 0, 0
    while i < len(block):
        if block[i] == "(":
            depth += 1
            if depth == 2:
                s, e = sexpr_span(block, i)
                expr = block[s:e]
                mm = re.match(r"\(\s*([^\s\(\)]+)", expr)
                if mm:
                    name = mm.group(1)
                    line_end = block.find("\n", e)
                    if line_end == -1:
                        line_end = len(block)
                    pred_map[name] = (expr + block[e:line_end]).rstrip()
        elif block[i] == ")":
            depth -= 1
        i += 1
    return ps, pe, pred_map

def extract_actions_with_comments(text: str):
    matches = list(re.finditer(r"\(\s*:action\s+([^\s\)]+)", text))
    if not matches:
        raise RuntimeError("No (:action ...) found in domain.")

    spans, act_map = [], {}
    for m in matches:
        name = m.group(1)
        s, e = sexpr_span(text, m.start())
        block = text[s:e].rstrip()

        cmt, cmt_start = comment_before(text, s)
        if cmt:
            act_map[name] = (cmt.rstrip() + "\n" + block).rstrip() + "\n"
            span_start = cmt_start
        else:
            act_map[name] = block.rstrip() + "\n"
            span_start = s

        spans.append((name, span_start, e))

    act_start = min(st for _, st, _ in spans)
    act_end   = max(en for _, _, en in spans)
    return act_start, act_end, act_map

def used_predicates_in_action(action_text: str):
    used = set()
    bad = {"and", "not", "when", "forall", "exists", "=", "increase", "decrease", "assign"}
    for key in (":precondition", ":effect"):
        idx = action_text.find(key)
        if idx == -1:
            continue
        p = action_text.find("(", idx)
        if p == -1:
            continue
        _, end = sexpr_span(action_text, p)
        chunk = action_text[p:end]
        for mm in re.finditer(r"\(\s*([^\s\(\)]+)", chunk):
            sym = mm.group(1)
            if sym not in bad and not sym.startswith(":"):
                used.add(sym)
    return used

def export_template_domain(domain_path: str, selected_ops, out_path: str):
    text = read_text(domain_path)
    pred_s, pred_e, pred_map = extract_predicates(text)
    act_s,  act_e,  act_map  = extract_actions_with_comments(text)

    kept_actions, used_preds = [], set()
    for op in selected_ops:
        if op in act_map:
            atxt = act_map[op]
            kept_actions.append(atxt)
            used_preds |= used_predicates_in_action(atxt)

    if not kept_actions:
        raise RuntimeError("选中的 operator 在 domain 里一个都没匹配上（检查 action 名是否一致）。")

    kept_pred_lines = ["    " + pred_map[p].strip() for p in sorted(used_preds) if p in pred_map]
    new_pred_block = "  (:predicates\n" + ("\n".join(kept_pred_lines) + "\n" if kept_pred_lines else "") + "  )\n"
    out_text = text[:pred_s] + new_pred_block + text[pred_e:act_s] + "\n" + "\n".join(kept_actions) + "\n" + text[act_e:]
    open(out_path, "w", encoding="utf-8").write(out_text)


# =========================
# 5) 主流程：steps -> reranker(topK) -> 输出
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    mapping_path, mapping = build_or_load_mapping_from_domain(BIG_DOMAIN_PDDL)
    steps = read_lines(STEPS_PATH)
    if not steps:
        raise RuntimeError("steps.txt 为空。")

    # 全库候选：operator + 短注释
    op_names = [k for k, v in mapping.items() if isinstance(v, str) and v.strip()]
    docs     = [mapping[k].strip() for k in op_names]
    if not op_names:
        raise RuntimeError("mapping 中没有有效的 operator 文本。")

    rr = Qwen3Reranker(MODEL_DIR)

    retrieval_log = []
    selected_ops = set()

    for step in steps:
        scores = rr.score_docs(QUERY_INSTRUCT, step, docs, batch_size=BATCH_SIZE)  # [N]
        k = min(TOPK, len(op_names))
        vals, idx = torch.topk(scores, k)

        top = []
        for sc, i in zip(vals.tolist(), idx.tolist()):
            op = op_names[i]
            top.append({"operator": op, "score": float(sc), "comment": mapping.get(op, "")})
            selected_ops.add(op)

        retrieval_log.append({
            "raw_step": step,
            "query_text": f"<Instruct>: {QUERY_INSTRUCT}\n<Query>: {step}",
            "topk": top
        })

    selected_ops = sorted(selected_ops)
    template_mapping = {op: mapping[op] for op in selected_ops if op in mapping}

    out_json = os.path.join(OUT_DIR, "operator_template.json")
    out_txt  = os.path.join(OUT_DIR, "operator_template.txt")
    out_log  = os.path.join(OUT_DIR, "retrieval_log.json")

    json.dump(template_mapping, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    open(out_txt, "w", encoding="utf-8").write("\n".join(selected_ops) + "\n")
    json.dump(retrieval_log, open(out_log, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"[OK] steps={len(steps)} ops_in_domain={len(op_names)} selected_unique_ops={len(selected_ops)}")
    print(f"[MAP] {mapping_path}")
    print(f"[MODEL] {MODEL_DIR}")
    print(f"[OUT] {out_json}")
    print(f"[OUT] {out_txt}")
    print(f"[OUT] {out_log}")

    if EXPORT_TEMPLATE_DOMAIN:
        out_domain = os.path.join(OUT_DIR, "template_domain.pddl")
        export_template_domain(BIG_DOMAIN_PDDL, selected_ops, out_domain)
        print(f"[OUT] {out_domain}")

if __name__ == "__main__":
    main()
