import os, json, re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# =========================
# 0) 配置：只改这里
# =========================
BIG_DOMAIN_PDDL = "/home/xyx/下载/swm/merged_domain.pddl"
STEPS_PATH      = "/home/xyx/下载/swm/eval_results/test/human/steps.txt"

RERANK_MODEL_DIR = "/home/xyx/下载/swm/Qwen3-Reranker-0.6B"
EMB_MODEL_DIR    = "/home/xyx/下载/VLPA/Qwen3-Embedding-0.6B"

OUT_DIR   = "/home/xyx/下载/swm/out"
TOPK      = 1          # 精排输出 topK
RECALL_K  = 10          # 召回候选数（建议 64~256）
EMB_BS    = 64         # embedding batch
RERANK_BS = 32         # rerank batch（原 BATCH_SIZE）

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
# 2) mapping：operator -> 精简注释（doc）
# =========================
def clean_action_comment(s: str) -> str:
    if not s:
        return ""
    s = s.lstrip(";").strip()
    s = re.sub(r"\?[A-Za-z0-9_\-]+", "", s)
    s = re.sub(r"\bwith\s+(the\s+)?(right\s+|left\s+)?hand\b", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+\.", ".", s)
    return s.strip()

def comment_before(text: str, idx: int):
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
        mapping[op] = clean_action_comment(raw_cmt) or op.replace("_", " ").strip()

    json.dump(mapping, open(mapping_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[OK] Created mapping: {mapping_path} (actions={len(mapping)})")
    return mapping_path, mapping


# =========================
# 3) Embedding 召回 + 缓存
# =========================
def encode(model: SentenceTransformer, texts):
    return model.encode(
        texts,
        batch_size=EMB_BS,
        normalize_embeddings=True,   # dot == cosine
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)

def load_or_build_op_cache(cache_path: str, fingerprint: str, model: SentenceTransformer, op_names, docs):
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        if str(data["fingerprint"]) == fingerprint:
            return list(data["op_names"]), data["op_embs"].astype(np.float32)
    op_embs = encode(model, docs)
    np.savez(
        cache_path,
        fingerprint=np.array(fingerprint, dtype=object),
        op_names=np.array(op_names, dtype=object),
        op_embs=op_embs.astype(np.float32),
    )
    return op_names, op_embs


# =========================
# 4) Qwen3-Reranker：精排（yes/no logits）
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
        scores = []
        for s in range(0, len(docs), batch_size):
            batch = [self._format(instruct, query, d) for d in docs[s:s+batch_size]]
            enc = self.tok(
                batch, padding=False, truncation="longest_first",
                return_attention_mask=False,
                max_length=self.max_len - len(self.prefix_ids) - len(self.suffix_ids),
            )
            for i, ids in enumerate(enc["input_ids"]):
                enc["input_ids"][i] = self.prefix_ids + ids + self.suffix_ids
            enc = self.tok.pad(enc, padding=True, return_tensors="pt", max_length=self.max_len).to(self.mdl.device)

            logits = self.mdl(**enc).logits[:, -1, :]
            two = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=1)
            scores.append(torch.softmax(two, dim=1)[:, 1].float().cpu())
        return torch.cat(scores, dim=0)


# =========================
# 5) （可选）导出 template_domain.pddl（沿用你的实现）
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
    return min(st for _, st, _ in spans), max(en for _, _, en in spans), act_map

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
# 6) 主流程：召回(embedding) + 精排(reranker)
# =========================
def main():
    import time
    os.makedirs(OUT_DIR, exist_ok=True)

    mapping_path, mapping = build_or_load_mapping_from_domain(BIG_DOMAIN_PDDL)
    steps = read_lines(STEPS_PATH)
    if not steps:
        raise RuntimeError("steps.txt 为空。")

    op_names = [k for k, v in mapping.items() if isinstance(v, str) and v.strip()]
    docs     = [mapping[k].strip() for k in op_names]
    # docs = [f"Operator: {k}\nDescription: {mapping[k].strip()}" for k in op_names]

    if not op_names:
        raise RuntimeError("mapping 中没有有效的 operator 文本。")

    # 1) cache / build op embeddings
    emb = SentenceTransformer(EMB_MODEL_DIR)
    cache_path = os.path.join(OUT_DIR, "cache_ops_comment_emb.npz")
    fp = f"{EMB_MODEL_DIR}|dom={BIG_DOMAIN_PDDL}|dm={os.path.getmtime(BIG_DOMAIN_PDDL)}|map={mapping_path}|mm={os.path.getmtime(mapping_path)}"

    t0 = time.perf_counter()
    op_names, op_embs = load_or_build_op_cache(cache_path, fp, emb, op_names, docs)  # [N,D]
    t_cache = time.perf_counter() - t0

    # 2) reranker
    rr = Qwen3Reranker(RERANK_MODEL_DIR)
    t_recall, t_rerank = 0.0, 0.0

    retrieval_log, selected_ops = [], set()

    for step in steps:
        # qtxt = f"Instruct: {QUERY_INSTRUCT}\nStep: {step}"
        qtxt = f"{step}"

        # ---- recall (embedding) ----
        t0 = time.perf_counter()
        q = encode(emb, [qtxt])[0]          # [D]
        sims = op_embs @ q                  # [N]
        rk = min(RECALL_K, sims.shape[0])
        ridx = np.argpartition(-sims, rk - 1)[:rk]
        ridx = ridx[np.argsort(-sims[ridx])]

        cand_ops  = [op_names[i] for i in ridx.tolist()]
        cand_docs = [docs[i] for i in ridx.tolist()]
        recall_topk = [
            {"operator": op_names[i], "score": float(sims[i]), "comment": mapping.get(op_names[i], "")}
            for i in ridx.tolist()
        ]
        t_recall += time.perf_counter() - t0

        # ---- rerank ----
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        rs = rr.score_docs(QUERY_INSTRUCT, step, cand_docs, batch_size=RERANK_BS)  # [rk]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_rerank += time.perf_counter() - t0

        k = min(TOPK, len(cand_ops))
        vals, idx = torch.topk(rs, k)

        rerank_topk = []
        for sc, j in zip(vals.tolist(), idx.tolist()):
            op = cand_ops[j]
            rerank_topk.append({"operator": op, "score": float(sc), "comment": mapping.get(op, "")})
            selected_ops.add(op)

        retrieval_log.append({
            "raw_step": step,
            "query_text": qtxt,
            "recall_topk": recall_topk,     # ✅ embedding 召回结果
            "rerank_topk": rerank_topk      # ✅ reranker 精排结果
        })

    selected_ops = sorted(selected_ops)
    template_mapping = {op: mapping[op] for op in selected_ops if op in mapping}

    out_json = os.path.join(OUT_DIR, "operator_template.json")
    out_txt  = os.path.join(OUT_DIR, "operator_template.txt")
    out_log  = os.path.join(OUT_DIR, "retrieval_log.json")

    json.dump(template_mapping, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    open(out_txt, "w", encoding="utf-8").write("\n".join(selected_ops) + "\n")
    json.dump(retrieval_log, open(out_log, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    n = len(steps)
    print(f"[OK] steps={n} ops_in_domain={len(op_names)} selected_unique_ops={len(selected_ops)}")
    print(f"[MAP] {mapping_path}")
    print(f"[CACHE] {cache_path}")
    print(f"[EMB] {EMB_MODEL_DIR}")
    print(f"[RERANK] {RERANK_MODEL_DIR}")
    print(f"[OUT] {out_json}")
    print(f"[OUT] {out_txt}")
    print(f"[OUT] {out_log}")
    print(f"[TIME] cache_build_or_load={t_cache:.3f}s")
    print(f"[TIME] recall_total={t_recall:.3f}s  recall_avg={t_recall/n:.4f}s/step")
    print(f"[TIME] rerank_total={t_rerank:.3f}s  rerank_avg={t_rerank/n:.4f}s/step")

    if EXPORT_TEMPLATE_DOMAIN:
        out_domain = os.path.join(OUT_DIR, "template_domain.pddl")
        export_template_domain(BIG_DOMAIN_PDDL, selected_ops, out_domain)
        print(f"[OUT] {out_domain}")

main() 