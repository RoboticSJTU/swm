import json
import re
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================
# 全局：模型单例 + 推理锁（避免并发把CPU/GPU打爆）
# =========================
_MODEL_LOCK = threading.Lock()
_RERANKER_SINGLETON = None


# =========================
# 1) 读文件
# =========================
def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")

def _read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


# =========================
# 2) domain -> mapping: operator -> comment(doc)
# =========================
def _clean_action_comment(s: str) -> str:
    if not s:
        return ""
    s = s.lstrip(";").strip()
    s = re.sub(r"\?[A-Za-z0-9_\-]+", "", s)  # 去变量
    s = re.sub(r"\bwith\s+(the\s+)?(right\s+|left\s+)?hand\b", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+\.", ".", s)
    return s.strip()

def _comment_before(text: str, idx: int) -> Tuple[str, int]:
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

def _build_or_load_mapping_from_domain(unified_domain_path: Path) -> Dict[str, str]:
    """
    operator -> short comment（用于reranker doc）
    同目录缓存 unified_domain_mapping.json
    """
    domain_dir = unified_domain_path.parent
    mapping_path = domain_dir / "unified_domain_mapping.json"

    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        mapping = {k: v for k, v in mapping.items() if isinstance(v, str) and v.strip()}
        if not mapping:
            raise RuntimeError(f"{mapping_path} 存在但为空/无有效文本，请删除后重试。")
        return mapping

    text = _read_text(unified_domain_path)
    acts = list(re.finditer(r"\(\s*:action\s+([^\s\)]+)", text))
    if not acts:
        raise RuntimeError("在 unified_domain.pddl 中没有找到任何 (:action ...)")

    mapping: Dict[str, str] = {}
    for m in acts:
        op = m.group(1)
        raw_cmt, _ = _comment_before(text, m.start())
        mapping[op] = _clean_action_comment(raw_cmt) or op.replace("_", " ").strip()

    mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    return mapping


# =========================
# 3) （可选）取完整 action block（用于 pddl_action_template）
# =========================
def _sexpr_span(text: str, lparen_idx: int) -> Tuple[int, int]:
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

def _extract_actions_with_comments(domain_text: str) -> Dict[str, str]:
    matches = list(re.finditer(r"\(\s*:action\s+([^\s\)]+)", domain_text))
    if not matches:
        raise RuntimeError("No (:action ...) found in domain.")

    act_map: Dict[str, str] = {}
    for m in matches:
        name = m.group(1)
        s, e = _sexpr_span(domain_text, m.start())
        block = domain_text[s:e].rstrip()

        cmt, _ = _comment_before(domain_text, s)
        if cmt:
            act_map[name] = (cmt.rstrip() + "\n" + block).rstrip() + "\n"
        else:
            act_map[name] = block.rstrip() + "\n"
    return act_map


# =========================
# 4) 极简 Qwen3-Reranker（不需要 accelerate；cpu/gpu 都能跑）
# =========================
class Qwen3RerankerLite:
    def __init__(self, model_dir: str, device: str = "cpu", max_len: int = 2048):
        self.device = device
        self.max_len = max_len

        self.tok = AutoTokenizer.from_pretrained(model_dir, padding_side="left", trust_remote_code=True)

        # 关键：不使用 device_map="auto"（否则会要求 accelerate）
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.mdl = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            dtype=dtype,
        ).eval().to(device)

        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        self.yes_id = self.tok.convert_tokens_to_ids("yes")
        self.no_id = self.tok.convert_tokens_to_ids("no")

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
    def score_docs(self, instruct: str, query: str, docs: List[str], batch_size: int = 16) -> torch.Tensor:
        """
        返回每个 doc 的 p(yes)
        """
        max_body_len = self.max_len - len(self.prefix_ids) - len(self.suffix_ids)
        scores = []

        for s in range(0, len(docs), batch_size):
            batch_docs = docs[s:s+batch_size]
            batch = [self._format(instruct, query, d) for d in batch_docs]

            enc = self.tok(
                batch,
                padding=False,
                truncation="longest_first",
                max_length=max_body_len,
            )
            input_ids = [self.prefix_ids + ids + self.suffix_ids for ids in enc["input_ids"]]

            enc2 = self.tok.pad(
                {"input_ids": input_ids},
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            logits = self.mdl(**enc2).logits[:, -1, :]
            two = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=1)
            p_yes = torch.softmax(two, dim=1)[:, 1]
            scores.append(p_yes.float().cpu())

        return torch.cat(scores, dim=0)


def _get_reranker(model_dir: str, device: str) -> Qwen3RerankerLite:
    global _RERANKER_SINGLETON
    if _RERANKER_SINGLETON is not None:
        return _RERANKER_SINGLETON
    with _MODEL_LOCK:
        if _RERANKER_SINGLETON is None:
            _RERANKER_SINGLETON = Qwen3RerankerLite(model_dir=model_dir, device=device)
    return _RERANKER_SINGLETON


# =========================
# 5) ✅ 你项目真正要的函数：steps -> topk ops -> retrieval_log.json + pddl_action_template
# =========================
def build_pddl_action_template(
    kf_plan_path: Path,
    unified_domain_path: Path,
    topk: int = 1,
    rerank_model_dir: str = "/inspire/hdd/project/robot-decision/xiaoyunxiao-240108120113/swm/Qwen3-Reranker-0.6B",
    batch_size: int = 16,
    device: Optional[str] = None,   # None: 自动选 cuda/cpu
    query_instruct: str = "Select the single robot action operator whose semantic meaning best matches the given natural language step.",
) -> str:
    kf_plan_path = Path(kf_plan_path)
    unified_domain_path = Path(unified_domain_path)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = kf_plan_path.parent
    out_log = out_dir / "retrieval_log.json"

    steps = _read_lines(kf_plan_path)
    if not steps:
        out_log.write_text(json.dumps([], ensure_ascii=False, indent=2), encoding="utf-8")
        return ""

    mapping = _build_or_load_mapping_from_domain(unified_domain_path)
    op_names = [k for k, v in mapping.items() if isinstance(v, str) and v.strip()]
    docs = [mapping[k].strip() for k in op_names]
    if not op_names:
        raise RuntimeError("mapping 中没有有效的 operator 文本。")

    domain_text = _read_text(unified_domain_path)
    act_map = _extract_actions_with_comments(domain_text)

    rr = _get_reranker(rerank_model_dir, device=device)

    retrieval_log = []
    selected_ops = set()

    # 从简：串行跑（你外面是多线程任务，这里串行最稳）
    with _MODEL_LOCK:
        for step in steps:
            scores = rr.score_docs(query_instruct, step, docs, batch_size=batch_size)  # [N]
            k = min(int(topk), len(op_names))
            vals, idx = torch.topk(scores, k)

            top = []
            for sc, i in zip(vals.tolist(), idx.tolist()):
                op = op_names[i]
                top.append({"operator": op, "comment": mapping.get(op, ""), "score": float(sc)})
                selected_ops.add(op)

            retrieval_log.append({
                "step": step,
                "retrieve": top
            })

    out_log.write_text(json.dumps(retrieval_log, ensure_ascii=False, indent=2), encoding="utf-8")

    # 返回 action blocks 作为 pddl_action_template
    blocks = []
    for op in sorted(selected_ops):
        blk = act_map.get(op, "")
        if blk.strip():
            blocks.append(blk.rstrip())

    return "\n\n".join(blocks).strip()
