from pathlib import Path
import json
from typing import List, Dict, Any, Optional
import re
from swm.utils.apis import call_gpt_json

def get_first_keyframe_image(keyframe_dir: Path) -> Path:
    """
    从任务关键帧目录中找到第一张关键帧图像。
    目录结构通常为：
        keyframe_dir/
            seg_00/
            seg_01/
            ...
    """
    seg_dirs = sorted(d for d in keyframe_dir.iterdir() if d.is_dir() and d.name.startswith("seg_"))
    for seg_dir in seg_dirs:
        imgs = list_keyframes_sorted(seg_dir)
        if imgs:
            return imgs[0]
    raise ValueError(f"No keyframes found in {keyframe_dir}")

def list_keyframes_sorted(episode_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    imgs = [p for p in episode_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]

    def key_fn(p: Path):
        if p.stem.isdigit():
            return (0, int(p.stem))
        return (1, p.stem)

    imgs.sort(key=key_fn)
    return imgs

def clean_action_lines(x) -> List[str]:
    if x is None:
        return ["none"]
    if isinstance(x, list):
        x = "\n".join([str(t) for t in x])
    s = str(x).strip()
    
    if "\\n" in s:
        s = s.replace("\\n", "\n")
        
    if not s or s.lower().strip().strip(".") == "none":
        return ["none"]

    out: List[str] = []
    for ln in s.splitlines():
        ln = ln.strip()
        if not ln:
            continue

        ln = re.sub(r"^\s*[\-\*\u2022]\s+", "", ln)
        ln = re.sub(r"^\s*\(?\d+\)?[.)]\s+", "", ln).strip()
        ln = re.sub(r"\\([.,;:!?])", r"\1", ln)   
        if not ln or ln.lower().strip().strip(".") == "none":
            continue
        if not ln.endswith("."):
            ln += "."
        out.append(ln)
    return out if out else ["none"]

def postprocess_plan_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    prev_key: Optional[str] = None

    for s in lines:
        s = (s or "").strip()
        if not s:
            continue

        key = s.lower().strip().strip(".")
        if key == "none":
            continue

        # 只去掉“相邻重复”
        if prev_key is not None and key == prev_key:
            continue

        out.append(s)
        prev_key = key

    return out

def get_prompt_from_template(path, **kwargs) -> str:
    with path.open("r") as f:
        template = f.read()

    prompt = template.format(**kwargs)
    return prompt

def construct_sharegpt_training_data(all_results, output_path: Path) -> None:
    ok = []
    all_results.sort(key=lambda r: (r["task_domain"], r["task_id"]))
    for r in all_results:
        if r["success"] and r["judge_pass"]:
            ok.append(r["sharegpt_sample"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(ok, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[sharegpt] saved {len(ok)} samples -> {output_path}")


def remove_grasp_actions(lines: List[str]) -> List[str]:
    """
    后处理删掉最后kf_plan.txt里的grasp，使其不参与pddl生成。
    """
    return [line for line in lines if not line.strip().lower().startswith("grasp")]


def print_plan_learning(
    group_idx: int,
    img_a: Path, 
    img_b: Path,
    data: Dict[str, Any],
    debug_log_path: Optional[Path] = None,
    history: Optional[List[str]] = None,
) -> None:
    sep = "=" * 24
    lines = [
        f"{sep} PAIR {sep}",
        f"[G{group_idx}]: {img_a.name}->{img_b.name}",
        f"[history_actions]\n{history}",
        f"[history_reasoning] {data['history_reasoning']}",
        f"[history_check] {data['history_check']}",
        f"[history_feedback] {data['history_feedback']}",
        f"[action_reasoning] {data['action_reasoning']}",
        f"[action] {data['action']}",
        "",
    ]

    if debug_log_path is not None:
        debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))
            

def generate_steps_from_segments(
    img_paths: List[Path],
    instruction: str,
    history_actions: List[str],
    error_action: str,
    feedback: str,
    group_idx: int,
    num_groups: int,
    model_name: str,
    debug_txt_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    针对一个关键帧片段（一个 seg_* 目录下的一组图像），调用 VLM 生成该片段对应的动作描述。
    """
    history = "\n".join(f"{i+1}. {a}" for i, a in enumerate(history_actions)) or "none"
    error_action = error_action or "none"
    feedback = feedback or ""
    use_retry_prompt = bool(feedback)

    prompt_dir = Path(__file__).parent.parent / "prompt_templates"
    prompt_path = (prompt_dir / "plan_learning_with_feedback.txt" if use_retry_prompt
        else prompt_dir / "plan_learning.txt"
    )

    prompt_kwargs = dict(
        instruction=instruction,
        history=history,
        group_idx=group_idx,
        num_groups=num_groups,
    )
    if use_retry_prompt:
        prompt_kwargs["error_action"] = error_action
        prompt_kwargs["feedback"] = feedback

    prompt = get_prompt_from_template(prompt_path, **prompt_kwargs)

    data = call_gpt_json(model_name, prompt, img_paths)
    print_plan_learning(group_idx, img_paths[0], img_paths[-1], data, debug_txt_path, history)

    history_check = data["history_check"]
    history_feedback = "" if history_check else data["history_feedback"]
    actions = clean_action_lines(data["action"])

    earliest_bad_group = int(data["earliest_bad_group"])

    return {
        "history_check": history_check,
        "earliest_bad_group": earliest_bad_group,
        "history_feedback": history_feedback,
        "actions_cleaned": actions,
        "action_field_raw": data["action"],
    }

def generate_steps_with_backtracking(
    seg_imgs_list: List[List[Path]],
    instruction: str,
    kf_plan_path: Path,
    model_name: str,
    debug_txt_path: Optional[Path] = None,
    max_backtracks: int = 20
) -> List[str]:
    seg_imgs_list = [imgs for imgs in seg_imgs_list if imgs]

    plan_lines: List[str] = []
    history: List[str] = []
    meta: List[Dict[str, Any]] = []
    retry_hints: Dict[int, Dict[str, str]] = {}

    i = 0
    backtracks = 0
    num_groups = len(seg_imgs_list)

    while i < num_groups:
        img_paths = seg_imgs_list[i]

        if i in retry_hints:
            error_action = retry_hints[i]["error_action"]
            feedback = retry_hints[i]["feedback"]
        else:
            error_action = ""
            feedback = ""

        out = generate_steps_from_segments(
            img_paths=img_paths,
            instruction=instruction,
            history_actions=history,
            error_action=error_action,
            feedback=feedback,
            group_idx=i,
            num_groups=num_groups,
            debug_txt_path=debug_txt_path,
            model_name=model_name,
        )

        if not out["history_check"]:
            backtracks += 1
            if backtracks > max_backtracks:
                raise RuntimeError(
                    f"Too many backtracks. last feedback: {out['history_feedback']}"
                )

            rollback_i = out["earliest_bad_group"]
            bad_meta = meta[rollback_i]

            retry_hints[rollback_i] = {
                "error_action": bad_meta["action_field_raw"],
                "feedback": out["history_feedback"],
            }

            for k in list(retry_hints.keys()):
                if k > rollback_i:
                    del retry_hints[k]

            remove_flat_n = sum(m["flat_n"] for m in meta[rollback_i:])
            remove_hist_n = sum(m["hist_n"] for m in meta[rollback_i:])

            if remove_flat_n:
                del plan_lines[-remove_flat_n:]
            if remove_hist_n:
                del history[-remove_hist_n:]
            del meta[rollback_i:]

            kf_plan_path.write_text(
                "\n".join(plan_lines) + ("\n" if plan_lines else ""),
                encoding="utf-8",
            )

            i = rollback_i
            continue

        actions = out["actions_cleaned"]
        plan_lines.extend(actions)

        hist_added = [a for a in actions if a.strip().lower() != "none"]
        history.extend(f"[G{i}] {a}" for a in hist_added)

        meta.append({
            "action_field_raw": out["action_field_raw"],
            "flat_n": len(actions),
            "hist_n": len(hist_added),
            "actions": actions,  
        })
        if i in retry_hints:
            del retry_hints[i]

        kf_plan_path.write_text("\n".join(plan_lines) + "\n", encoding="utf-8")
        i += 1

    plan_lines = postprocess_plan_lines(plan_lines)
    kf_plan_path.write_text("\n".join(plan_lines) + ("\n" if plan_lines else ""),encoding="utf-8")

    group_lines = []
    for group_idx, m in enumerate(meta):
        group_actions = postprocess_plan_lines(m["actions"])
        group_lines.extend(f"[G{group_idx}] {a}" for a in group_actions)
    kf_plan_group_path = kf_plan_path.with_stem(f"{kf_plan_path.stem}_group")
    kf_plan_group_path.write_text("\n".join(group_lines) + ("\n" if group_lines else ""),encoding="utf-8")

    return plan_lines

def learn_steps_from_keyframes(
    model_name, 
    keyframe_dir: Path,
    instruction: str,
    save_dir: Path,
    max_backtracks: int = 10,
):
    """
    从某个任务的关键帧目录中学习完整动作序列 raw_steps。

    输入：
    - keyframe_dir: 该任务的关键帧目录，里面通常有 seg_00、seg_01、...
    - instruction：原始自然语言任务指令
    - save_dir：该任务结果保存目录
    - max_backtracks：关键帧学习允许的最大回退次数

    输出：
    - first_img：第一张关键帧图像，后续作为场景图传给 PDDL 生成和 judge
    - raw_steps：从关键帧学习得到的完整动作序列
    """

    # 找到所有 seg_xx 子目录，并按名字排序
    seg_dirs = sorted(d for d in keyframe_dir.iterdir() if d.is_dir() and d.name.startswith("seg_"))

    # 每个 seg 目录里再取出按时间排序后的关键帧列表并过滤掉空分段
    seg_imgs_list = [list_keyframes_sorted(d) for d in seg_dirs]
    seg_imgs_list = [imgs for imgs in seg_imgs_list if imgs]

    if not seg_imgs_list:
        raise ValueError(f"No keyframes found in {keyframe_dir}")

    first_img = seg_imgs_list[0][0]
    raw_steps = generate_steps_with_backtracking(
        seg_imgs_list=seg_imgs_list,
        instruction=instruction,
        kf_plan_path=save_dir / "kf_plan.txt",
        debug_txt_path=save_dir / "pair_debug.log",
        max_backtracks=max_backtracks,
        model_name=model_name,
    )

    return first_img, raw_steps