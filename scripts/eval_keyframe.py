import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from swm.utils.apis import call_gpt_json


PRED_ROOT = Path("/home/xyx/下载/swm/eval_results/gemini-3-flash-preview/human_energy_无指令")

MODEL = "gemini-3-flash-preview"
GT_ROOT = Path("/home/xyx/下载/swm/eval_results/gemini-3-flash-preview/human_GT")
SAVE_PATH = PRED_ROOT.with_name(PRED_ROOT.name + "_eval.json")
MAX_WORKERS = 40

PROMPT = """
You are evaluating whether two action sequences describe the same task logic.

Sequence A is Ground Truth (GT).
Sequence B is Prediction (PRED).

Compare them semantically, not by surface wording.
Rules:
1. Two actions match if they describe the same meaningful manipulation, state change, placement, or interaction on the same relevant object, source, target, or result.

2. Minor wording differences are acceptable (e.g., tool names like brush/spoon, container names like foil tray/metal tray, appliances like oven/microwave, surface names like table/counter, or implicit surface descriptions such as "from the drawer cabinet" = "from the top of the drawer cabinet"). Actions are incorrect only if the object, source, target, or resulting state differs (e.g., "Pick up broom from floor" ≠ "Pick up broom from dustpan").

3. Granularity redundancy in PRED is allowed only at the sub-action level.
   Example: GT says "pick up apple", while PRED says "grasp apple" + "pick up apple".
   This is allowed and should NOT be treated as wrong extra logic.

4. Non-granularity action redundancy is NOT allowed.
   If GT requires performing an action a certain number of times, PRED must not add extra full actions beyond that logic.
   Example: if GT washes an object twice, but PRED washes it three times, that extra wash is incorrect.

5. Split / merge is allowed, as long as the same task-relevant logic is preserved and no additional full action logic is introduced.

6. Wrong extra logic means a predicted step introduces unrelated, incorrect, contradictory, or semantically additional task logic not implied by GT. This includes extra full actions, repeated task-level actions beyond GT, or extra state-changing operations.

7. Missing GT logic means some GT step is not semantically covered by PRED. This also includes incomplete action chains where a necessary follow-up action is missing.
   Example: PRED picks up a lid but never puts it down when GT requires that completion.

8. Order matters for strict equivalence, but split / merge and granularity refinement do not break order consistency if the high-level logic order is preserved.

Important:
- Fine-grained decomposition is allowed, but extra task-level actions are not.
- Check both over-completion and under-completion: do not allow extra repeated full actions, and do not miss required follow-up actions.

Return JSON only in this exact format:
{{
  "reasoning": "think step by step",
  "equivalent": true,
  "order_consistent": true,
  "gt_covered": [1, 2, 3],
  "pred_supported": [1, 2, 3],
  "wrong_extra_pred": [],
  "summary": "用中文简单总结"
}}

Definitions:
- gt_covered: GT step indices whose logic is covered by PRED.
- pred_supported: PRED step indices that are semantically supported by GT, including harmless fine-grained redundancy.
- wrong_extra_pred: PRED step indices that add wrong / unrelated / contradictory logic.
- equivalent=true only if the two sequences describe the same overall task logic with no missing GT logic, no wrong extra logic, and preserved order at the logical level.

Use 1-based indices.

Ground Truth (GT):
{gt_text}

Prediction (PRED):
{pred_text}
""".strip()


def episode_key(name):
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else name


def read_steps(path):
    return [x.strip() for x in path.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()] if path.exists() else []


def clean_indices(xs, n):
    if not isinstance(xs, list):
        return []
    out = []
    seen = set()
    for x in xs:
        try:
            i = int(x)
        except Exception:
            continue
        if 1 <= i <= n and i not in seen:
            out.append(i)
            seen.add(i)
    out.sort()
    return out


def eval_episode(episode):
    gt_path = GT_ROOT / episode / "kf_plan.txt"
    pred_path = PRED_ROOT / episode / "kf_plan.txt"

    if not gt_path.exists():
        return {
            "episode": episode,
            "success": False,
            "f1": 0.0,
            "status": "missing_gt_file",
            "missing_gt": 0,
            "wrong_extra_pred": 0,
            "summary": "缺少 GT 文件",
        }

    if not pred_path.exists():
        return {
            "episode": episode,
            "success": False,
            "f1": 0.0,
            "status": "missing_pred_file",
            "missing_gt": 0,
            "wrong_extra_pred": 0,
            "summary": "缺少 PRED 文件",
        }

    gt_steps = read_steps(gt_path)
    pred_steps = read_steps(pred_path)

    if not gt_steps:
        return {
            "episode": episode,
            "success": not pred_steps,
            "f1": 1.0 if not pred_steps else 0.0,
            "status": "ok",
            "missing_gt": 0,
            "wrong_extra_pred": 0 if not pred_steps else len(pred_steps),
            "summary": "GT 为空" if pred_steps else "GT 和 PRED 都为空",
        }

    gt_text = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(gt_steps))
    pred_text = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(pred_steps)) if pred_steps else "(empty)"

    try:
        resp = call_gpt_json(MODEL, PROMPT.format(gt_text=gt_text, pred_text=pred_text), None)
    except Exception as e:
        return {
            "episode": episode,
            "success": False,
            "f1": 0.0,
            "status": "vlm_failed",
            "missing_gt": len(gt_steps),
            "wrong_extra_pred": 0,
            "summary": f"模型调用失败: {e}",
        }

    if not isinstance(resp, dict) or not all(k in resp for k in ("equivalent", "order_consistent", "gt_covered", "pred_supported", "wrong_extra_pred", "summary")):
        return {
            "episode": episode,
            "success": False,
            "f1": 0.0,
            "status": "vlm_failed",
            "missing_gt": len(gt_steps),
            "wrong_extra_pred": 0,
            "summary": "模型返回格式错误",
        }

    gt_covered = clean_indices(resp["gt_covered"], len(gt_steps))
    pred_supported = clean_indices(resp["pred_supported"], len(pred_steps))
    wrong_extra_pred = clean_indices(resp["wrong_extra_pred"], len(pred_steps))

    equivalent = bool(resp["equivalent"])
    order_consistent = bool(resp["order_consistent"])

    if equivalent and order_consistent and not wrong_extra_pred:
        if not gt_covered:
            gt_covered = list(range(1, len(gt_steps) + 1))
        if pred_steps and not pred_supported:
            pred_supported = list(range(1, len(pred_steps) + 1))

    recall = len(gt_covered) / len(gt_steps)
    precision = len(pred_supported) / len(pred_steps) if pred_steps else 0.0
    f1 = 0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)

    return {
        "episode": episode,
        "success": equivalent and order_consistent and len(gt_covered) == len(gt_steps) and not wrong_extra_pred,
        "f1": round(f1, 6),
        "status": "ok",
        "missing_gt": len(gt_steps) - len(gt_covered),
        "wrong_extra_pred": len(wrong_extra_pred),
        "summary": str(resp["summary"]).strip(),
    }


def main():
    episodes = sorted([p.name for p in GT_ROOT.iterdir() if p.is_dir()], key=episode_key)

    print(f"episodes: {len(episodes)}")
    print(f"model: {MODEL}")
    print(f"workers: {MAX_WORKERS}")
    print()

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(eval_episode, episode): episode for episode in episodes}
        for i, fut in enumerate(as_completed(futures), 1):
            episode = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {
                    "episode": episode,
                    "success": False,
                    "f1": 0.0,
                    "status": "exception",
                    "missing_gt": 0,
                    "wrong_extra_pred": 0,
                    "summary": str(e),
                }
            results.append(r)
            print(f"[{i}/{len(episodes)}] {episode} | success={r['success']} | f1={r['f1']:.3f} | status={r['status']}")

    results.sort(key=lambda x: (x["success"], episode_key(x["episode"])))

    success_count = sum(r["success"] for r in results)
    failed_episodes = [r["episode"] for r in results if not r["success"]]

    output = {
        "summary": {
            "total_episodes": len(results),
            "success_episodes": success_count,
            "failed_episodes": len(results) - success_count,
            "success_rate": round(success_count / len(results), 6) if results else 0.0,
            "avg_f1": round(sum(r["f1"] for r in results) / len(results), 6) if results else 0.0,
            "failed_episode_list": failed_episodes,
        },
        "results": results,
    }

    SAVE_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n==================== summary ====================")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(f"\nsaved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()