import json
import re
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path("/inspire/hdd/project/robot-decision/xiaoyunxiao-240108120113/swm/eval_results/gemini-3-flash-preview/agibot_aug_v1")
OUT_TXT = ROOT / "judge_fail_summary.txt"
MAX_WORKERS = 1000


def num_suffix(name: str) -> int:
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else -1


def process_episode(task_dir: Path, episode_dir: Path):
    round_dirs = [
        p for p in episode_dir.iterdir()
        if p.is_dir() and re.fullmatch(r"round\d+", p.name)
    ]
    if not round_dirs:
        return None

    round_dir = max(round_dirs, key=lambda p: num_suffix(p.name))
    judge_path = round_dir / "judge.json"
    if not judge_path.exists():
        return None

    try:
        data = json.loads(judge_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if data.get("pass") is False:
        return task_dir.name, num_suffix(episode_dir.name)

    return None


def main():
    jobs = []
    for task_dir in ROOT.iterdir():
        if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
            continue
        for episode_dir in task_dir.iterdir():
            if episode_dir.is_dir() and episode_dir.name.startswith("episode_"):
                jobs.append((task_dir, episode_dir))

    fail_map = defaultdict(list)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_episode, task_dir, episode_dir) for task_dir, episode_dir in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result is not None:
                task_name, episode_idx = result
                fail_map[task_name].append(episode_idx)

            if i % 500 == 0 or i == len(futures):
                print(f"done: {i}/{len(futures)}")

    task_names = sorted(fail_map.keys(), key=num_suffix)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        for task_name in task_names:
            episodes = sorted(set(fail_map[task_name]))
            f.write(f"{task_name}: {','.join(map(str, episodes))}\n")

    print(f"saved to: {OUT_TXT}")


if __name__ == "__main__":
    main()