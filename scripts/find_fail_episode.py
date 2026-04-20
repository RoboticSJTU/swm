from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import shutil

ROOT = Path("/home/xyx/下载/swm/eval_results/gemini-3-flash-preview/human_aug_v2")
OUT_TXT = ROOT / "judge_fail_summary.txt"
MAX_WORKERS = 1000
DELETE_FAIL_EPISODES = True  # True: 删除失败的 episode 目录；False: 只统计不删除


def num_suffix(name: str) -> int:
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else -1


def process_episode(task_name: str, episode_dir: Path):
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
        return task_name, num_suffix(episode_dir.name), episode_dir

    return None


def main():
    jobs = []

    for p in ROOT.iterdir():
        if not p.is_dir():
            continue

        # 嵌套结构: human/task_x/episode_x
        if p.name.startswith("task_"):
            task_name = p.name
            for episode_dir in p.iterdir():
                if episode_dir.is_dir() and episode_dir.name.startswith("episode_"):
                    jobs.append((task_name, episode_dir))

        # 平层结构: human/episode_x
        elif p.name.startswith("episode_"):
            idx = num_suffix(p.name)
            if idx >= 0:
                task_name = f"task_{idx}"
                jobs.append((task_name, p))

    fail_map = defaultdict(list)
    fail_dirs = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_episode, task_name, episode_dir) for task_name, episode_dir in jobs]

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result is not None:
                task_name, episode_idx, episode_dir = result
                fail_map[task_name].append(episode_idx)
                fail_dirs.append(episode_dir)

            if i % 500 == 0 or i == len(futures):
                print(f"done: {i}/{len(futures)}")

    task_names = sorted(fail_map.keys(), key=num_suffix)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        for task_name in task_names:
            episodes = sorted(set(fail_map[task_name]))
            f.write(f"{task_name}: {','.join(map(str, episodes))}\n")

    print(f"saved to: {OUT_TXT}")

    if DELETE_FAIL_EPISODES:
        for episode_dir in fail_dirs:
            if episode_dir.exists():
                shutil.rmtree(episode_dir)
                print(f"deleted: {episode_dir}")


if __name__ == "__main__":
    main()