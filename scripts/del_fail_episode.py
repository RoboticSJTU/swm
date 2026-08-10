from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import json
import re
import shutil

ROOT_DIR = Path(__file__).resolve().parent.parent
BASE = ROOT_DIR / "eval_results" / "gpt-5.6-sol"
ROOTS = [BASE / "human_aug"]

MAX_WORKERS = 2560
DELETE_FAIL_EPISODES = True


def num_suffix(path: Path) -> int:
    m = re.search(r"(\d+)$", path.name)
    return int(m.group(1)) if m else -1


def check_episode(episode_dir: Path):
    round_dirs = [
        p for p in episode_dir.iterdir()
        if p.is_dir() and re.fullmatch(r"round\d+", p.name)
    ]

    if not round_dirs:
        return episode_dir, "no_round"

    round_dir = max(round_dirs, key=num_suffix)
    judge_path = round_dir / "judge.json"

    if not judge_path.exists():
        return episode_dir, "no_judge"

    try:
        data = json.loads(judge_path.read_text(encoding="utf-8"))
    except Exception:
        return episode_dir, "bad_judge_json"

    if data.get("pass") is not True:
        return episode_dir, "judge_not_pass"

    return None


for root in ROOTS:
    if not root.exists():
        print(f"[{root.name}] missing, skip")
        continue

    episode_dirs = [p for p in root.glob("task_*/episode_*") if p.is_dir()]
    fail_results = []

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, max(1, len(episode_dirs)))) as executor:
        futures = [executor.submit(check_episode, p) for p in episode_dirs]

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                fail_results.append(result)

    print(f"\n[{root.name}] checked: {len(episode_dirs)}/{len(episode_dirs)}")
    print(f"[{root.name}] fail episodes: {len(fail_results)}")

    fail_map = defaultdict(list)
    for episode_dir, reason in fail_results:
        fail_map[episode_dir.parent.name].append((episode_dir.name, reason))

    for task_name in sorted(fail_map, key=lambda x: num_suffix(Path(x))):
        episodes = sorted(fail_map[task_name], key=lambda x: num_suffix(Path(x[0])))
        episode_text = ", ".join([f"{ep}({reason})" for ep, reason in episodes])
        print(f"[{root.name}] delete {task_name}: {episode_text}")

    if DELETE_FAIL_EPISODES:
        fail_dirs = [episode_dir for episode_dir, _ in fail_results]

        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, max(1, len(fail_dirs)))) as executor:
            futures = [executor.submit(shutil.rmtree, p) for p in fail_dirs]

            for future in as_completed(futures):
                future.result()
