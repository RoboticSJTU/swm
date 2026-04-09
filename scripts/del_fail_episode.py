import json
import shutil
from pathlib import Path

# root = Path("/inspire/hdd/project/robot-decision/xiaoyunxiao-240108120113/swm/eval_results/gemini-3-flash-preview/agibot")

# for ep_dir in root.iterdir():
#     if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
#         continue

#     round_dirs = [
#         d for d in ep_dir.iterdir()
#         if d.is_dir() and d.name.startswith("round") and d.name[5:].isdigit()
#     ]
#     if not round_dirs:
#         continue

#     max_round_dir = max(round_dirs, key=lambda d: int(d.name[5:]))
#     judge_path = max_round_dir / "judge.json"

#     if not judge_path.is_file():
#         continue

#     with open(judge_path, "r", encoding="utf-8") as f:
#         judge = json.load(f)

#     if judge["pass"] is False:
#         print(f"delete {ep_dir}")
#         shutil.rmtree(ep_dir)


from pathlib import Path
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed


BASE = Path("/inspire/hdd/project/robot-decision/xiaoyunxiao-240108120113/swm/eval_results/gemini-3-flash-preview/agibot_aug_v1")
MAX_WORKERS = 1000


def process_task(i: int) -> int:
    root = BASE / f"task_{i}"

    if not root.exists():
        print(f"skip non-existent task_{i}")
        return 0

    deleted = 0

    for ep_dir in root.iterdir():
        if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
            continue

        round_dirs = [
            d for d in ep_dir.iterdir()
            if d.is_dir() and d.name.startswith("round") and d.name[5:].isdigit()
        ]
        if not round_dirs:
            continue

        max_round_dir = max(round_dirs, key=lambda d: int(d.name[5:]))
        judge_path = max_round_dir / "judge.json"

        if not judge_path.is_file():
            continue

        try:
            with open(judge_path, "r", encoding="utf-8") as f:
                judge = json.load(f)
        except Exception as e:
            print(f"failed to read {judge_path}: {e}")
            continue

        if judge.get("pass") is False:
            try:
                print(f"delete {ep_dir}")
                shutil.rmtree(ep_dir)
                deleted += 1
            except Exception as e:
                print(f"failed to delete {ep_dir}: {e}")

    return deleted


if __name__ == "__main__":
    total_deleted = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_task, i): i for i in range(327, 791)}

        for future in as_completed(futures):
            i = futures[future]
            try:
                total_deleted += future.result()
            except Exception as e:
                print(f"task_{i} failed: {e}")

    print(f"\nDone. Total deleted episodes: {total_deleted}")