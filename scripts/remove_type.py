from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import re
import shutil

ROOT = Path("/inspire/hdd/project/robot-decision/xiaoyunxiao-240108120113/swm/eval_results/gemini-3-flash-preview/bridgedata_v2")
MAX_WORKERS = 1000
OBJECTS_RE = re.compile(r"(\(\s*:objects\b)(.*?)(\n\s*\))", re.S)
"""
process() 里就是一条线：
找最大 round
没 round 就删 episode
没 problem.pddl 就删 episode, 有的话就删 (:objects ...) 里的类型标注
"""

def process(ep_dir: Path):
    try:
        rounds = [
            d for d in ep_dir.iterdir()
            if d.is_dir() and d.name.startswith("round") and d.name[5:].isdigit()
        ]
        if not rounds:
            shutil.rmtree(ep_dir)
            return ep_dir.name, "deleted_no_round"

        pddl_path = max(rounds, key=lambda d: int(d.name[5:])) / "problem.pddl"
        if not pddl_path.is_file():
            shutil.rmtree(ep_dir)
            return ep_dir.name, "deleted_no_problem"

        text = pddl_path.read_text(encoding="utf-8")
        m = OBJECTS_RE.search(text)
        if not m:
            return ep_dir.name, "unchanged"

        new_body = re.sub(r"\s*-\s*[^\s()]+", "", m.group(2))
        if new_body == m.group(2):
            return ep_dir.name, "unchanged"

        new_text = text[:m.start(2)] + new_body + text[m.end(2):]
        pddl_path.write_text(new_text, encoding="utf-8")
        return ep_dir.name, "modified"

    except Exception as e:
        return ep_dir.name, f"error: {e}"


def main():
    episodes = [d for d in ROOT.iterdir() if d.is_dir() and d.name.startswith("episode")]
    stats = dict(modified=0, unchanged=0, deleted_no_round=0, deleted_no_problem=0, error=0)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for ep_name, status in ex.map(process, episodes):
            if status in stats:
                stats[status] += 1
                if status == "modified":
                    print(ep_name)
                elif status.startswith("deleted"):
                    print(f"[DELETE] {ep_name} -> {status}")
            else:
                print(f"[ERROR] {ep_name}: {status}")
                stats["error"] += 1

    print("\nDone.")
    for k, v in stats.items():
        print(f"{k:<18}: {v}")


if __name__ == "__main__":
    main()