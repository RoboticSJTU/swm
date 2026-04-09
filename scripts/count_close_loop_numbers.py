from pathlib import Path
import re

root = Path("/home/xyx/下载/swm/eval_results/gemini-3-flash-preview/human")

task_dirs = sorted(
    [p for p in root.glob("task_*") if p.is_dir()],
    key=lambda p: int(re.search(r"\d+$", p.name).group())
)

total = 0

for task_dir in task_dirs:
    log_path = task_dir / "pair_debug.log"
    if not log_path.exists():
        continue

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    count = text.count("[history_check] False")
    total += count

    if count > 0:
        print(f"{task_dir.name}: {count}")

print(f"\n总计: {total}")