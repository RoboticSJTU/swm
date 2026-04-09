from pathlib import Path
import subprocess

fast_downward_path = Path(__file__).parent.parent.parent.parent.parent / "downward" / "fast-downward.py"
def solve_pddl(domain_file, problem_file):
    """求解PDDL，并返回True或False来判断是否求解成功，且支持并行化求解"""

    plan_file = Path(domain_file).parent / "plan.txt"
    error_file = Path(domain_file).parent / "error.log"

    cmd = [
        "python",
        fast_downward_path,
        "--plan-file",
        plan_file,
        domain_file,
        problem_file,
        "--search",
        "astar(lmcut())"
    ]
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=3000,
            cwd=domain_file.parent,
        )
        return True

    except Exception as e:
        with open(error_file, "w", encoding="utf-8") as f:
            f.write(e.stdout)
        return False