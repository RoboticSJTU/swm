import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


def build_sharegpt_json(prompt_for_training: str, image_path: Path, domain_str: str, problem_str: str) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": f"<image>{prompt_for_training}"},
            {"role": "assistant", "content": json.dumps({"domain": domain_str, "problem": problem_str}, ensure_ascii=False)},
        ],
        "images": [str(image_path)],
    }