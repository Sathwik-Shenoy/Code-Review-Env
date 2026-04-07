from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from .graders import grade_step


@dataclass
class SnippetCase:
    snippet_id: str
    code: str
    data: Dict[str, Any]


class CodeReviewEnv:
    def __init__(self, snippets_dir: Optional[Path] = None):
        self.snippets_dir = snippets_dir or Path(__file__).parent / "snippets"
        self._cases: Dict[int, List[SnippetCase]] = {1: [], 2: [], 3: []}
        self._load_cases()

        self.current_case: Optional[SnippetCase] = None
        self.current_task_id: Optional[int] = None
        self.step_num: int = 0
        self.max_steps: int = 0
        self.history: List[Dict[str, Any]] = []
        self.last_reward_breakdown: Dict[str, float] = {}

    def _load_cases(self) -> None:
        json_files = sorted(self.snippets_dir.glob("*.json"))
        if not json_files:
            raise RuntimeError(f"No snippet metadata found in {self.snippets_dir}")

        for meta_path in json_files:
            base = meta_path.stem
            code_path = self.snippets_dir / f"{base}.py"
            if not code_path.exists():
                raise RuntimeError(f"Missing code file for snippet {base}")

            with meta_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            code = code_path.read_text(encoding="utf-8")

            task_id = int(data["task_id"])
            if task_id not in self._cases:
                raise RuntimeError(f"Invalid task_id {task_id} in {meta_path.name}")

            self._cases[task_id].append(SnippetCase(snippet_id=base, code=code, data=data))

        for task_id, task_cases in self._cases.items():
            if not task_cases:
                raise RuntimeError(f"No snippets loaded for task {task_id}")

    def reset(self, task_id: Optional[int] = None, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            random.seed(seed)

        if task_id is None:
            task_id = random.choice([1, 2, 3])
        if task_id not in self._cases:
            raise ValueError("task_id must be one of: 1, 2, 3")

        self.current_task_id = task_id
        self.current_case = random.choice(self._cases[task_id])
        self.step_num = 1
        self.max_steps = 3 if task_id == 3 else 1
        self.history = []
        self.last_reward_breakdown = {}

        return self.state()

    def _build_context(self) -> str:
        if not self.current_case:
            return ""

        base_context = str(self.current_case.data.get("context", ""))
        staged = self.current_case.data.get("staged_context", [])
        if self.current_task_id == 3 and isinstance(staged, list) and staged:
            typed_staged = cast(List[str], staged)
            idx = min(self.step_num - 1, len(typed_staged) - 1)
            extra = typed_staged[idx]
            if extra:
                return f"{base_context}\n\nAdditional context: {extra}"
        return base_context

    def state(self) -> Dict[str, Any]:
        if not self.current_case or self.current_task_id is None:
            return {
                "code_snippet": "",
                "context": "Environment not initialized. Call reset().",
                "diff_mode": False,
                "history": [],
                "task_id": 0,
                "step_num": 0,
            }

        return {
            "code_snippet": self.current_case.code,
            "context": self._build_context(),
            "diff_mode": bool(self.current_case.data.get("diff_mode", False)),
            "history": list(self.history),
            "task_id": self.current_task_id,
            "step_num": self.step_num,
        }

    def step(self, action: object) -> Dict[str, Any]:
        if not self.current_case or self.current_task_id is None:
            raise RuntimeError("Call reset() before step()")

        if not isinstance(action, dict):
            action = {}

        result = grade_step(
            case_data=self.current_case.data,
            action=action,
            step_num=self.step_num,
            history=self.history,
        )

        self.history.append(action)
        self.last_reward_breakdown = result.breakdown

        done = self.step_num >= self.max_steps
        if not done:
            self.step_num += 1

        return {
            "observation": self.state(),
            "reward": result.reward,
            "done": done,
            "info": {
                "snippet_id": self.current_case.snippet_id,
                "matched_issue_count": result.matched_issue_count,
                "expected_issue_count": result.expected_issue_count,
                "reward_breakdown": result.breakdown,
            },
        }
