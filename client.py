from __future__ import annotations

from typing import Any, Dict, Optional

from env import CodeReviewEnv


class CodeReviewEnvClient:
    """Local client wrapper matching OpenEnv-style reset/step/state interactions."""

    def __init__(self, seed: Optional[int] = None):
        self._env = CodeReviewEnv()
        self._seed = seed

    def reset(self, task_id: Optional[int] = None) -> Dict[str, Any]:
        return self._env.reset(task_id=task_id, seed=self._seed)

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        return self._env.step(action)

    def state(self) -> Dict[str, Any]:
        return self._env.state()
