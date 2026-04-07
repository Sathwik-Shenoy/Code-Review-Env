from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class Issue(BaseModel):
    line_number: int
    severity: Literal["critical", "major", "minor", "nit"]
    category: Literal["bug", "security", "performance", "style", "logic"]
    description: str


class Action(BaseModel):
    issues: List[Issue] = Field(default_factory=list)
    overall_score: int = Field(ge=1, le=10, default=5)
    requires_changes: bool = True
    summary: str = ""


class Observation(BaseModel):
    code_snippet: str
    context: str
    diff_mode: bool
    history: List[Dict[str, Any]]
    task_id: int
    step_num: int


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]
