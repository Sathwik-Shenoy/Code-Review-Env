from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set, cast

from .reward import compute_shaped_reward


@dataclass
class GradeResult:
    reward: float
    breakdown: Dict[str, float]
    matched_issue_count: int
    expected_issue_count: int


def _safe_issues(action: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = action.get("issues", [])
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(cast(Dict[str, Any], item))
    return out


def _count_matches(predicted: List[Dict[str, Any]], truth_issues: List[Dict[str, Any]]) -> int:
    used: Set[int] = set()
    count = 0
    for pred in predicted:
        line = pred.get("line_number")
        category = str(pred.get("category", "")).lower()
        if not isinstance(line, int):
            continue
        for idx, truth in enumerate(truth_issues):
            if idx in used:
                continue
            if category != str(truth.get("category", "")).lower():
                continue
            if abs(int(truth.get("line_number", -999)) - line) <= 2:
                used.add(idx)
                count += 1
                break
    return count


def _apply_stage_overrides(case_data: Dict[str, Any], step_num: int) -> List[Dict[str, Any]]:
    issues = [dict(issue) for issue in case_data["seeded_issues"]]
    staged_targets = case_data.get("stage_severity_targets", [])
    if not isinstance(staged_targets, list):
        return issues
    typed_targets = cast(List[Dict[str, str]], staged_targets)

    for stage_idx in range(min(step_num, len(typed_targets))):
        overrides = typed_targets[stage_idx] or {}
        for issue in issues:
            key = f"{issue.get('category')}:{issue.get('line_number')}"
            if key in overrides:
                issue["severity"] = overrides[key]
    return issues


def _revision_bonus(
    action: Dict[str, Any],
    history: List[Dict[str, Any]],
    effective_truth: List[Dict[str, Any]],
) -> float:
    if not history:
        return 0.0

    prev_issues = history[-1].get("issues", [])
    if not isinstance(prev_issues, list):
        prev_issues = []
    current_issues = _safe_issues(action)

    target_map = {
        (str(t["category"]).lower(), int(t["line_number"])): str(t["severity"]).lower()
        for t in effective_truth
    }

    improved = 0
    for curr in current_issues:
        line = curr.get("line_number")
        cat = str(curr.get("category", "")).lower()
        if not isinstance(line, int):
            continue
        target = target_map.get((cat, line))
        if not target:
            continue

        prev_match = None
        for prev_item in prev_issues:
            if not isinstance(prev_item, dict):
                continue
            prev = cast(Dict[str, Any], prev_item)
            if prev.get("line_number") == line and str(prev.get("category", "")).lower() == cat:
                prev_match = str(prev.get("severity", "minor")).lower()
                break

        if prev_match is None:
            continue

        if prev_match != target and str(curr.get("severity", "")).lower() == target:
            improved += 1

    return min(0.05, improved * 0.025)


def grade_step(
    case_data: Dict[str, Any],
    action: Dict[str, Any],
    step_num: int,
    history: List[Dict[str, Any]],
) -> GradeResult:
    task_id = int(case_data["task_id"])
    expected_overall = int(case_data["expected_overall_score"])

    effective_truth = (
        _apply_stage_overrides(case_data, step_num)
        if task_id == 3
        else list(case_data["seeded_issues"])
    )
    revision_bonus = _revision_bonus(action, history, effective_truth) if task_id == 3 else 0.0

    breakdown = compute_shaped_reward(
        action=action,
        truth_issues=effective_truth,
        expected_overall_score=expected_overall,
        revision_bonus=revision_bonus,
    )

    predicted = _safe_issues(action)
    match_count = _count_matches(predicted, effective_truth)

    return GradeResult(
        reward=breakdown.total,
        breakdown=breakdown.as_dict(),
        matched_issue_count=match_count,
        expected_issue_count=len(effective_truth),
    )
