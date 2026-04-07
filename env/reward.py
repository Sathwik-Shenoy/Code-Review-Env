from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple, TypedDict, cast

SEVERITY_ORDER = {"nit": 0, "minor": 1, "major": 2, "critical": 3}


class ReviewIssue(TypedDict):
    line_number: int
    severity: str
    category: str
    description: str


@dataclass
class RewardBreakdown:
    bug_detection: float
    false_positive_penalty: float
    severity_calibration: float
    overall_score_accuracy: float
    no_action_penalty: float
    revision_bonus: float
    total: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "bug_detection": round(self.bug_detection, 4),
            "false_positive_penalty": round(self.false_positive_penalty, 4),
            "severity_calibration": round(self.severity_calibration, 4),
            "overall_score_accuracy": round(self.overall_score_accuracy, 4),
            "no_action_penalty": round(self.no_action_penalty, 4),
            "revision_bonus": round(self.revision_bonus, 4),
            "total": round(self.total, 4),
        }


def _safe_issues(action: Dict[str, Any]) -> List[ReviewIssue]:
    issues = action.get("issues", [])
    if not isinstance(issues, list):
        return []
    safe: List[ReviewIssue] = []
    for item in issues:
        if not isinstance(item, dict):
            continue
        issue = cast(Dict[str, Any], item)
        line = issue.get("line_number")
        if not isinstance(line, int):
            continue
        safe.append(
            {
                "line_number": line,
                "severity": str(issue.get("severity", "minor")).lower(),
                "category": str(issue.get("category", "style")).lower(),
                "description": str(issue.get("description", "")).strip(),
            }
        )
    return safe


def _issue_match(pred: ReviewIssue, truth: Dict[str, Any]) -> bool:
    line_ok = abs(pred["line_number"] - int(truth["line_number"])) <= 2
    category_ok = pred["category"] == str(truth["category"]).lower()
    return line_ok and category_ok


def _best_matches(
    predicted: List[ReviewIssue],
    truth_issues: List[Dict[str, Any]],
) -> Tuple[List[Tuple[ReviewIssue, Dict[str, Any]]], List[ReviewIssue]]:
    used_truth: Set[int] = set()
    matches: List[Tuple[ReviewIssue, Dict[str, Any]]] = []
    unmatched_pred: List[ReviewIssue] = []

    for pred in predicted:
        found_index = None
        for idx, truth in enumerate(truth_issues):
            if idx in used_truth:
                continue
            if _issue_match(pred, truth):
                found_index = idx
                break
        if found_index is None:
            unmatched_pred.append(pred)
            continue
        used_truth.add(found_index)
        matches.append((pred, truth_issues[found_index]))

    return matches, unmatched_pred


def compute_shaped_reward(
    action: Dict[str, Any],
    truth_issues: List[Dict[str, Any]],
    expected_overall_score: int,
    revision_bonus: float = 0.0,
) -> RewardBreakdown:
    predicted = _safe_issues(action)
    gt_count = max(1, len(truth_issues))
    matches, unmatched_pred = _best_matches(predicted, truth_issues)

    bug_detection = 0.4 * (len(matches) / gt_count)

    false_critical = [
        p
        for p in unmatched_pred
        if p.get("severity") == "critical" and p.get("category") in {"bug", "security", "logic"}
    ]
    false_positive_penalty = min(0.15, 0.05 * len(false_critical))

    if matches:
        severity_points = 0.0
        for pred, truth in matches:
            p = SEVERITY_ORDER.get(pred.get("severity", "minor"), 1)
            t = SEVERITY_ORDER.get(str(truth.get("severity", "minor")).lower(), 1)
            diff = abs(p - t)
            if diff == 0:
                severity_points += 1.0
            elif diff == 1:
                severity_points += 0.5
        severity_calibration = 0.2 * (severity_points / len(matches))
    else:
        severity_calibration = 0.0

    predicted_overall = action.get("overall_score")
    if isinstance(predicted_overall, int):
        mse = float((predicted_overall - expected_overall_score) ** 2)
        overall_score_accuracy = 0.15 * max(0.0, 1.0 - mse / 81.0)
    else:
        overall_score_accuracy = 0.0

    no_action_penalty = -0.3 if not predicted and truth_issues else 0.0

    raw_total = (
        bug_detection
        - false_positive_penalty
        + severity_calibration
        + overall_score_accuracy
        + no_action_penalty
        + revision_bonus
    )
    total = max(0.0, min(1.0, raw_total))

    return RewardBreakdown(
        bug_detection=bug_detection,
        false_positive_penalty=false_positive_penalty,
        severity_calibration=severity_calibration,
        overall_score_accuracy=overall_score_accuracy,
        no_action_penalty=no_action_penalty,
        revision_bonus=revision_bonus,
        total=total,
    )
