---
title: CodeReviewEnv
emoji: "🧪"
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# CodeReviewEnv

CodeReviewEnv is an OpenEnv-compliant reinforcement learning environment for training and evaluating AI agents on **realistic software code review**.

## Why this task matters

Code review is a daily engineering responsibility with high practical value:
- It tests comprehension of code intent and context.
- It requires finding bugs, security risks, logic mistakes, and performance issues.
- It demands severity triage, not just issue spotting.
- It mirrors real PR workflows where context evolves over time.

This makes code review a strong benchmark for coding agents beyond pass/fail execution tasks.

## Project Layout

```text
codereviewenv/
├── env/
│   ├── __init__.py
│   ├── environment.py
│   ├── graders.py
│   ├── reward.py
│   └── snippets/
│       ├── *.py
│       └── *.json
├── server.py
├── inference.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Observation Space

Each observation is a JSON object:

| Field | Type | Description |
|---|---|---|
| `code_snippet` | `str` | Python code or diff-style snippet (15-80 lines) |
| `context` | `str` | Task intent and runtime context |
| `diff_mode` | `bool` | `true` for git-diff style review mode |
| `history` | `list[dict]` | Prior actions submitted in current episode |
| `task_id` | `int` | Active task (1, 2, or 3) |
| `step_num` | `int` | Current step in episode |

## Action Space

Agent submits a JSON-serializable review action:

| Field | Type | Description |
|---|---|---|
| `issues` | `list[issue]` | Structured issue reports |
| `overall_score` | `int` (`1..10`) | PR quality / approval confidence |
| `requires_changes` | `bool` | Whether PR should be blocked |
| `summary` | `str` | Concise review summary |

`issue` format:

```json
{
  "line_number": 17,
  "severity": "major",
  "category": "logic",
  "description": "Off-by-one slice drops final record"
}
```

Allowed `severity`: `critical | major | minor | nit`  
Allowed `category`: `bug | security | performance | style | logic`

## Tasks and Difficulty

### Task 1: Single Function Review (easy)
- One standalone function.
- 1-2 seeded issues (off-by-one, missing null check, resource handling).
- Single-step review.
- Expected baseline score: `~0.55-0.70`.

### Task 2: Class-Level Diff Review (medium)
- Git-diff style class changes.
- 3-5 issues across methods (security + logic + performance patterns).
- Single-step review.
- Expected baseline score: `~0.35-0.50`.

### Task 3: Multi-Turn Adversarial Review (hard)
- Module reviewed across 3 steps.
- Extra context is revealed each step and can change risk profile.
- Agent should revise severity as context increases impact.
- Expected baseline score: `~0.20-0.35`.

## Reward Function

Reward is shaped with partial credit and capped to `[0.0, 1.0]`.

| Component | Range | Logic |
|---|---|---|
| Bug detection rate | `0.0..0.4` | Fraction of seeded issues found (line match within ±2 and category match) |
| False positive penalty | `0.0..-0.15` | Penalizes unmatched critical bug/security/logic claims |
| Severity calibration | `0.0..0.2` | Rewards correct triage (exact + near misses) |
| Overall score accuracy | `0.0..0.15` | MSE-shaped reward on `overall_score` vs ground truth |
| No-action penalty | `0.0..-0.3` | Heavy penalty if no issues submitted while issues exist |
| Revision bonus (task 3) | `0.0..0.05` | Bonus for updating severity correctly when context changes |

Total:

```text
total = clamp(0.0, 1.0,
  bug_detection
  - false_positive_penalty
  + severity_calibration
  + overall_score_accuracy
  + no_action_penalty
  + revision_bonus
)
```

## API

FastAPI server exposes OpenEnv-style endpoints:
- `POST /reset`
- `POST /step`
- `POST /state`

### Example reset

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 1}'
```

### Example step

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "issues":[{"line_number":7,"severity":"major","category":"logic","description":"Off-by-one"}],
    "overall_score":4,
    "requires_changes":true,
    "summary":"Has a correctness bug"
  }'
```

## Demo UI

Open `http://localhost:7860/` for an interactive environment demo:
- Dark sidebar: current task, step, reward.
- Main panel: context, syntax-friendly code panel, editable action JSON.
- Reward cards and collapsible Observation/Action/Response JSON viewer.

No authentication is required.

## Baseline Inference Script

`inference.py` runs one episode per task using OpenAI-compatible endpoints:
- Reads env vars: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`.
- Sends observation + strict JSON response instructions.
- Handles invalid JSON by submitting empty action and taking penalty.
- Emits required lifecycle logs: `[START]`, `[STEP]`, `[END]`.

### Environment Variables

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="<your-token>"
```

### Run inference

```bash
python inference.py
```

## Sample Logs

```text
[START] task=1 step=1
[STEP] task=1 step=1 reward=0.6420 done=True issues=1 breakdown={"bug_detection":0.4,"false_positive_penalty":0.0,"severity_calibration":0.2,"overall_score_accuracy":0.042,"no_action_penalty":0.0,"revision_bonus":0.0,"total":0.642}
[END] task=1 total_reward=0.6420 steps=1
[START] task=3 step=1
[STEP] task=3 step=1 reward=0.2100 done=False issues=1 breakdown={...}
[STEP] task=3 step=2 reward=0.3150 done=False issues=1 breakdown={...}
[STEP] task=3 step=3 reward=0.3680 done=True issues=1 breakdown={...}
[END] task=3 total_reward=0.8930 steps=3
[END] all_tasks_completed elapsed_seconds=123.41
```

## Docker

Build image:

```bash
docker build -t codereviewenv .
```

Run container:

```bash
docker run --rm -p 7860:7860 \
  -e API_BASE_URL="$API_BASE_URL" \
  -e MODEL_NAME="$MODEL_NAME" \
  -e HF_TOKEN="$HF_TOKEN" \
  codereviewenv
```

## Hugging Face Spaces Notes

- The included `Dockerfile` is compatible with Docker Spaces.
- App binds to `0.0.0.0:7860`.
- Environment logic and grading are fully local and deterministic.
- Designed to run within `2 vCPU / 8GB RAM` limits.

## Validation Checklist

- OpenEnv API surface: `reset`, `step`, `state`
- Local deterministic grading (no external calls in env logic)
- Multi-task curriculum (easy -> medium -> hard)
- Structured action schema with severity/category validation
- Baseline runner logs `[START] [STEP] [END]`
