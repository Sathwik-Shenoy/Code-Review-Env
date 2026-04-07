from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from env import CodeReviewEnv

app = FastAPI(title="CodeReviewEnv", version="1.0.0")
env = CodeReviewEnv()


class Issue(BaseModel):
    line_number: int
    severity: Literal["critical", "major", "minor", "nit"]
    category: Literal["bug", "security", "performance", "style", "logic"]
    description: str


class ActionModel(BaseModel):
    issues: List[Issue] = Field(default_factory=list)
    overall_score: int = Field(ge=1, le=10, default=5)
    requires_changes: bool = True
    summary: str = ""


class ObservationModel(BaseModel):
    code_snippet: str
    context: str
    diff_mode: bool
    history: List[Dict[str, Any]]
    task_id: int
    step_num: int


class RewardModel(BaseModel):
    reward: float
    done: bool
    breakdown: Dict[str, float]


class StepResponse(BaseModel):
    observation: ObservationModel
    reward: RewardModel
    info: Dict[str, Any]


class ResetRequest(BaseModel):
    task_id: Optional[int] = Field(default=None)
    seed: Optional[int] = Field(default=None)


class ResetResponse(BaseModel):
    observation: ObservationModel


class StateResponse(BaseModel):
    observation: ObservationModel


@app.post("/reset", response_model=ResetResponse)
def reset(req: Optional[ResetRequest] = None) -> ResetResponse:
    task_id = req.task_id if req is not None else None
    seed = req.seed if req is not None else None
    observation = env.reset(task_id=task_id, seed=seed)
    return ResetResponse(observation=ObservationModel(**observation))


@app.post("/step", response_model=StepResponse)
def step(action: ActionModel) -> StepResponse:
    result = env.step(action.model_dump())
    reward_obj = RewardModel(
        reward=result["reward"],
        done=result["done"],
        breakdown=result["info"].get("reward_breakdown", {}),
    )
    return StepResponse(
        observation=ObservationModel(**result["observation"]),
        reward=reward_obj,
        info=result["info"],
    )


@app.post("/state", response_model=StateResponse)
def state() -> StateResponse:
    observation = env.state()
    return StateResponse(observation=ObservationModel(**observation))


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse(_demo_html())


@app.get("/{full_path:path}", response_class=HTMLResponse)
def spa_fallback(full_path: str) -> HTMLResponse:
  # Keep browser navigation resilient on Space URLs and stale subpaths.
  return HTMLResponse(_demo_html())


def _demo_html() -> str:
    return """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>CodeReviewEnv Demo</title>
  <style>
    :root {
      --bg: #0a0f14;
      --panel: #121a22;
      --panel-alt: #0f151d;
      --text: #e8f0ff;
      --muted: #9fb0c8;
      --accent: #1fb6ff;
      --ok: #20bf6b;
      --warn: #f7b731;
      --danger: #eb4d4b;
      --border: #273242;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
      background: radial-gradient(circle at 20% 10%, #13293f 0%, #0a0f14 45%), linear-gradient(160deg, #080d13 0%, #0f1720 100%);
      color: var(--text);
    }
    .layout {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 300px 1fr;
    }
    .sidebar {
      background: linear-gradient(180deg, #0f1720 0%, #0b1118 100%);
      border-right: 1px solid var(--border);
      padding: 20px;
    }
    .brand { font-size: 1.2rem; font-weight: 700; letter-spacing: 0.5px; }
    .meta { color: var(--muted); margin-top: 4px; margin-bottom: 18px; font-size: 0.9rem; }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      margin-bottom: 12px;
    }
    .label { color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; }
    .value { font-size: 1.2rem; margin-top: 4px; font-weight: 600; }
    .controls { display: grid; gap: 8px; margin-top: 12px; }
    button {
      border: 1px solid var(--border);
      background: #16212d;
      color: var(--text);
      border-radius: 10px;
      padding: 10px 12px;
      cursor: pointer;
      font-weight: 600;
      transition: transform .15s ease, background .15s ease;
    }
    button:hover { transform: translateY(-1px); background: #1a2a3a; }
    .main { padding: 24px; }
    .panel {
      background: rgba(15, 21, 29, 0.88);
      border: 1px solid var(--border);
      border-radius: 14px;
      margin-bottom: 16px;
      overflow: hidden;
      animation: fadeIn .35s ease;
    }
    .panel h3 {
      margin: 0;
      padding: 12px 14px;
      font-size: 0.95rem;
      background: var(--panel-alt);
      border-bottom: 1px solid var(--border);
      color: #b7caf0;
    }
    .content { padding: 14px; }
    pre {
      margin: 0;
      max-height: 420px;
      overflow: auto;
      background: #0a0f14;
      border: 1px solid #1d2632;
      border-radius: 10px;
      padding: 12px;
      font-family: 'IBM Plex Mono', ui-monospace, monospace;
      font-size: 13px;
      line-height: 1.45;
      white-space: pre;
    }
    textarea {
      width: 100%;
      min-height: 220px;
      background: #0a0f14;
      color: #e6f1ff;
      border: 1px solid #1d2632;
      border-radius: 10px;
      padding: 10px;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 13px;
    }
    details {
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #101922;
      margin-top: 10px;
    }
    summary { padding: 10px 12px; cursor: pointer; color: #c2d4ef; }
    .reward-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
      gap: 8px;
    }
    .pill {
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 8px;
      background: #0f1722;
      font-size: 0.85rem;
    }
    .status-ok { color: var(--ok); }
    .status-warn { color: var(--warn); }
    .status-bad { color: var(--danger); }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar { border-right: none; border-bottom: 1px solid var(--border); }
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(4px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>
  <div class=\"layout\">
    <aside class=\"sidebar\">
      <div class=\"brand\">CodeReviewEnv</div>
      <div class=\"meta\">OpenEnv-compatible code review simulator</div>

      <div class=\"card\">
        <div class=\"label\">Task</div>
        <div id=\"taskVal\" class=\"value\">-</div>
      </div>
      <div class=\"card\">
        <div class=\"label\">Step</div>
        <div id=\"stepVal\" class=\"value\">-</div>
      </div>
      <div class=\"card\">
        <div class=\"label\">Last Reward</div>
        <div id=\"rewardVal\" class=\"value\">-</div>
      </div>
      <div class=\"controls\">
        <button onclick=\"resetEnv(1)\">Reset Task 1</button>
        <button onclick=\"resetEnv(2)\">Reset Task 2</button>
        <button onclick=\"resetEnv(3)\">Reset Task 3</button>
        <button onclick=\"submitAction()\">Submit Review</button>
      </div>
    </aside>

    <main class=\"main\">
      <section class=\"panel\">
        <h3>Observation: Context</h3>
        <div class=\"content\"><div id=\"ctx\">Press a reset button to start.</div></div>
      </section>

      <section class=\"panel\">
        <h3>Observation: Code Snippet</h3>
        <div class=\"content\"><pre id=\"codeView\"></pre></div>
      </section>

      <section class=\"panel\">
        <h3>Action JSON</h3>
        <div class=\"content\">
<textarea id=\"actionInput\">{
  "issues": [
    {
      "line_number": 1,
      "severity": "major",
      "category": "logic",
      "description": "Describe issue here"
    }
  ],
  "overall_score": 4,
  "requires_changes": true,
  "summary": "Short review summary"
}</textarea>
        </div>
      </section>

      <section class=\"panel\">
        <h3>Reward Breakdown</h3>
        <div class=\"content\">
          <div id=\"rewardGrid\" class=\"reward-grid\"></div>
          <details>
            <summary>Observation / Action / Response JSON</summary>
            <div class=\"content\">
              <pre id=\"jsonView\"></pre>
            </div>
          </details>
        </div>
      </section>
    </main>
  </div>

  <script>
    let lastObservation = null;
    let lastAction = null;
    let lastResponse = null;

    function renderSidebar(task, step, reward) {
      document.getElementById('taskVal').textContent = task ?? '-';
      document.getElementById('stepVal').textContent = step ?? '-';
      document.getElementById('rewardVal').textContent = reward != null ? Number(reward).toFixed(4) : '-';
    }

    function renderReward(breakdown = {}) {
      const grid = document.getElementById('rewardGrid');
      grid.innerHTML = '';
      const entries = Object.entries(breakdown);
      if (!entries.length) {
        grid.innerHTML = '<div class="pill">No reward yet</div>';
        return;
      }
      for (const [k, v] of entries) {
        const el = document.createElement('div');
        el.className = 'pill';
        el.textContent = `${k}: ${Number(v).toFixed(4)}`;
        grid.appendChild(el);
      }
    }

    function renderJson() {
      const view = document.getElementById('jsonView');
      const payload = {
        observation: lastObservation,
        action: lastAction,
        response: lastResponse,
      };
      view.textContent = JSON.stringify(payload, null, 2);
    }

    function renderObservation(obs) {
      lastObservation = obs;
      document.getElementById('ctx').textContent = obs.context || '';
      document.getElementById('codeView').textContent = obs.code_snippet || '';
      renderSidebar(obs.task_id, obs.step_num, null);
      renderJson();
    }

    async function resetEnv(taskId) {
      const res = await fetch('/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_id: taskId })
      });
      const data = await res.json();
      lastResponse = data;
      renderObservation(data.observation);
      renderReward({});
    }

    async function submitAction() {
      let action;
      try {
        action = JSON.parse(document.getElementById('actionInput').value);
      } catch (err) {
        alert('Invalid JSON action: ' + err.message);
        return;
      }

      lastAction = action;
      const res = await fetch('/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(action)
      });
      const data = await res.json();
      lastResponse = data;
      renderObservation(data.observation);
      renderSidebar(data.observation.task_id, data.observation.step_num, data.reward.reward);
      renderReward(data.reward.breakdown || {});
      renderJson();
    }
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
