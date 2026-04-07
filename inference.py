from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Optional

from openai import OpenAI

from env import CodeReviewEnv


def _system_prompt() -> str:
    return (
        "You are a senior software engineer performing code review. "
        "Return ONLY valid JSON with this schema: "
        "{issues:[{line_number:int,severity:'critical|major|minor|nit',"
        "category:'bug|security|performance|style|logic',description:str}],"
        "overall_score:int(1-10),requires_changes:bool,summary:str}. "
        "Use line numbers from the provided snippet."
    )


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("empty model response")

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise
        return json.loads(match.group(0))


def _empty_action() -> Dict[str, Any]:
    return {
        "issues": [],
        "overall_score": 5,
        "requires_changes": True,
        "summary": "Failed to parse model output.",
    }


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _compact_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, separators=(",", ":"), sort_keys=True)


def run_episode_for_task(
    env: CodeReviewEnv,
    client: Optional[OpenAI],
    model_name: str,
    task_id: int,
    dry_run: bool = False,
) -> None:
    observation = env.reset(task_id=task_id)
    print(f"[START] task={task_id} step={observation['step_num']}")

    done = False
    episode_reward = 0.0
    step_idx = 0

    while not done:
        step_idx += 1
        user_payload = {
            "observation": observation,
            "instructions": "Return only JSON action."
        }
        if dry_run:
            action = _empty_action()
        else:
            if client is None:
                raise RuntimeError("OpenAI client is required when dry_run is disabled")
            response = client.chat.completions.create(
                model=model_name,
                temperature=0,
                timeout=40,
                messages=[
                    {"role": "system", "content": _system_prompt()},
                    {"role": "user", "content": json.dumps(user_payload)},
                ],
            )
            raw_text = response.choices[0].message.content or ""

            try:
                action = _extract_json(raw_text)
            except Exception:
                action = _empty_action()

        transition = env.step(action)
        reward = float(transition["reward"])
        done = bool(transition["done"])
        observation = transition["observation"]
        episode_reward += reward

        print(
            "[STEP] "
            f"task={task_id} step={step_idx} reward={reward:.4f} done={_bool_text(done)} "
            f"issues={len(action.get('issues', []))} "
            f"breakdown={_compact_json(transition['info'].get('reward_breakdown', {}))}"
        )

    print(f"[END] task={task_id} total_reward={episode_reward:.4f} steps={step_idx}")


def main() -> None:
    api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    hf_token = os.getenv("HF_TOKEN")
    # Optional variable used by some OpenEnv runners when loading local images.
    local_image_name = os.getenv("LOCAL_IMAGE_NAME")
    dry_run = os.getenv("INFERENCE_DRY_RUN", "0") == "1"
    _ = local_image_name

    if not dry_run and not hf_token:
        raise RuntimeError("Set HF_TOKEN before running inference.py")

    client = OpenAI(base_url=api_base_url, api_key=hf_token) if not dry_run else None
    env = CodeReviewEnv()

    started = time.time()
    for task_id in (1, 2, 3):
        run_episode_for_task(
            env=env,
            client=client,
            model_name=model_name,
            task_id=task_id,
            dry_run=dry_run,
        )

    elapsed = time.time() - started
    print(f"[END] all_tasks_completed elapsed_seconds={elapsed:.2f}")


if __name__ == "__main__":
    main()
