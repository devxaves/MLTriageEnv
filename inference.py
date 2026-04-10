#!/usr/bin/env python3
"""Autonomous inference loop for MLTriageEnv.

Emits strict evaluator-compatible logs:
- [START] task=<task_name> env=ml_triage_env model=<model_name>
- [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
- [END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from openai import OpenAI

from client import MLTriageEnv
from models import MLTriageAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or ""
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

TASK_TYPES = ["config", "logs", "pipeline"]
EPISODES_PER_TASK = 3
BENCHMARK_NAME = "ml_triage_env"


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _format_reward(value: float) -> str:
    return f"{value:.2f}"


def _single_line(value: str) -> str:
    return " ".join(str(value).split())


def _action_str(action: MLTriageAction) -> str:
    target = _single_line(action.target)
    value = _single_line(action.value)
    if value:
        return f"{action.action_type}('{target}','{value}')"
    return f"{action.action_type}('{target}')"


def _print_start(task_name: str) -> None:
    print(f"[START] task={task_name} env={BENCHMARK_NAME} model={MODEL_NAME}", flush=True)


def _print_step(step: int, action: MLTriageAction, reward: float, done: bool, error: str | None) -> None:
    error_value = "null" if error in (None, "") else _single_line(error)
    print(
        f"[STEP] step={step} action={_action_str(action)} reward={_format_reward(reward)} "
        f"done={_bool_str(done)} error={error_value}",
        flush=True,
    )


def _print_end(success: bool, rewards: List[float]) -> None:
    rewards_csv = ",".join(_format_reward(r) for r in rewards)
    print(
        f"[END] success={_bool_str(success)} steps={len(rewards)} rewards={rewards_csv}",
        flush=True,
    )


def _make_client() -> OpenAI:
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def _obs_to_dict(obs: Any) -> Dict[str, Any]:
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if isinstance(obs, dict):
        return obs
    return {}


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object in model response")
    return json.loads(text[start : end + 1])


def _next_action(openai_client: OpenAI, observation: Dict[str, Any]) -> MLTriageAction:
    system_prompt = (
        "You are an autonomous ML triage agent. Output exactly one JSON object matching "
        "MLTriageAction with keys: action_type, target, value, metadata. "
        "Valid action_type values are: inspect, diagnose, patch, fix_stage, validate, done. "
        "Do not include markdown or explanations."
    )
    user_prompt = (
        "Given this observation JSON, produce the next action as valid JSON only.\n"
        f"observation={json.dumps(observation, ensure_ascii=True)}"
    )

    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=220,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        parsed = _extract_json_object(content)
        return MLTriageAction(**parsed)
    except Exception:
        # Required fallback behavior on parse/model errors.
        return MLTriageAction(
            action_type="error",
            target="syntax",
            value="invalid json",
            metadata={},
        )


def run_episode(openai_client: OpenAI, task_type: str, episode_idx: int) -> None:
    _print_start(task_type)
    rewards: List[float] = []
    success = False

    with MLTriageEnv(base_url=ENV_URL).sync() as env:
        observation = env.reset(task_type=task_type, seed=1000 + episode_idx)
        obs_data = _obs_to_dict(observation)
        max_steps = int(obs_data.get("max_steps", 20) or 20)
        done = bool(obs_data.get("done", False))

        step = 0
        while not done and step < max_steps:
            step += 1
            action = _next_action(openai_client, obs_data)

            step_error: str | None = None
            try:
                observation = env.step(action)
                obs_data = _obs_to_dict(observation)
                reward = float(obs_data.get("reward", 0.0) or 0.0)
                done = bool(obs_data.get("done", False))
            except Exception as exc:
                reward = 0.0
                done = True
                step_error = str(exc)

            if step_error is None:
                metadata = obs_data.get("metadata", {}) if isinstance(obs_data.get("metadata"), dict) else {}
                step_error = obs_data.get("last_action_error") or metadata.get("last_action_error")

            rewards.append(reward)
            _print_step(step=step, action=action, reward=reward, done=done, error=step_error)

        success = done

    _print_end(success=success, rewards=rewards)


def main() -> None:
    openai_client = _make_client()
    for task_type in TASK_TYPES:
        for episode_idx in range(EPISODES_PER_TASK):
            run_episode(openai_client=openai_client, task_type=task_type, episode_idx=episode_idx)


if __name__ == "__main__":
    main()
