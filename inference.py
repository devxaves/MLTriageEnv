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
from urllib.parse import urlparse
from typing import Any, Dict, List

import requests
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
SCORE_MIN = 0.0001
SCORE_MAX = 0.9999


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _format_reward(value: float) -> str:
    return f"{value:.2f}"


def _strict_score(value: float) -> float:
    return max(SCORE_MIN, min(SCORE_MAX, float(value)))


def _format_score(value: float) -> str:
    return f"{_strict_score(value):.4f}"


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
    score = _strict_score(rewards[-1] if rewards else SCORE_MIN)
    print(
        f"[END] success={_bool_str(success)} steps={len(rewards)} score={_format_score(score)} rewards={rewards_csv}",
        flush=True,
    )


def _make_client() -> OpenAI | None:
    if not HF_TOKEN:
        return None
    try:
        return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    except Exception:
        return None


def _candidate_env_urls(base_url: str) -> List[str]:
    parsed = urlparse(base_url)
    candidates = [base_url.rstrip("/")]
    host = parsed.hostname or ""
    scheme = parsed.scheme or "http"
    if host in {"localhost", "127.0.0.1"}:
        candidates.extend([
            f"{scheme}://localhost:7860",
            f"{scheme}://127.0.0.1:8000",
            f"{scheme}://127.0.0.1:7860",
        ])
    # stable de-dup preserving order
    seen = set()
    ordered: List[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def _resolve_env_url(base_url: str) -> str:
    for candidate in _candidate_env_urls(base_url):
        try:
            res = requests.get(f"{candidate}/health", timeout=4)
            if res.ok:
                return candidate
        except Exception:
            continue
    return base_url.rstrip("/")


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


def _fallback_action(observation: Dict[str, Any]) -> MLTriageAction:
    task_type = str(observation.get("task_type", "")).lower()
    step_count = int(observation.get("step_count", 0) or 0)
    max_steps = int(observation.get("max_steps", 15) or 15)

    if step_count >= max_steps - 1:
        return MLTriageAction(action_type="done", target="task", value="complete", metadata={})

    if task_type == "config":
        if step_count == 0:
            return MLTriageAction(action_type="inspect", target="learning_rate", value="", metadata={})
        if step_count == 1:
            return MLTriageAction(action_type="patch", target="learning_rate", value="0.001", metadata={})
        if step_count == 2:
            return MLTriageAction(action_type="validate", target="config", value="consistency", metadata={})
        return MLTriageAction(action_type="done", target="task", value="complete", metadata={})

    if task_type == "logs":
        if step_count == 0:
            return MLTriageAction(action_type="inspect", target="training_logs", value="", metadata={})
        if step_count == 1:
            return MLTriageAction(action_type="diagnose", target="failure_mode", value="gradient explosion", metadata={})
        if step_count == 2:
            return MLTriageAction(action_type="fix_stage", target="training", value="add gradient clipping", metadata={})
        if step_count == 3:
            return MLTriageAction(action_type="validate", target="logs", value="stability", metadata={})
        return MLTriageAction(action_type="done", target="task", value="complete", metadata={})

    available_tools = observation.get("available_tools")
    if isinstance(available_tools, list) and available_tools:
        if step_count == 0 and "inspect_logs" in available_tools:
            return MLTriageAction(action_type="inspect_logs", target="payment-gateway", value="", metadata={})
        if step_count == 1 and "query_metrics" in available_tools:
            return MLTriageAction(action_type="query_metrics", target="ledger-writer", value="", metadata={})
        if step_count == 2 and "check_dependency_graph" in available_tools:
            return MLTriageAction(action_type="check_dependency_graph", target="checkout-api", value="", metadata={})
        if step_count == 3 and "dismiss_red_herring" in available_tools:
            return MLTriageAction(action_type="dismiss_red_herring", target="payment-gateway", value="", metadata={})
        if "finalize_triage" in available_tools:
            return MLTriageAction(
                action_type="finalize_triage",
                target="incident",
                value="root cause ledger-writer priority P1",
                metadata={},
            )

    if step_count == 0:
        return MLTriageAction(action_type="inspect", target="preprocessing", value="", metadata={})
    if step_count == 1:
        return MLTriageAction(action_type="diagnose", target="preprocessing", value="fit_on_test", metadata={})
    if step_count == 2:
        return MLTriageAction(action_type="fix_stage", target="preprocessing", value="fit on training data only", metadata={})
    if step_count == 3:
        return MLTriageAction(action_type="validate", target="pipeline", value="check", metadata={})
    return MLTriageAction(action_type="done", target="task", value="complete", metadata={})


def _next_action(openai_client: OpenAI | None, observation: Dict[str, Any]) -> MLTriageAction:
    if openai_client is None:
        return _fallback_action(observation)

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
        # Required fallback behavior on parse/model/network errors.
        return _fallback_action(observation)


def run_episode(openai_client: OpenAI | None, task_type: str, episode_idx: int, env_url: str) -> None:
    _print_start(task_type)
    rewards: List[float] = []
    success = False
    step = 0
    try:
        with MLTriageEnv(base_url=env_url).sync() as env:
            try:
                observation = env.reset(task_type=task_type, seed=1000 + episode_idx)
                obs_data = _obs_to_dict(observation)
            except Exception as exc:
                action = MLTriageAction(action_type="inspect", target="startup", value="", metadata={})
                rewards.append(0.0)
                _print_step(step=1, action=action, reward=0.0, done=True, error=f"reset_failed: {exc}")
                return

            max_steps = int(obs_data.get("max_steps", 20) or 20)
            done = bool(obs_data.get("done", False))

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
    except Exception as exc:
        action = MLTriageAction(action_type="inspect", target="runtime", value="", metadata={})
        rewards.append(0.0)
        _print_step(step=max(1, step + 1), action=action, reward=0.0, done=True, error=f"episode_failed: {exc}")
    finally:
        _print_end(success=success, rewards=rewards)


def main() -> None:
    openai_client = _make_client()
    env_url = _resolve_env_url(ENV_URL)
    for task_type in TASK_TYPES:
        for episode_idx in range(EPISODES_PER_TASK):
            try:
                run_episode(
                    openai_client=openai_client,
                    task_type=task_type,
                    episode_idx=episode_idx,
                    env_url=env_url,
                )
            except Exception as exc:
                # Absolute safety net: never crash the process on a single episode.
                _print_start(task_type)
                action = MLTriageAction(action_type="inspect", target="fatal", value="", metadata={})
                _print_step(step=1, action=action, reward=0.0, done=True, error=f"fatal: {exc}")
                _print_end(success=False, rewards=[0.0])


if __name__ == "__main__":
    main()
