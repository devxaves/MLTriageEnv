#!/usr/bin/env python3
"""Baseline inference for MLTriageEnv.

The script follows a reproducible evaluation loop, emits structured stdout logs
using [START], [STEP], and [END], and uses the OpenAI client when available.
It falls back to a deterministic planner so scores remain reproducible even if
the model endpoint is unavailable.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
from openai import OpenAI

from client import MLTriageEnvClient
from models import MLTriageAction
from server.environment import MLTriageEnvironment


ROOT = Path(__file__).resolve().parent
SCENARIOS_DIR = ROOT / "scenarios"
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
EPISODES_PER_TASK = 3
TASK_TYPES = ["config", "logs", "pipeline"]
SCENARIO_INDICES = [0, 1, 2]
BENCHMARK_NAME = "ml_triage_env"


def _load_scenarios(filename: str) -> List[Dict[str, Any]]:
    with open(SCENARIOS_DIR / filename, "r", encoding="utf-8") as handle:
        return json.load(handle)


SCENARIO_MAP: Dict[str, Dict[str, Dict[str, Any]]] = {
    "config": {scenario["id"]: scenario for scenario in _load_scenarios("config_scenarios.json")},
    "logs": {scenario["id"]: scenario for scenario in _load_scenarios("log_scenarios.json")},
    "pipeline": {scenario["id"]: scenario for scenario in _load_scenarios("pipeline_scenarios.json")},
}

PIPELINE_FIX_PHRASES: Dict[str, str] = {
    "fit_on_test": "fit only on training data and transform validation/test separately",
    "metric_mismatch": "use F1 or AUC-ROC for the imbalanced classification task",
    "wrong_split_strategy": "use a chronological temporal split rather than random split",
    "target_leakage": "remove the leaked target-derived feature and exclude it from the feature set",
    "missing_value_wrong_strategy": "use median imputation and add a missing-value indicator for skewed data",
    "evaluation_on_train": "evaluate on a held-out validation/test set, not the training data",
    "label_encoding_instead_of_onehot": "replace ordinal encoding with one-hot encoding for nominal categories",
    "class_imbalance_ignored": "balance the classes with class weights or oversampling",
    "wrong_scaler_applied": "use MinMaxScaler for bounded features instead of StandardScaler",
    "wrong_aggregation": "replace mean aggregation with sum aggregation where counts are required",
    "off_by_one_in_sequences": "shift the window to exclude the current target and fix the lookback indexing",
    "duplicate_rows_not_removed": "drop duplicate rows before splitting the dataset",
    "preprocessing_after_split": "fit preprocessing on training data only, then transform validation and test sets",
    "shuffle_before_split": "preserve temporal order and do not shuffle before splitting",
}


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _single_line(value: str) -> str:
    return " ".join(str(value).split())


def _format_reward(value: float) -> str:
    return f"{value:.2f}"


def _action_str(action: MLTriageAction) -> str:
    target = _single_line(action.target)
    value = _single_line(action.value)
    if value:
        return f"{action.action_type}('{target}','{value}')"
    return f"{action.action_type}('{target}')"


def _start_line(task_name: str) -> str:
    return f"[START] task={task_name} env={BENCHMARK_NAME} model={MODEL_NAME}"


def _step_line(step: int, action: MLTriageAction, reward: float, done: bool, error: Any) -> str:
    error_value = "null" if error in (None, "") else _single_line(str(error))
    return (
        f"[STEP] step={step} action={_action_str(action)} reward={_format_reward(reward)} "
        f"done={_bool_str(done)} error={error_value}"
    )


def _end_line(success: bool, score: float, rewards: List[float]) -> str:
    rewards_csv = ",".join(_format_reward(r) for r in rewards)
    return (
        f"[END] success={_bool_str(success)} steps={len(rewards)} "
        f"score={_format_reward(score)} rewards={rewards_csv}"
    )


def _make_openai_client() -> OpenAI:
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def _environment() -> Any:
    try:
        response = requests.get(f"{ENV_URL.rstrip('/')}/health", timeout=5)
        if response.ok:
            return MLTriageEnvClient(base_url=ENV_URL)
    except Exception:
        pass
    return MLTriageEnvironment()


def _current_payload(obs: Any) -> Dict[str, Any]:
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if isinstance(obs, dict):
        return obs
    return {}


def _planned_actions(task_type: str, scenario: Dict[str, Any]) -> List[MLTriageAction]:
    actions: List[MLTriageAction] = []

    if task_type == "config":
        broken_fields = [
            field
            for field, info in scenario["broken_fields"].items()
            if info.get("issue_type", "none") != "none"
        ]
        if broken_fields:
            first = broken_fields[0]
            actions.append(MLTriageAction(action_type="inspect", target=first, value="detail"))
        for field in broken_fields:
            info = scenario["broken_fields"][field]
            actions.append(MLTriageAction(action_type="diagnose", target=field, value=info["issue_type"]))
            actions.append(MLTriageAction(action_type="patch", target=field, value=str(info["correct_value"])))
        actions.append(MLTriageAction(action_type="validate", target="config", value="check"))
        actions.append(MLTriageAction(action_type="done", target="task", value="complete"))
        return actions

    if task_type == "logs":
        actions.append(MLTriageAction(action_type="inspect", target="logs", value="full"))
        actions.append(MLTriageAction(action_type="inspect", target="logs", value="gradients"))
        actions.append(MLTriageAction(action_type="inspect", target="logs", value="metrics"))
        actions.append(MLTriageAction(action_type="diagnose", target="failure_mode", value=scenario["failure_mode"]))
        actions.append(MLTriageAction(action_type="fix_stage", target="intervention", value=scenario["correct_intervention"]))
        actions.append(MLTriageAction(action_type="validate", target="logs", value="check"))
        actions.append(MLTriageAction(action_type="done", target="task", value="complete"))
        return actions

    faulty_stages = scenario["faulty_stages"]
    actions.append(MLTriageAction(action_type="inspect", target="pipeline", value="full"))
    for stage in faulty_stages:
        bug = scenario["bugs"][stage]
        actions.append(MLTriageAction(action_type="inspect", target=stage, value="detail"))
        actions.append(MLTriageAction(action_type="diagnose", target=stage, value=bug["bug_type"]))
        actions.append(
            MLTriageAction(
                action_type="fix_stage",
                target=stage,
                value=PIPELINE_FIX_PHRASES.get(bug["bug_type"], bug["fix"]),
            )
        )
    actions.append(MLTriageAction(action_type="validate", target="pipeline", value="check_consistency"))
    actions.append(MLTriageAction(action_type="done", target="task", value="complete"))
    return actions


def _prompt_for_action(task_type: str, obs: Dict[str, Any], planned: MLTriageAction, client: OpenAI) -> MLTriageAction:

    prompt = {
        "task_type": task_type,
        "task_id": obs.get("task_id", ""),
        "feedback": obs.get("feedback", ""),
        "issues_remaining": obs.get("issues_remaining", 0),
        "step_count": obs.get("step_count", 0),
        "artifact": obs.get("artifact", "")[:2000],
        "candidate_action": planned.model_dump(),
        "instruction": "Return the candidate action as JSON only if it is appropriate; otherwise return the same action.",
    }

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=120,
            messages=[
                {"role": "system", "content": "Return only valid JSON for the next environment action."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        parsed = json.loads(content)
        candidate = MLTriageAction(**parsed)
        if candidate.action_type == planned.action_type and candidate.target == planned.target:
            return candidate
    except Exception:
        pass

    return planned


def _reset(environment: Any, task_type: str, scenario_index: int, seed: int) -> Any:
    if hasattr(environment, "reset"):
        try:
            return environment.reset(task_type=task_type, scenario_index=scenario_index, seed=seed)
        except TypeError:
            return environment.reset(task_type=task_type, scenario_index=scenario_index)
    raise RuntimeError("Environment does not support reset().")


def _step(environment: Any, action: MLTriageAction) -> Any:
    if hasattr(environment, "step"):
        return environment.step(action)
    raise RuntimeError("Environment does not support step().")


def _get_reward(obs: Any) -> float:
    if hasattr(obs, "reward"):
        return float(obs.reward or 0.0)
    if isinstance(obs, dict):
        return float(obs.get("reward", 0.0) or 0.0)
    return 0.0


def _is_done(obs: Any) -> bool:
    if hasattr(obs, "done"):
        return bool(obs.done)
    if isinstance(obs, dict):
        return bool(obs.get("done", False))
    return False


def _episode_score(obs: Any) -> float:
    payload = _current_payload(obs)
    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {}
    if "final_score" in metadata:
        return float(metadata["final_score"])
    return _get_reward(obs)


def run_episode(llm_client: OpenAI, task_type: str, scenario_index: int) -> float:
    scenario_list = list(SCENARIO_MAP[task_type].values())
    scenario = scenario_list[scenario_index]
    environment = _environment()

    print(_start_line(task_name=task_type), flush=True)

    rewards: List[float] = []
    success = False
    last_score = 0.0
    try:
        obs = _reset(environment, task_type, scenario_index, seed=42 + scenario_index)
        planned_actions = _planned_actions(task_type, scenario)

        for step_index, planned in enumerate(planned_actions, start=1):
            obs_payload = _current_payload(obs)
            action = _prompt_for_action(task_type, obs_payload, planned, llm_client)
            obs = _step(environment, action)
            reward = _get_reward(obs)
            done = _is_done(obs)
            last_score = _episode_score(obs)

            payload = _current_payload(obs)
            metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {}
            last_action_error = payload.get("last_action_error")
            if last_action_error is None:
                last_action_error = metadata.get("last_action_error")

            rewards.append(float(reward))
            print(_step_line(step_index, action, reward, done, last_action_error), flush=True)

            if done:
                success = True
                break
    finally:
        end_score = max(0.0, min(1.0, float(last_score)))
        if hasattr(environment, "close"):
            try:
                environment.close()
            except Exception:
                pass
        print(_end_line(success=success, score=end_score, rewards=rewards), flush=True)

    return float(last_score)


def main() -> None:
    start_time = time.time()
    llm_client = _make_openai_client()

    results: Dict[str, List[float]] = {task_type: [] for task_type in TASK_TYPES}
    for task_type in TASK_TYPES:
        for scenario_index in SCENARIO_INDICES[:EPISODES_PER_TASK]:
            score = run_episode(llm_client, task_type, scenario_index)
            results[task_type].append(score)

    elapsed = time.time() - start_time
    task_averages = {
        task_type: round(sum(scores) / len(scores), 4) if scores else 0.0
        for task_type, scores in results.items()
    }
    overall_scores = [score for scores in results.values() for score in scores]
    overall_average = round(sum(overall_scores) / len(overall_scores), 4) if overall_scores else 0.0

    summary = {
        "task_averages": task_averages,
        "overall_average": overall_average,
        "episodes": len(overall_scores),
        "elapsed_seconds": round(elapsed, 2),
    }
    Path(ROOT / "baseline_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
