"""Gradio dashboard for interactive ML triage episodes."""

from __future__ import annotations

import json
from typing import Any, Dict

import gradio as gr

from models import MLTriageAction
from server.environment import MLTriageEnvironment


VALID_ACTIONS = ["inspect", "diagnose", "patch", "fix_stage", "validate", "done"]


def _obs_to_dict(obs: Any) -> Dict[str, Any]:
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if isinstance(obs, dict):
        return obs
    return {}


def _new_session(task_type: str) -> tuple[str, str, MLTriageEnvironment, Dict[str, Any]]:
    env = MLTriageEnvironment()
    obs = env.reset(task_type=task_type)
    data = _obs_to_dict(obs)
    artifact = data.get("artifact", "")
    feedback = data.get("feedback", "")
    return artifact, feedback, env, data


def _submit_action(
    action_type: str,
    target: str,
    value: str,
    env: MLTriageEnvironment | None,
    obs_state: Dict[str, Any] | None,
    task_type: str,
) -> tuple[str, str, MLTriageEnvironment, Dict[str, Any]]:
    try:
        if env is None:
            artifact, feedback, env, obs_state = _new_session(task_type)
        action = MLTriageAction(
            action_type=action_type,
            target=target or "",
            value=value or "",
            metadata={},
        )
        obs = env.step(action)
        data = _obs_to_dict(obs)
        artifact = data.get("artifact", "")
        metadata = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
        console_output = metadata.get("console_output", "")
        feedback = data.get("feedback", "")
        terminal = "\nEpisode finished." if data.get("done", False) else ""
        console = f"{console_output}\n{feedback}{terminal}".strip()
        return artifact, console, env, data
    except Exception as exc:
        return (
            (obs_state or {}).get("artifact", ""),
            f"Execution Failed: {exc}",
            env,
            obs_state or {},
        )


with gr.Blocks(title="ML Triage Ops Dashboard") as demo:
    gr.Markdown("# 🚀 ML Triage Ops Dashboard")

    with gr.Row():
        with gr.Column(scale=3):
            environment_view = gr.Textbox(
                label="Broken Environment State / Logs",
                lines=20,
                interactive=False,
            )
        with gr.Column(scale=2):
            task_type = gr.Dropdown(
                choices=["config", "logs", "pipeline"],
                value="config",
                label="Task Type",
            )
            action_type = gr.Dropdown(
                choices=VALID_ACTIONS,
                value="inspect",
                label="Action Type",
            )
            target = gr.Textbox(label="Target", placeholder="field/stage/component")
            value = gr.Textbox(label="Value", placeholder="diagnosis/fix/details")
            with gr.Row():
                reset_button = gr.Button("Reset Episode", variant="secondary")
                submit_button = gr.Button("Submit Action", variant="primary")

    terminal_console = gr.Textbox(label="Terminal Console", lines=8, interactive=False)

    env_state = gr.State(value=None)
    obs_state = gr.State(value={})

    reset_button.click(
        fn=_new_session,
        inputs=[task_type],
        outputs=[environment_view, terminal_console, env_state, obs_state],
    )

    submit_button.click(
        fn=_submit_action,
        inputs=[action_type, target, value, env_state, obs_state, task_type],
        outputs=[environment_view, terminal_console, env_state, obs_state],
    )
