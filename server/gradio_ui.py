"""Gradio dashboard for interactive ML triage episodes."""

from __future__ import annotations

from typing import Any, Dict

import gradio as gr

from models import MLTriageAction
from server.environment import MLTriageEnvironment


TASK_OPTIONS = ["config", "logs", "pipeline"]

TASK_ACTIONS = {
    "config": ["inspect", "diagnose", "patch", "validate", "done"],
    "logs": ["inspect", "diagnose", "fix_stage", "validate", "done"],
    "pipeline": [
        "inspect",
        "diagnose",
        "fix_stage",
        "inspect_logs",
        "query_metrics",
        "check_dependency_graph",
        "dismiss_red_herring",
        "finalize_triage",
        "validate",
        "done",
    ],
}


def _obs_to_dict(obs: Any) -> Dict[str, Any]:
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if isinstance(obs, dict):
        return obs
    return {}


def _status_panel(data: Dict[str, Any]) -> str:
    if not data:
        return "No active episode. Click **Reset Episode**."

    done = bool(data.get("done", False))
    task = data.get("task_type", "")
    task_id = data.get("task_id", "")
    step = int(data.get("step_count", 0) or 0)
    max_steps = int(data.get("max_steps", 0) or 0)
    reward = float(data.get("reward", 0.0) or 0.0)
    issues_remaining = int(data.get("issues_remaining", 0) or 0)

    return (
        f"Task: **{task}** (`{task_id}`)  \n"
        f"Step: **{step}/{max_steps}**  \n"
        f"Latest reward: **{reward:.4f}**  \n"
        f"Issues remaining: **{issues_remaining}**  \n"
        f"Episode done: **{str(done).lower()}**"
    )


def _new_session(task_type: str) -> tuple[str, str, str, MLTriageEnvironment, Dict[str, Any]]:
    env = MLTriageEnvironment()
    obs = env.reset(task_type=task_type)
    data = _obs_to_dict(obs)
    artifact = data.get("artifact", "") or "No artifact returned."
    feedback = data.get("feedback", "") or "Episode reset."
    status = _status_panel(data)
    return artifact, feedback, status, env, data


def _actions_for_task(task_type: str):
    choices = TASK_ACTIONS.get(task_type, TASK_ACTIONS["config"])
    return gr.Dropdown(choices=choices, value=choices[0])


def _submit_action(
    action_type: str,
    target: str,
    value: str,
    env: MLTriageEnvironment | None,
    obs_state: Dict[str, Any] | None,
    task_type: str,
) -> tuple[str, str, str, MLTriageEnvironment, Dict[str, Any]]:
    try:
        if env is None:
            artifact, feedback, status, env, obs_state = _new_session(task_type)
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
        status = _status_panel(data)
        return artifact, console, status, env, data
    except Exception as exc:
        return (
            (obs_state or {}).get("artifact", ""),
            f"Execution Failed: {exc}",
            _status_panel(obs_state or {}),
            env,
            obs_state or {},
        )


with gr.Blocks(title="ML Triage Ops Dashboard") as demo:
    gr.Markdown(
        "# 🚀 ML Triage Ops Dashboard\n"
        "Professional incident-style workspace for the OpenEnv ML triage benchmark."
    )

    with gr.Row():
        with gr.Column(scale=3):
            environment_view = gr.Textbox(
                label="Broken Environment State / Logs",
                lines=20,
                interactive=False,
            )
            episode_status = gr.Markdown(value="No active episode.")
        with gr.Column(scale=2):
            task_type = gr.Dropdown(
                choices=TASK_OPTIONS,
                value="config",
                label="Task Type",
            )
            action_type = gr.Dropdown(
                choices=TASK_ACTIONS["config"],
                value=TASK_ACTIONS["config"][0],
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
        outputs=[environment_view, terminal_console, episode_status, env_state, obs_state],
    )

    task_type.change(
        fn=_actions_for_task,
        inputs=[task_type],
        outputs=[action_type],
    )

    task_type.change(
        fn=_new_session,
        inputs=[task_type],
        outputs=[environment_view, terminal_console, episode_status, env_state, obs_state],
    )

    submit_button.click(
        fn=_submit_action,
        inputs=[action_type, target, value, env_state, obs_state, task_type],
        outputs=[environment_view, terminal_console, episode_status, env_state, obs_state],
    )

    demo.load(
        fn=_new_session,
        inputs=[task_type],
        outputs=[environment_view, terminal_console, episode_status, env_state, obs_state],
    )
