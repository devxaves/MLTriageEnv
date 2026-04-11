# Copyright (c) 2026. MLTriageEnv - OpenEnv RL Environment for ML Pipeline Debugging.
# All rights reserved.

"""
Pydantic models for MLTriageEnv.

Defines typed Action, Observation, and State models that extend
the OpenEnv base classes for ML pipeline debugging tasks.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field, model_validator

from openenv.core.env_server.types import Action, Observation, State


# Valid action types for the environment
VALID_ACTION_TYPES = frozenset([
    "inspect",
    "diagnose",
    "patch",
    "fix_stage",
    "validate",
    "done",
    "inspect_logs",
    "query_metrics",
    "check_dependency_graph",
    "dismiss_red_herring",
    "finalize_triage",
])


class MLTriageAction(Action):
    """Action model for ML pipeline debugging.

    Agents interact with the environment by sending structured actions
    with a type, target, and value.

    Action Types:
        - inspect: View artifact details (target=what to inspect, value=filter/scope)
        - diagnose: Identify an issue (target=field/stage/failure_mode, value=diagnosis)
        - patch: Fix a config field (target=field_name, value=corrected_value)
        - fix_stage: Fix a pipeline stage (target=stage_name, value=fix_description)
        - validate: Verify fixes (target=artifact_type, value=check_type)
        - done: End the episode (target="task", value="complete")
    """

    model_config = {"extra": "forbid"}

    action_type: str = Field(
        ...,
        description="Type of action: inspect, diagnose, patch, fix_stage, validate, done",
    )
    target: str = Field(
        ...,
        description="The field, stage, or component being acted upon",
    )
    value: str = Field(
        default="",
        description="The proposed fix, diagnosis label, or inspection query",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured metadata for the action",
    )
    service: str = Field(
        default="",
        description="Optional target service for investigation-style actions",
    )
    query: str = Field(
        default="",
        description="Optional query/filter for inspection or metrics actions",
    )
    root_cause: str = Field(
        default="",
        description="Optional root-cause declaration used by finalize_triage",
    )
    priority: str = Field(
        default="",
        description="Optional priority declaration (e.g. P1/P2/P3)",
    )
    rationale: str = Field(
        default="",
        description="Optional free-text rationale for the action",
    )


class MLTriageReward(BaseModel):
    """Typed reward model for shaped episode feedback."""

    model_config = {"extra": "forbid"}

    value: float = Field(..., description="Scalar reward value for the step")
    shaped: float = Field(
        default=0.0,
        description="Dense shaping component used to guide the agent",
    )
    terminal: bool = Field(
        default=False,
        description="Whether the reward was assigned on a terminal step",
    )
    explanation: str = Field(
        default="",
        description="Human-readable explanation of the reward",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured metadata for downstream consumers",
    )


class MLTriageObservation(Observation):
    """Observation model returned after each step.

    Extends the base Observation (which provides done, reward, metadata)
    with domain-specific fields for ML pipeline debugging.
    """

    model_config = {"extra": "forbid"}

    task_id: str = Field(
        default="",
        description="Unique identifier for the current scenario",
    )
    task_type: str = Field(
        default="",
        description="Type of task: config, logs, or pipeline",
    )
    artifact: str = Field(
        default="",
        description="The ML artifact (config YAML / log text / pipeline descriptor)",
    )
    history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Previous actions and their outcomes",
    )
    feedback: str = Field(
        default="",
        description="Result of the last action taken",
    )
    issues_found: List[str] = Field(
        default_factory=list,
        description="Issues identified so far",
    )
    issues_remaining: int = Field(
        default=0,
        ge=0,
        description="Number of issues still unresolved",
    )
    step_count: int = Field(
        default=0,
        ge=0,
        description="Steps taken in this episode",
    )
    max_steps: int = Field(
        default=15,
        description="Maximum steps allowed per episode",
    )
    available_tools: List[str] = Field(
        default_factory=list,
        description="Tool-style actions available to the agent in the current scenario",
    )
    evidence: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured evidence gathered so far during investigation",
    )
    last_action_error: str = Field(
        default="",
        description="Raw error string for the last action, if any",
    )


class MLTriageState(State):
    """State model for episode tracking.

    Extends the base State (which provides episode_id, step_count)
    with task-specific metadata.
    """

    model_config = {"extra": "forbid"}

    task_id: str = Field(
        default="",
        description="Scenario identifier",
    )
    task_type: str = Field(
        default="",
        description="Task type: config, logs, or pipeline",
    )
    total_issues: int = Field(
        default=0,
        ge=0,
        description="Total number of issues in this scenario",
    )
    issues_resolved: int = Field(
        default=0,
        ge=0,
        description="Number of issues resolved so far",
    )
    current_score: float = Field(
        default=0.0001,
        description="Cumulative reward for this episode",
    )
    max_steps: int = Field(
        default=15,
        description="Maximum steps allowed",
    )
    investigated_services: List[str] = Field(
        default_factory=list,
        description="Services inspected so far",
    )
    queried_metrics: List[str] = Field(
        default_factory=list,
        description="Services/keys queried through metrics actions",
    )
    dismissed_red_herrings: List[str] = Field(
        default_factory=list,
        description="Red-herring services explicitly dismissed",
    )
    knowledge_base: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured investigation memory accumulated by the agent",
    )

    def __call__(self) -> "MLTriageState":
        """Allow `env.state()` style calls while keeping property-based access."""
        return self

    @model_validator(mode="after")
    def clamp_score_to_strict_range(self) -> "MLTriageState":
        """Ensure current_score is strictly within (0.0, 1.0)."""
        if self.current_score <= 0.0:
            self.current_score = 0.0001
        elif self.current_score >= 1.0:
            self.current_score = 0.9999
        return self
