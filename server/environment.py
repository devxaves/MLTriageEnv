"""
MLTriageEnvironment — Core OpenEnv Environment.

Implements the Environment[MLTriageAction, MLTriageObservation, MLTriageState] interface.
Handles episode lifecycle: reset → step → done.
"""

import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from models import (
    MLTriageAction,
    MLTriageObservation,
    MLTriageState,
    VALID_ACTION_TYPES,
)
from tasks.task1_config import ConfigFixerTask
from tasks.task2_logs import LogDiagnosticianTask
from tasks.task3_pipeline import PipelineDebuggerTask
from graders.grader1 import grade_config_episode
from graders.grader2 import grade_log_episode
from graders.grader3 import grade_pipeline_episode


# Max steps per task type
MAX_STEPS = {"config": 15, "logs": 15, "pipeline": 20}

# Task type mapping
TASK_TYPES = ["config", "logs", "pipeline"]


class MLTriageEnvironment(Environment):
    """OpenEnv Environment for ML pipeline debugging.

    The agent interacts with broken ML artifacts (configs, logs, pipelines)
    and must diagnose and fix issues through structured actions.

    Supports concurrent sessions.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        # Initialize tasks
        self._tasks = {
            "config": ConfigFixerTask(),
            "logs": LogDiagnosticianTask(),
            "pipeline": PipelineDebuggerTask(),
        }

        # Episode state
        self._state = MLTriageState(episode_id=str(uuid4()), step_count=0)
        self._scenario: Dict[str, Any] = {}
        self._task_type: str = ""
        self._history: List[Dict[str, str]] = []
        self._issues_found: List[Dict[str, Any]] = []
        self._episode_done: bool = False
        self._cumulative_reward: float = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MLTriageObservation:
        """Reset the environment for a new episode.

        Kwargs:
            task_type: Optional[str] — force a specific task type (config/logs/pipeline)
            scenario_index: Optional[int] — force a specific scenario index
        """
        if seed is not None:
            random.seed(seed)

        # Choose task type
        task_type = kwargs.get("task_type")
        if task_type and task_type in TASK_TYPES:
            self._task_type = task_type
        else:
            self._task_type = random.choice(TASK_TYPES)

        # Sample scenario
        task = self._tasks[self._task_type]
        scenario_index = kwargs.get("scenario_index")
        if scenario_index is not None and 0 <= scenario_index < len(task.scenarios):
            self._scenario = task.scenarios[scenario_index]
        else:
            self._scenario = task.sample_scenario()

        # Reset episode state
        ep_id = episode_id or str(uuid4())
        max_steps = MAX_STEPS.get(self._task_type, 15)
        total_issues = self._scenario.get("total_issues", 0)

        self._history = []
        self._issues_found = []
        self._episode_done = False
        self._cumulative_reward = 0.0

        self._state = MLTriageState(
            episode_id=ep_id,
            step_count=0,
            task_id=self._scenario.get("id", ""),
            task_type=self._task_type,
            total_issues=total_issues,
            issues_resolved=0,
            current_score=0.0,
            max_steps=max_steps,
        )

        return MLTriageObservation(
            done=False,
            reward=0.0,
            metadata={
                "message": f"New {self._task_type} task loaded.",
                "scenario_id": self._scenario.get("id", ""),
                "difficulty": self._scenario.get("difficulty", "unknown"),
            },
            task_id=self._scenario.get("id", ""),
            task_type=self._task_type,
            artifact=self._scenario.get("artifact", ""),
            history=[],
            feedback=f"ML Triage: {self._scenario.get('description', '')}. "
                     f"You have {max_steps} steps to diagnose and fix all issues.",
            issues_found=[],
            issues_remaining=total_issues,
            step_count=0,
            max_steps=max_steps,
        )

    def step(
        self,
        action: MLTriageAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MLTriageObservation:
        """Execute one step in the environment."""
        # Auto-reset if step is called before reset
        # (HTTP server creates a new instance per request)
        if not self._task_type:
            self.reset()

        if self._episode_done:
            return self._terminal_observation("Episode already ended. Call reset().")

        # Validate action type
        if action.action_type not in VALID_ACTION_TYPES:
            return self._step_observation(
                reward=-0.05,
                feedback=f"Invalid action_type '{action.action_type}'. "
                         f"Valid: {', '.join(sorted(VALID_ACTION_TYPES))}.",
            )

        # Increment step
        self._state.step_count += 1
        max_steps = MAX_STEPS.get(self._task_type, 15)

        # Add to history
        self._history.append({
            "step": str(self._state.step_count),
            "action_type": action.action_type,
            "target": action.target,
            "value": action.value,
        })

        # Process action through task handler
        task = self._tasks[self._task_type]
        result = task.process_action(
            action_type=action.action_type,
            target=action.target,
            value=action.value,
            scenario=self._scenario,
            history=self._history,
            issues_found=self._issues_found,
        )

        reward = result.get("reward", 0.0)
        feedback = result.get("feedback", "")
        issue = result.get("issue_found")
        task_complete = result.get("task_complete", False)

        # Track found issues
        if issue is not None:
            self._issues_found.append(issue)

        # Update cumulative score
        self._cumulative_reward += reward
        self._state.current_score = self._cumulative_reward

        # Count resolved issues
        if self._task_type == "config":
            resolved = sum(
                1 for i in self._issues_found
                if i.get("type") == "patch" and i.get("correct")
            )
        elif self._task_type == "logs":
            resolved = sum(
                1 for i in self._issues_found
                if i.get("correct") is True
            )
        else:  # pipeline
            resolved = sum(
                1 for i in self._issues_found
                if i.get("type") == "fix" and i.get("correct")
            )
        self._state.issues_resolved = resolved

        # Check terminal conditions
        done = task_complete or self._state.step_count >= max_steps

        if done:
            self._episode_done = True
            # Compute final graded score
            final_score = self._compute_final_score()
            self._state.current_score = final_score

            if self._state.step_count >= max_steps and not task_complete:
                feedback += f" | Max steps ({max_steps}) reached."

            return self._terminal_observation(
                feedback=feedback,
                reward=final_score,
            )

        return self._step_observation(reward=reward, feedback=feedback)

    def _compute_final_score(self) -> float:
        """Compute the final graded score using the appropriate grader."""
        if self._task_type == "config":
            return grade_config_episode(
                scenario=self._scenario,
                issues_found=self._issues_found,
                step_count=self._state.step_count,
                max_steps=MAX_STEPS.get("config", 15),
            )
        elif self._task_type == "logs":
            return grade_log_episode(
                scenario=self._scenario,
                issues_found=self._issues_found,
                step_count=self._state.step_count,
                max_steps=MAX_STEPS.get("logs", 15),
            )
        elif self._task_type == "pipeline":
            return grade_pipeline_episode(
                scenario=self._scenario,
                issues_found=self._issues_found,
                step_count=self._state.step_count,
                max_steps=MAX_STEPS.get("pipeline", 20),
            )
        return 0.0

    def _step_observation(self, reward: float, feedback: str) -> MLTriageObservation:
        """Build a mid-episode observation."""
        total = self._scenario.get("total_issues", 0)
        return MLTriageObservation(
            done=False,
            reward=reward,
            metadata={
                "step": self._state.step_count,
                "cumulative_reward": self._cumulative_reward,
            },
            task_id=self._scenario.get("id", ""),
            task_type=self._task_type,
            artifact=self._scenario.get("artifact", ""),
            history=self._history.copy(),
            feedback=feedback,
            issues_found=[str(i) for i in self._issues_found],
            issues_remaining=max(0, total - self._state.issues_resolved),
            step_count=self._state.step_count,
            max_steps=MAX_STEPS.get(self._task_type, 15),
        )

    def _terminal_observation(self, feedback: str, reward: float = 0.0) -> MLTriageObservation:
        """Build a terminal observation."""
        return MLTriageObservation(
            done=True,
            reward=reward,
            metadata={
                "final_score": self._state.current_score,
                "steps_taken": self._state.step_count,
                "task_type": self._task_type,
                "scenario_id": self._scenario.get("id", ""),
            },
            task_id=self._scenario.get("id", ""),
            task_type=self._task_type,
            artifact=self._scenario.get("artifact", ""),
            history=self._history.copy(),
            feedback=feedback,
            issues_found=[str(i) for i in self._issues_found],
            issues_remaining=max(
                0,
                self._scenario.get("total_issues", 0) - self._state.issues_resolved,
            ),
            step_count=self._state.step_count,
            max_steps=MAX_STEPS.get(self._task_type, 15),
        )

    @property
    def state(self) -> MLTriageState:
        """Get the current environment state."""
        return self._state

    @property
    def current_state(self) -> MLTriageState:
        """Backward-compatible alias for callers that expect an attribute."""
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        """Return environment metadata for the web UI."""
        return EnvironmentMetadata(
            name="MLTriageEnv",
            description=(
                "An OpenEnv RL environment for training AI agents to diagnose and fix "
                "ML pipeline failures. Tasks: config fixing (easy), log diagnosis (medium), "
                "pipeline debugging (hard)."
            ),
            version="1.0.0",
            author="MLTriageEnv Team",
        )

    def close(self) -> None:
        """Clean up resources."""
        pass
