"""Tests for MLTriageEnvironment — core API contract."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from models import MLTriageAction, MLTriageObservation, MLTriageState
from server.app import app
from server.environment import MLTriageEnvironment


class TestEnvironmentReset:
    """Test reset() behavior."""

    def test_reset_returns_observation(self):
        env = MLTriageEnvironment()
        obs = env.reset()
        assert isinstance(obs, MLTriageObservation)
        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.artifact != ""
        assert obs.task_type in ("config", "logs", "pipeline")

    def test_reset_with_seed_is_deterministic(self):
        env1 = MLTriageEnvironment()
        obs1 = env1.reset(seed=42)
        env2 = MLTriageEnvironment()
        obs2 = env2.reset(seed=42)
        assert obs1.task_id == obs2.task_id
        assert obs1.task_type == obs2.task_type

    def test_reset_with_task_type(self):
        env = MLTriageEnvironment()
        for tt in ("config", "logs", "pipeline"):
            obs = env.reset(task_type=tt)
            assert obs.task_type == tt

    def test_reset_with_episode_id(self):
        env = MLTriageEnvironment()
        obs = env.reset(episode_id="test-ep-123")
        assert env.state().episode_id == "test-ep-123"


class TestEnvironmentStep:
    """Test step() behavior."""

    def test_step_returns_observation(self):
        env = MLTriageEnvironment()
        env.reset(task_type="config", seed=1)
        action = MLTriageAction(action_type="inspect", target="learning_rate", value="")
        obs = env.step(action)
        assert isinstance(obs, MLTriageObservation)
        assert obs.done is False
        assert obs.step_count == 1

    def test_step_invalid_action_type(self):
        env = MLTriageEnvironment()
        env.reset(task_type="config", seed=1)
        action = MLTriageAction(action_type="fly", target="moon", value="")
        obs = env.step(action)
        assert "Invalid action_type" in obs.feedback

    def test_step_increments_count(self):
        env = MLTriageEnvironment()
        env.reset(task_type="config", seed=1)
        for i in range(3):
            action = MLTriageAction(action_type="inspect", target="learning_rate", value="")
            obs = env.step(action)
        assert obs.step_count == 3
        assert env.state().step_count == 3

    def test_done_action_ends_episode(self):
        env = MLTriageEnvironment()
        env.reset(task_type="config", seed=1)
        action = MLTriageAction(action_type="done", target="task", value="complete")
        obs = env.step(action)
        assert obs.done is True

    def test_max_steps_ends_episode(self):
        env = MLTriageEnvironment()
        env.reset(task_type="config", seed=1)
        for i in range(20):
            action = MLTriageAction(action_type="inspect", target="something", value="")
            obs = env.step(action)
            if obs.done:
                break
        assert obs.done is True

    def test_step_after_done_returns_terminal(self):
        env = MLTriageEnvironment()
        env.reset(task_type="config", seed=1)
        action = MLTriageAction(action_type="done", target="task", value="")
        env.step(action)
        obs = env.step(
            MLTriageAction(action_type="inspect", target="a", value="")
        )
        assert obs.done is True


class TestEnvironmentState:
    """Test state property."""

    def test_state_has_required_fields(self):
        env = MLTriageEnvironment()
        env.reset(task_type="config", seed=1)
        s = env.state()
        assert isinstance(s, MLTriageState)
        assert s.episode_id is not None
        assert s.step_count == 0
        assert s.task_type == "config"
        assert s.total_issues >= 0
        assert s.max_steps > 0

    def test_state_updates_after_step(self):
        env = MLTriageEnvironment()
        env.reset(task_type="config", seed=1)
        action = MLTriageAction(action_type="inspect", target="x", value="")
        env.step(action)
        assert env.state().step_count == 1


class TestModels:
    """Test Pydantic model validation."""

    def test_action_validation(self):
        a = MLTriageAction(action_type="inspect", target="lr", value="check")
        assert a.action_type == "inspect"

    def test_action_rejects_extra_fields(self):
        with pytest.raises(Exception):
            MLTriageAction(action_type="inspect", target="lr", extra="bad")

    def test_observation_defaults(self):
        obs = MLTriageObservation()
        assert obs.done is False
        assert obs.reward is None
        assert obs.step_count == 0

    def test_state_defaults(self):
        s = MLTriageState()
        assert s.step_count == 0
        assert s.current_score == 0.0


class TestAPIHealth:
    """Test API readiness endpoint."""

    def test_health_endpoint_returns_200(self):
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
