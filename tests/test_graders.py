"""Tests for graders — deterministic scoring logic."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pytest
from pathlib import Path
from graders.grader1 import grade_config_episode
from graders.grader2 import grade_log_episode
from graders.grader3 import grade_pipeline_episode

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"


class TestGrader1Config:
    """Test config fixer grader."""

    def _load_scenario(self, idx: int = 0):
        with open(SCENARIOS_DIR / "config_scenarios.json") as f:
            return json.load(f)[idx]

    def test_perfect_score(self):
        s = self._load_scenario(0)
        # Simulate all issues found and patched
        total = sum(
            1 for v in s["broken_fields"].values()
            if v.get("issue_type", "none") != "none"
        )
        issues = [
            {"field": k, "type": "patch", "correct": True}
            for k, v in s["broken_fields"].items()
            if v.get("issue_type", "none") != "none"
        ]
        issues += [
            {"field": k, "type": "diagnosis", "issue_type": v["issue_type"]}
            for k, v in s["broken_fields"].items()
            if v.get("issue_type", "none") != "none"
        ]
        score = grade_config_episode(s, issues, step_count=5, max_steps=15)
        assert score > 0.8
        assert score <= 1.0

    def test_no_action_score_is_zero(self):
        s = self._load_scenario(0)
        score = grade_config_episode(s, [], step_count=15, max_steps=15)
        assert 0.0 < score < 1.0

    def test_partial_fix_gives_partial_score(self):
        s = self._load_scenario(0)
        # Fix one of the issues
        broken = [
            k for k, v in s["broken_fields"].items()
            if v.get("issue_type", "none") != "none"
        ]
        issues = [{"field": broken[0], "type": "patch", "correct": True}]
        score = grade_config_episode(s, issues, step_count=10, max_steps=15)
        assert 0.0 < score < 1.0

    def test_score_in_range(self):
        s = self._load_scenario(0)
        for issues in [
            [],
            [{"field": "x", "type": "patch", "correct": True}],
            [{"field": "x", "type": "patch", "correct": False}],
        ]:
            score = grade_config_episode(s, issues, step_count=10)
            assert 0.0 < score < 1.0


class TestGrader2Logs:
    """Test log diagnostician grader."""

    def _load_scenario(self, idx: int = 0):
        with open(SCENARIOS_DIR / "log_scenarios.json") as f:
            return json.load(f)[idx]

    def test_perfect_score(self):
        s = self._load_scenario(0)
        issues = [
            {"type": "diagnosis", "mode": s["failure_mode"], "correct": True},
            {"type": "intervention", "correct": True},
        ]
        score = grade_log_episode(s, issues, step_count=3, max_steps=15)
        assert score > 0.8

    def test_diagnosis_only(self):
        s = self._load_scenario(0)
        issues = [
            {"type": "diagnosis", "mode": s["failure_mode"], "correct": True},
        ]
        score = grade_log_episode(s, issues, step_count=5, max_steps=15)
        assert 0.3 < score < 0.6

    def test_no_action_is_zero(self):
        s = self._load_scenario(0)
        score = grade_log_episode(s, [], step_count=15, max_steps=15)
        assert 0.0 < score < 1.0


class TestGrader3Pipeline:
    """Test pipeline debugger grader."""

    def _load_scenario(self, idx: int = 0):
        with open(SCENARIOS_DIR / "pipeline_scenarios.json") as f:
            return json.load(f)[idx]

    def test_perfect_score(self):
        s = self._load_scenario(0)
        issues = []
        for stage in s["bugs"]:
            issues.append({
                "stage": stage,
                "type": "diagnosis",
                "bug_type": s["bugs"][stage]["bug_type"],
                "correct": True,
            })
            issues.append({
                "stage": stage,
                "type": "fix",
                "correct": True,
            })
        score = grade_pipeline_episode(s, issues, step_count=5, max_steps=20)
        assert score > 0.8

    def test_no_action_is_zero(self):
        s = self._load_scenario(0)
        score = grade_pipeline_episode(s, [], step_count=20, max_steps=20)
        assert 0.0 < score < 1.0

    def test_partial_diagnosis_partial_score(self):
        s = self._load_scenario(0)
        stages = list(s["bugs"].keys())
        issues = [
            {"stage": stages[0], "type": "diagnosis", "correct": True, "bug_type": "x"},
        ]
        score = grade_pipeline_episode(s, issues, step_count=10, max_steps=20)
        assert 0.0 < score < 0.5


class TestGraderDeterminism:
    """Graders must be deterministic — same inputs → same output."""

    def test_config_deterministic(self):
        with open(SCENARIOS_DIR / "config_scenarios.json") as f:
            s = json.load(f)[0]
        issues = [{"field": "x", "type": "patch", "correct": True}]
        s1 = grade_config_episode(s, issues, 5)
        s2 = grade_config_episode(s, issues, 5)
        assert s1 == s2

    def test_log_deterministic(self):
        with open(SCENARIOS_DIR / "log_scenarios.json") as f:
            s = json.load(f)[0]
        issues = [{"type": "diagnosis", "correct": True}]
        s1 = grade_log_episode(s, issues, 5)
        s2 = grade_log_episode(s, issues, 5)
        assert s1 == s2

    def test_pipeline_deterministic(self):
        with open(SCENARIOS_DIR / "pipeline_scenarios.json") as f:
            s = json.load(f)[0]
        issues = [{"stage": "x", "type": "fix", "correct": True}]
        s1 = grade_pipeline_episode(s, issues, 5)
        s2 = grade_pipeline_episode(s, issues, 5)
        assert s1 == s2
