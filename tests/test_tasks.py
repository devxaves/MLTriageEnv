"""Tests for task handlers."""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path
from tasks.task1_config import ConfigFixerTask
from tasks.task2_logs import LogDiagnosticianTask
from tasks.task3_pipeline import PipelineDebuggerTask


SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"


class TestConfigFixerTask:
    def test_load_scenarios(self):
        task = ConfigFixerTask()
        assert len(task.scenarios) == 15

    def test_sample_scenario(self):
        task = ConfigFixerTask()
        s = task.sample_scenario()
        assert "id" in s
        assert "broken_fields" in s
        assert "artifact" in s

    def test_inspect_broken_field(self):
        task = ConfigFixerTask()
        s = task.scenarios[0]
        broken_fields = [
            k for k, v in s["broken_fields"].items()
            if v.get("issue_type", "none") != "none"
        ]
        result = task.process_action("inspect", broken_fields[0], "", s, [], [])
        assert result["reward"] >= 0.0
        assert result["feedback"] != ""

    def test_patch_correct_value(self):
        task = ConfigFixerTask()
        s = task.scenarios[0]
        for field, info in s["broken_fields"].items():
            if info.get("issue_type", "none") != "none":
                result = task.process_action(
                    "patch", field, info["correct_value"], s, [], []
                )
                assert result["reward"] > 0.0
                assert result["issue_found"] is not None
                assert result["issue_found"]["correct"] is True
                break

    def test_patch_wrong_value(self):
        task = ConfigFixerTask()
        s = task.scenarios[0]
        for field, info in s["broken_fields"].items():
            if info.get("issue_type", "none") != "none":
                result = task.process_action(
                    "patch", field, "totally_wrong", s, [], []
                )
                assert result["reward"] > 0.0  # partial credit
                assert result["issue_found"]["correct"] is False
                break


class TestLogDiagnosticianTask:
    def test_load_scenarios(self):
        task = LogDiagnosticianTask()
        assert len(task.scenarios) == 15

    def test_correct_diagnosis(self):
        task = LogDiagnosticianTask()
        s = task.scenarios[0]  # gradient_explosion
        result = task.process_action(
            "diagnose", "failure_mode", "gradient explosion", s, [], []
        )
        assert result["reward"] > 0.0
        assert result["issue_found"]["correct"] is True

    def test_wrong_diagnosis(self):
        task = LogDiagnosticianTask()
        s = task.scenarios[0]  # gradient_explosion
        result = task.process_action(
            "diagnose", "failure_mode", "underfitting", s, [], []
        )
        assert result["reward"] <= 0.0

    def test_correct_intervention(self):
        task = LogDiagnosticianTask()
        s = task.scenarios[0]  # gradient_explosion → clip
        result = task.process_action(
            "fix_stage", "training", "add gradient clipping", s, [], []
        )
        assert result["reward"] > 0.0


class TestPipelineDebuggerTask:
    def test_load_scenarios(self):
        task = PipelineDebuggerTask()
        assert len(task.scenarios) == 16

    def test_inspect_faulty_stage(self):
        task = PipelineDebuggerTask()
        s = task.scenarios[0]
        faulty = list(s["bugs"].keys())[0]
        result = task.process_action("inspect", faulty, "", s, [], [])
        assert result["reward"] >= 0.0
        assert result["feedback"] != ""

    def test_correct_diagnosis(self):
        task = PipelineDebuggerTask()
        s = task.scenarios[0]
        faulty = list(s["bugs"].keys())[0]
        bug_type = s["bugs"][faulty]["bug_type"]
        result = task.process_action(
            "diagnose", faulty, bug_type, s, [], []
        )
        assert result["reward"] > 0.0
        assert result["issue_found"]["correct"] is True

    def test_fix_correct_stage(self):
        task = PipelineDebuggerTask()
        s = task.scenarios[0]
        faulty = list(s["bugs"].keys())[0]
        fix_desc = s["bugs"][faulty]["fix"]
        result = task.process_action(
            "fix_stage", faulty, fix_desc, s, [], []
        )
        assert result["reward"] > 0.0

    def test_diagnose_valid_stage_is_penalty(self):
        task = PipelineDebuggerTask()
        s = task.scenarios[0]
        valid = s["valid_stages"][0]
        result = task.process_action("diagnose", valid, "some bug", s, [], [])
        assert result["reward"] < 0.0
