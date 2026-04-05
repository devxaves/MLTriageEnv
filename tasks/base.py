"""
Base task interface for MLTriageEnv.

All tasks (config, logs, pipeline) implement this interface to provide
scenario sampling and action processing.
"""

import json
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List


SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"


class BaseTask(ABC):
    """Abstract base for all MLTriageEnv tasks."""

    scenario_file: str = ""  # Subclass must set this

    def __init__(self):
        self._scenarios: List[Dict[str, Any]] = []
        self._load_scenarios()

    def _load_scenarios(self) -> None:
        """Load scenarios from JSON file."""
        path = SCENARIOS_DIR / self.scenario_file
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self._scenarios = json.load(f)

    @property
    def scenarios(self) -> List[Dict[str, Any]]:
        return self._scenarios

    def sample_scenario(self) -> Dict[str, Any]:
        """Return a random scenario from the pool."""
        return random.choice(self._scenarios)

    @abstractmethod
    def process_action(
        self,
        action_type: str,
        target: str,
        value: str,
        scenario: Dict[str, Any],
        history: List[Dict[str, str]],
        issues_found: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Process an agent action and return result dict.

        Returns:
            Dict with keys: reward (float), feedback (str),
            issue_found (optional dict), task_complete (bool)
        """
        ...
