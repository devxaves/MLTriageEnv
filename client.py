"""
HTTP client for MLTriageEnv.

This is a lightweight client that talks to the FastAPI/OpenEnv server
over HTTP and returns parsed observation payloads.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests

from models import MLTriageAction, MLTriageObservation, MLTriageState


class MLTriageEnvClient:
    """Small HTTP client for the environment server."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def reset(self, **kwargs: Any) -> MLTriageObservation:
        response = requests.post(
            f"{self.base_url}/reset",
            json=kwargs,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return MLTriageObservation(**self._extract_payload(response.json()))

    def step(self, action: MLTriageAction) -> MLTriageObservation:
        response = requests.post(
            f"{self.base_url}/step",
            json={"action": action.model_dump()},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return MLTriageObservation(**self._extract_payload(response.json()))

    def state(self) -> MLTriageState:
        response = requests.get(f"{self.base_url}/state", timeout=self.timeout)
        response.raise_for_status()
        return MLTriageState(**response.json())

    def health(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _extract_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        if "observation" in payload and isinstance(payload["observation"], dict):
            merged = dict(payload["observation"])
            for key in ("reward", "done", "metadata", "info"):
                if key in payload and key not in merged:
                    merged[key] = payload[key]
            return merged
        return payload
