"""
HTTP client for MLTriageEnv.

This is a lightweight client that talks to the FastAPI/OpenEnv server
over HTTP and returns parsed observation payloads.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests
from websockets.sync.client import connect as ws_connect

from models import MLTriageAction, MLTriageObservation, MLTriageState


class MLTriageEnv:
    """Thin wrapper exposing a `.sync()` context manager style API."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self.base_url = base_url
        self.timeout = timeout

    def sync(self) -> MLTriageEnvClient:
        """Return a WebSocket-enabled client context manager."""
        return MLTriageEnvClient(
            base_url=self.base_url,
            timeout=self.timeout,
            use_websocket=True,
        )


class MLTriageEnvClient:
    """Small HTTP client for the environment server."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        use_websocket: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.use_websocket = use_websocket
        self._session = requests.Session()
        self._ws = None

    @property
    def ws_url(self) -> str:
        if self.base_url.startswith("https://"):
            return self.base_url.replace("https://", "wss://", 1) + "/ws"
        if self.base_url.startswith("http://"):
            return self.base_url.replace("http://", "ws://", 1) + "/ws"
        return self.base_url + "/ws"

    def _ensure_ws(self):
        if self._ws is None:
            self._ws = ws_connect(self.ws_url, open_timeout=self.timeout)
        return self._ws

    def _ws_request(self, msg_type: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ws = self._ensure_ws()
        payload: Dict[str, Any] = {"type": msg_type}
        if data is not None:
            payload["data"] = data
        ws.send(json.dumps(payload))
        raw = ws.recv()
        message = json.loads(raw)
        if message.get("type") == "error":
            details = message.get("data", {})
            raise RuntimeError(details.get("message", "WebSocket environment error"))
        return message.get("data", {})

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws_request("close")
            except Exception:
                pass
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._session.close()

    def __enter__(self) -> "MLTriageEnvClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def reset(self, **kwargs: Any) -> MLTriageObservation:
        if self.use_websocket:
            ws_payload = self._ws_request("reset", kwargs)
            return MLTriageObservation(**self._extract_payload(ws_payload))

        response = self._session.post(
            f"{self.base_url}/reset",
            json=kwargs,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return MLTriageObservation(**self._extract_payload(response.json()))

    def step(self, action: MLTriageAction) -> MLTriageObservation:
        if self.use_websocket:
            ws_payload = self._ws_request("step", action.model_dump())
            return MLTriageObservation(**self._extract_payload(ws_payload))

        response = self._session.post(
            f"{self.base_url}/step",
            json={"action": action.model_dump()},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return MLTriageObservation(**self._extract_payload(response.json()))

    def state(self) -> MLTriageState:
        if self.use_websocket:
            ws_payload = self._ws_request("state")
            return MLTriageState(**ws_payload)

        response = self._session.get(f"{self.base_url}/state", timeout=self.timeout)
        response.raise_for_status()
        return MLTriageState(**response.json())

    def health(self) -> Dict[str, Any]:
        response = self._session.get(f"{self.base_url}/health", timeout=self.timeout)
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
