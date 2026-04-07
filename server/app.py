"""
FastAPI application for MLTriageEnv.

Exposes the MLTriageEnvironment over HTTP and WebSocket endpoints
using OpenEnv's create_app factory.
"""

import sys
import os

# Ensure the project root is on sys.path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from openenv.core.env_server.http_server import create_app

from server.environment import MLTriageEnvironment
from models import MLTriageAction, MLTriageObservation

# Create the FastAPI app using OpenEnv's factory
# The factory takes the class (not an instance), uses it
# as a factory to create per-session environments.
app = create_app(
    MLTriageEnvironment,
    MLTriageAction,
    MLTriageObservation,
    env_name="ml_triage_env",
)


@app.get("/")
def root() -> dict:
    """Friendly root endpoint for Spaces/browser access."""
    return {
        "name": "MLTriageEnv",
        "status": "running",
        "message": "Use /health, /reset, /step, /state (or /docs if enabled).",
    }


def main():
    """Entry point for direct execution.

    Usage:
        python -m server.app
        uv run --project . server
    """
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
