# Project Guidelines

## Project Purpose

MLTriageEnv is an OpenEnv-based reinforcement learning environment for ML debugging tasks.
The codebase models a full loop of agent action -> environment transition -> deterministic grading across three task types:

- config (easy)
- logs (medium)
- pipeline (hard)

## Architecture

- `models.py`: Strict Pydantic contracts for action, observation, and state. Extra fields are forbidden.
- `server/environment.py`: Core episode lifecycle (reset/step/done), task dispatch, terminal scoring.
- `tasks/`: Task-specific action semantics and reward shaping.
- `graders/`: Pure deterministic episode scorers with normalized output in [0.0, 1.0].
- `server/app.py`: FastAPI/OpenEnv app factory wiring.
- `client.py`: Client wrapper (HTTP + WebSocket mode) for interacting with the environment.
- `inference.py`: Baseline autonomous runner and evaluation loop.

Prefer keeping logic in task and grader modules instead of embedding task-specific behavior in the server layer.

## Build And Test

Use one of the established flows.

- Install project:
  - `pip install -e .`
  - or `uv sync`
- Run server locally:
  - `python -m server.app`
  - or `uvicorn server.app:app --host 0.0.0.0 --port 8000`
- Run tests:
  - `pytest tests/`
  - `pytest tests/ -v --cov=.`

When modifying grading, tasks, or environment transitions, run at least:

- `pytest tests/test_graders.py`
- `pytest tests/test_tasks.py`
- `pytest tests/test_environment.py`

## Conventions

- Keep Pydantic models strict (`model_config = {"extra": "forbid"}`) unless there is a strong compatibility reason.
- Keep action vocabulary aligned with `VALID_ACTION_TYPES` and OpenEnv contracts.
- Task handlers should follow `BaseTask.process_action(...)` and return:
  - `reward` (float)
  - `feedback` (string)
  - `issue_found` (optional dict)
  - `task_complete` (bool)
- Graders must remain deterministic and clip final scores to [0.0, 1.0].
- Prefer explicit, structured feedback strings in observations so inference agents can react reliably.

## Project-Specific Pitfalls

- The environment can randomize task type on reset if `task_type` is not provided. Set `task_type` in tests that expect specific behavior.
- Local and deployment ports differ in docs and Docker contexts (8000 vs 7860). Keep host/port choices explicit.
- Inference logging format is strict for external evaluators. If changing `inference.py`, preserve required tag formats exactly.
- This repository includes a roadmap file named `propmt.md` (intentional existing filename). Keep references consistent unless a dedicated rename is requested.

## Key References (Link, Do Not Duplicate)

- High-level overview and architecture: `README.md`
- Competition upgrade roadmap and strict output requirements: `propmt.md`
- OpenEnv metadata and deployment settings: `openenv.yaml`
- API contracts and schema details: `models.py`
- Task behavior details: `tasks/`
- Scoring formulas: `graders/`
- Runtime wiring: `server/app.py`, `server/environment.py`

## Change Discipline

- Make small, scoped edits and avoid broad refactors unless requested.
- Preserve public contracts used by tests and scenario JSON files.
- If adjusting score formulas or action semantics, update or add tests in `tests/` in the same change.
