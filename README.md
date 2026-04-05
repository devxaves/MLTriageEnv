# MLTriageEnv 🔧

**An OpenEnv RL Environment for ML Pipeline Debugging**

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-v1-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MLTriageEnv trains AI agents to diagnose and fix real-world ML pipeline failures. The environment simulates three progressively difficult debugging scenarios that ML engineers face every day. It exposes the OpenEnv-style `step()` / `reset()` / `state()` workflow and is designed for reproducible evaluation rather than toy gameplay.

## 🎯 Tasks

| Task                  | Difficulty | Description                                                                          | Max Steps |
| --------------------- | ---------- | ------------------------------------------------------------------------------------ | --------- |
| **Config Fixer**      | 🟢 Easy    | Find and fix broken YAML training config fields (invalid LR, typos, wrong types)     | 15        |
| **Log Diagnostician** | 🟡 Medium  | Read training logs, classify failure modes, and propose interventions                | 15        |
| **Pipeline Debugger** | 🔴 Hard    | Trace silent bugs across multi-stage ML pipelines (data leakage, wrong splits, etc.) | 20        |

Each task has **15 unique scenarios** (45 total), all with realistic ML content — no stubs or placeholders.

## 🧬 OpenEnv Spec Compliance

- ✅ Typed Pydantic models: `MLTriageAction`, `MLTriageObservation`, `MLTriageState`
- ✅ Full `step()` / `reset()` / `state()` API
- ✅ `openenv.yaml` manifest (spec version 1)
- ✅ Deterministic graders (0.0–1.0, partial credit)
- ✅ Dense reward signal with shaped intermediate rewards
- ✅ Baseline inference script with OpenAI SDK
- ✅ Dockerfile for containerized deployment
- ✅ WebSocket + HTTP endpoint support

## 📐 Action / Observation Spaces

### Action Space (`MLTriageAction`)

```python
{
    "action_type": str,   # inspect | diagnose | patch | fix_stage | validate | done
    "target": str,        # field name, stage name, or component
    "value": str          # proposed fix, diagnosis, or query
}
```

### Observation Space (`MLTriageObservation`)

```python
{
    "done": bool,              # Episode terminated?
    "reward": float,           # Step reward
    "task_id": str,            # Scenario identifier
    "task_type": str,          # config | logs | pipeline
    "artifact": str,           # The ML artifact to debug
    "history": list,           # Previous actions and outcomes
    "feedback": str,           # Result of last action
    "issues_found": list,      # Issues discovered so far
    "issues_remaining": int,   # Count of unresolved issues
    "step_count": int,         # Current step number
    "max_steps": int           # Budget for this episode
}
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -e ".[dev]"
# or
pip install openenv-core[core] fastapi uvicorn pydantic requests openai
```

### 2. Start the Server

```bash
# From the project root
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Interact via HTTP

```bash
# Reset
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_type": "config"}'

# Step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "inspect", "target": "learning_rate", "value": ""}}'

# Health
curl http://localhost:8000/health
```

### 4. Run Inference Baseline

```bash
export HF_TOKEN="your_hf_token"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export ENV_URL="http://localhost:8000"
python inference.py
```

## 🐳 Docker

```bash
# Build
docker build -t ml-triage-env .

# Run
docker run -p 8000:8000 ml-triage-env

# Verify
curl http://localhost:8000/health
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📊 Reward Design

Rewards are **dense and shaped** with partial credit:

- **Config Fixer**: 0.25 per correct patch, 0.15 per correct diagnosis, efficiency bonus
- **Log Diagnostician**: 0.35 for correct failure mode, 0.30 for correct intervention
- **Pipeline Debugger**: 0.20 per correct stage fix, 0.15 per correct diagnosis, completion bonus

Final episode score (0.0–1.0) is computed by the deterministic grader — no LLM involved in scoring.

## 📈 Baseline Scores

Baseline run: `python inference.py`

| Task              | Episodes | Average Score |
| ----------------- | -------- | ------------- |
| Config Fixer      | 3        | 0.94          |
| Log Diagnostician | 3        | 0.94          |
| Pipeline Debugger | 3        | 0.91          |
| Overall           | 9        | 0.93          |

The inference script emits structured stdout using `[START]`, `[STEP]`, and `[END]` records and writes the summary to `baseline_results.json`.

## 📁 Project Structure

```
├── models.py              # Pydantic Action, Observation, State models
├── client.py              # EnvClient subclass
├── inference.py           # Baseline LLM inference script
├── openenv.yaml           # OpenEnv environment manifest
├── pyproject.toml         # Dependencies and project metadata
├── server/
│   ├── app.py             # FastAPI application (create_app)
│   ├── environment.py     # MLTriageEnvironment(Environment)
│   ├── Dockerfile         # Container configuration
│   └── requirements.txt
├── tasks/
│   ├── base.py            # BaseTask ABC
│   ├── task1_config.py    # Config Fixer (Easy)
│   ├── task2_logs.py      # Log Diagnostician (Medium)
│   └── task3_pipeline.py  # Pipeline Debugger (Hard)
├── graders/
│   ├── grader1.py         # Config scoring
│   ├── grader2.py         # Log scoring
│   └── grader3.py         # Pipeline scoring
├── scenarios/
│   ├── config_scenarios.json    # 15 config scenarios
│   ├── log_scenarios.json       # 15 log scenarios
│   └── pipeline_scenarios.json  # 15 pipeline scenarios
└── tests/
    ├── test_environment.py
    ├── test_graders.py
    └── test_tasks.py
```

## 🤗 Deployment to HF Spaces

```bash
# Using OpenEnv CLI
openenv push your-username/ml-triage-env

# Or manually via git
# 1. Create a new HF Space (Docker type)
# 2. Push this repo to it
```

## 📝 Environment Variables

| Variable       | Default                                   | Description            |
| -------------- | ----------------------------------------- | ---------------------- |
| `API_BASE_URL` | `https://api-inference.huggingface.co/v1` | LLM API endpoint       |
| `MODEL_NAME`   | `meta-llama/Llama-3.1-8B-Instruct`        | Model for inference    |
| `HF_TOKEN`     | `""`                                      | Hugging Face API token |
| `ENV_URL`      | `http://localhost:8000`                   | MLTriageEnv server URL |

## 🧭 Validation Checklist

- `openenv.yaml` defines metadata, tasks, and hardware requirements
- `models.py` defines typed Action, Observation, State, and Reward models
- `server/environment.py` implements `step()`, `reset()`, and `state()`
- `inference.py` uses the OpenAI client and produces reproducible baseline scores
- Root-level `Dockerfile` supports `docker build -t ml-triage-env .`

## License

MIT
