You are an Expert Python Software Engineer and an AI Agent Architect. We are currently participating in the Meta PyTorch OpenEnv Hackathon. My current repository (`MLTriageEnv`) has a flawless base architecture but failed the automated Phase 1 evaluation due to strict regex mismatches and a hardcoded "cheat code" inference loop. We also need to add a Gradio UI to win Phase 3 (Human Evaluation).

Please completely execute the following 6 tasks to rebuild and upgrade this repository. Do not hallucinate external libraries; stick strictly to the OpenEnv, OpenAI, and Gradio specifications. 

First, briefly analyze my current workspace to understand the existing `MLTriageAction` and `MLTriageObservation` Pydantic schemas, then proceed with the following exact changes:

### TASK 1: Completely Rewrite `inference.py`
You must delete the current `inference.py` entirely, specifically removing `_planned_actions` and `SCENARIO_MAP`. Rewrite it to act as a genuine, autonomous LLM agent.

**Strict STDOUT Logging Rules (DO NOT DEVIATE):**
1. `[START] task=<task_name> env=ml_triage_env model=<model_name>`
2. `[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
3. `[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>`
*(CRITICAL: The `[END]` tag MUST NOT contain a `score=` field. Only success, steps, and rewards. Format rewards to 2 decimal places. Booleans must be lowercase 'true' or 'false'.)*

**Agent Logic Requirements:**
1. Initialize the OpenAI client using `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`.
2. Loop through `["config", "logs", "pipeline"]` tasks (3 episodes each).
3. **CRITICAL NETWORKING FIX:** You must use the OpenEnv client wrapper with `.sync()` to bypass HF Space HTTP blockers. Do not use raw `requests`.
   Example: `with MLTriageEnv(base_url="http://localhost:8000").sync() as env:`
4. Inside the episode loop (`while not done and step < MAX_STEPS:`):
   - Format a `user_prompt` that includes the current `observation` (JSON) and asks the LLM to output a valid `MLTriageAction` JSON.
   - Send it to the OpenAI client.
   - Use a `try/except` block to parse the LLM's JSON response. If parsing fails, create a fallback action: `{"action_type": "error", "target": "syntax", "value": "invalid json"}`. This prevents the script from crashing.
   - Call `result = env.step(action)` and print the exact `[STEP]` STDOUT log.
5. Print the exact `[END]` STDOUT log when the episode finishes.

### TASK 2: Fix the Math Bug in `graders/grader3.py`
There is a math bug in the Hard task scoring logic where a perfect score caps at 0.95. 
Locate the `grade_solution` or scoring function in `grader3.py` and change the weights to equal exactly 1.0:
- `diag_score = (correct_diag / total_bugs) * 0.30 + (partial_diag / total_bugs) * 0.05`
- `fix_score = (correct_fix / total_bugs) * 0.45 + (partial_fix / total_bugs) * 0.10`
*(Efficiency bonus remains 0.10, Completion bonus remains 0.15. Total = 1.0)*

### TASK 3: Add Rich Feedback in `server/environment.py`
To survive Phase 2 (Blind Agent Evaluation), the environment must talk back to the agent.
Locate the `step()` function in `server/environment.py`. Update the logic so that every time an action is processed, the returned `observation` dictionary includes a `feedback` or `console_output` string detailing exactly what happened (e.g., "Action applied successfully" or "Execution Failed: Syntax error on line 42").

### TASK 4: Create the "Wow Factor" UI (`server/gradio_ui.py`)
Create a brand new file named `server/gradio_ui.py` to build a visual dashboard for human evaluators.
- Use `import gradio as gr`.
- Create a Blocks app (`with gr.Blocks() as demo:`).
- Header: "🚀 ML Triage Ops Dashboard".
- Layout: 
  - Left Column: A `gr.Markdown` or `gr.Textbox` (interactive=False) to display the current broken environment state/logs.
  - Right Column: Input fields representing the `MLTriageAction` schema (Dropdown for Action Type, Textbox for Target, Textbox for Value), and a "Submit Action" button.
  - Bottom Row: A "Terminal Console" `gr.Textbox` to display the feedback/errors returned by the environment's `step()` function.
- Write a wrapper function that takes the Gradio inputs, instantiates the local environment, executes the step, and updates the UI elements.

### TASK 5: Mount the UI in `server/app.py`
Update `server/app.py` to serve the new Gradio dashboard.
- Import the Gradio app: `from .gradio_ui import demo as ui_app`
- Mount it to the FastAPI app: `import gradio as gr` -> `gr.mount_gradio_app(app, ui_app, path="/")`
This ensures that visiting the HF Space URL loads the dashboard instead of a blank API screen.

### TASK 6: Scale the `server/Dockerfile`
To survive the automated evaluation load testing, add the following two lines to your `Dockerfile`:
```dockerfile
ENV WORKERS=2
ENV MAX_CONCURRENT_ENVS=100