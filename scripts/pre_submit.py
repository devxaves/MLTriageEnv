#!/usr/bin/env python3
"""Pre-submission validator for MLTriageEnv.

Checks:
1) Required files present
2) Space/API health + reset ping
3) OpenEnv validation
4) Task/grader bounds sanity
5) Inference run + stdout contract + timeout
6) Docker build (optional skip)
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

START_RE = re.compile(r"^\[START\] task=\S+ env=\S+ model=.+$")
STEP_RE = re.compile(
    r"^\[STEP\] step=\d+ action=.+ reward=-?\d+\.\d{2} done=(true|false) error=(null|.+)$"
)
END_RE = re.compile(
    r"^\[END\] success=(true|false) steps=\d+ score=-?\d+\.\d{2,} rewards=.*$"
)


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        text=True,
        errors="replace",
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _require_files() -> bool:
    needed = [ROOT / "inference.py", ROOT / "openenv.yaml", ROOT / "models.py"]
    missing = [str(p.relative_to(ROOT)) for p in needed if not p.exists()]
    if missing:
        _fail(f"Missing required files: {', '.join(missing)}")
        return False
    _ok("Required files present (inference.py, openenv.yaml, models.py)")
    return True


def _check_env_vars() -> None:
    required = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
    for name in required:
        val = (os.getenv(name) or "").strip()
        if val:
            _ok(f"{name} is set")
        else:
            _warn(f"{name} is missing in current shell")


def _check_space(env_url: str) -> bool:
    try:
        h = requests.get(f"{env_url}/health", timeout=20)
        if h.status_code != 200:
            _fail(f"/health returned {h.status_code}")
            return False
        _ok("Space/API /health returned 200")

        r = requests.post(f"{env_url}/reset", json={"task_type": "config", "seed": 42}, timeout=30)
        if r.status_code != 200:
            _fail(f"/reset returned {r.status_code}")
            return False
        _ok("Space/API /reset responded successfully")
        return True
    except Exception as exc:
        _fail(f"Space/API check failed: {exc}")
        return False


def _openenv_validate() -> bool:
    exe = ROOT / ".venv" / "Scripts" / "openenv.exe"
    cmd = [str(exe), "validate"] if exe.exists() else ["openenv", "validate"]
    out = _run(cmd, timeout=240)
    if out.returncode != 0:
        _fail("openenv validate failed")
        print(out.stdout)
        print(out.stderr)
        return False
    _ok("openenv validate passed")
    return True


def _grader_sanity() -> bool:
    code = (
        "from tasks.task1_config import ConfigFixerTask\n"
        "from tasks.task2_logs import LogDiagnosticianTask\n"
        "from tasks.task3_pipeline import PipelineDebuggerTask\n"
        "from graders.grader1 import grade_config_episode\n"
        "from graders.grader2 import grade_log_episode\n"
        "from graders.grader3 import grade_pipeline_episode\n"
        "tasks=[('config',ConfigFixerTask(),grade_config_episode),('logs',LogDiagnosticianTask(),grade_log_episode),('pipeline',PipelineDebuggerTask(),grade_pipeline_episode)]\n"
        "ok=True\n"
        "for name,t,g in tasks:\n"
        " s=t.scenarios[0]\n"
        " score=float(g(s,[],step_count=1,max_steps=20))\n"
        " in_range=0.0<=score<=1.0\n"
        " print(name,len(t.scenarios),score,in_range)\n"
        " ok=ok and in_range\n"
        "raise SystemExit(0 if ok else 1)\n"
    )
    out = _run([PYTHON, "-c", code], timeout=120)
    if out.returncode != 0:
        _fail("Task/grader score-range check failed")
        print(out.stdout)
        print(out.stderr)
        return False
    _ok("3+ task families present and grader outputs are in [0.0, 1.0]")
    return True


def _check_inference(env_url: str, timeout_s: int) -> bool:
    env = os.environ.copy()
    env["ENV_URL"] = env_url
    start = time.time()
    proc = subprocess.run(
        [PYTHON, "inference.py"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=timeout_s,
        env=env,
        check=False,
    )
    duration = time.time() - start

    if proc.returncode != 0:
        _fail(f"inference.py exited non-zero ({proc.returncode})")
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        return False

    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    starts = [ln for ln in lines if ln.startswith("[START]")]
    steps = [ln for ln in lines if ln.startswith("[STEP]")]
    ends = [ln for ln in lines if ln.startswith("[END]")]

    if not starts or not steps or not ends:
        _fail("inference.py did not emit required [START]/[STEP]/[END] lines")
        return False

    if not all(START_RE.match(ln) for ln in starts[:3]):
        _fail("[START] format check failed")
        return False
    if not all(STEP_RE.match(ln) for ln in steps[:10]):
        _fail("[STEP] format check failed")
        return False
    if not all(END_RE.match(ln) for ln in ends):
        _fail("[END] format check failed")
        return False

    if duration > 20 * 60:
        _fail(f"inference runtime exceeded 20 min ({duration:.1f}s)")
        return False

    _ok(f"inference.py completed in {duration:.1f}s with valid structured stdout")
    return True


def _docker_build(skip_docker: bool) -> bool:
    if skip_docker:
        _warn("Docker check skipped by flag")
        return True

    ver = _run(["docker", "version"], timeout=30)
    if ver.returncode != 0:
        _fail("Docker daemon unavailable; cannot run docker build")
        return False

    context = ROOT if (ROOT / "Dockerfile").exists() else ROOT / "server"
    build = _run(["docker", "build", str(context), "-t", "mltriage-pre-submit"], timeout=1200)
    if build.returncode != 0:
        _fail("docker build failed")
        print(build.stdout[-2000:])
        print(build.stderr[-2000:])
        return False

    _ok("docker build succeeded")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pre-submission checks")
    parser.add_argument("--env-url", default=os.getenv("ENV_URL", "https://devxaves-mltriageenv.hf.space"))
    parser.add_argument("--timeout", type=int, default=1200, help="inference.py timeout seconds")
    parser.add_argument("--skip-docker", action="store_true")
    args = parser.parse_args()

    print("=== MLTriageEnv Pre-Submission Check ===")
    _check_env_vars()

    checks = [
        _require_files(),
        _check_space(args.env_url),
        _openenv_validate(),
        _grader_sanity(),
        _check_inference(args.env_url, args.timeout),
        _docker_build(args.skip_docker),
    ]

    if all(checks):
        print("\n[PASS] All checks passed. Submission is ready.")
        return 0

    print("\n[FAIL] One or more checks failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
