"""
Grader 3: Pipeline Debugger scoring.

Pure Python, deterministic, no LLM. Returns 0.0–1.0.
"""

from typing import Any, Dict, List


def grade_pipeline_episode(
    scenario: Dict[str, Any],
    issues_found: List[Dict[str, Any]],
    step_count: int,
    max_steps: int = 20,
) -> float:
    """Grade a pipeline debugger episode.

    Scoring breakdown (max 1.0):
        - Correct stage diagnosis:   0.15 per stage (max 0.30)
        - Correct stage fix:         0.25 per stage (max 0.50)
        - Efficiency bonus:          up to 0.10
        - Partial credit:            reduced scores for imprecise diagnoses/fixes
        - False positive penalty:    -0.05 per false positive

    Scores are normalized to [0.0, 1.0].
    """
    bugs = scenario.get("bugs", {})
    total_bugs = len(bugs)
    if total_bugs == 0:
        return 1.0

    # Count correct diagnoses
    correct_diag = sum(
        1 for i in issues_found
        if i.get("type") == "diagnosis" and i.get("correct") is True
    )
    partial_diag = sum(
        1 for i in issues_found
        if i.get("type") == "diagnosis" and i.get("correct") is False
    )

    # Count correct fixes
    correct_fix = sum(
        1 for i in issues_found
        if i.get("type") == "fix" and i.get("correct") is True
    )
    partial_fix = sum(
        1 for i in issues_found
        if i.get("type") == "fix" and i.get("correct") is False
    )

    # Diagnosis score (max 0.30)
    diag_score = (correct_diag / total_bugs) * 0.25 + (partial_diag / total_bugs) * 0.05

    # Fix score (max 0.50)
    fix_score = (correct_fix / total_bugs) * 0.45 + (partial_fix / total_bugs) * 0.10

    # Efficiency bonus (max 0.10)
    if correct_diag == total_bugs and correct_fix == total_bugs and step_count < max_steps:
        steps_saved = (max_steps - step_count) / max_steps
        efficiency_bonus = steps_saved * 0.10
    else:
        efficiency_bonus = 0.0

    # Completion bonus for finding ALL issues
    if correct_diag == total_bugs and correct_fix == total_bugs:
        completion_bonus = 0.15
    elif correct_diag == total_bugs or correct_fix == total_bugs:
        completion_bonus = 0.05
    else:
        completion_bonus = 0.0

    raw_score = diag_score + fix_score + efficiency_bonus + completion_bonus
    return max(0.0, min(1.0, raw_score))
