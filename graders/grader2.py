"""
Grader 2: Log Diagnostician scoring.

Pure Python, deterministic, no LLM. Returns 0.0–1.0.
"""

from typing import Any, Dict, List


SCORE_MIN = 0.0001
SCORE_MAX = 0.9999


def _strict_score(raw_score: float) -> float:
    return max(SCORE_MIN, min(SCORE_MAX, raw_score))


def grade_log_episode(
    scenario: Dict[str, Any],
    issues_found: List[Dict[str, Any]],
    step_count: int,
    max_steps: int = 15,
) -> float:
    """Grade a log diagnostician episode.

    Scoring breakdown (max 1.0):
        - Correct diagnosis:     0.45 (exact match) or 0.15 (partial)
        - Correct intervention:  0.40
        - Efficiency bonus:      up to 0.15 for finishing quickly
        - Wrong diagnosis:       -0.05 penalty

    Scores are normalized to [0.0, 1.0].
    """
    # Check diagnosis
    diag_correct = any(
        i.get("type") == "diagnosis" and i.get("correct") is True
        for i in issues_found
    )
    diag_partial = any(
        i.get("type") == "diagnosis" and i.get("correct") is False
        for i in issues_found
    )

    # Check intervention
    intervention_correct = any(
        i.get("type") == "intervention" and i.get("correct") is True
        for i in issues_found
    )
    intervention_partial = any(
        i.get("type") == "intervention" and i.get("correct") is False
        for i in issues_found
    )

    # Diagnosis score
    if diag_correct:
        diag_score = 0.45
    elif diag_partial:
        diag_score = 0.15
    else:
        diag_score = 0.0

    # Intervention score
    if intervention_correct:
        intervention_score = 0.40
    elif intervention_partial:
        intervention_score = 0.10
    else:
        intervention_score = 0.0

    # Efficiency bonus
    if diag_correct and intervention_correct and step_count < max_steps:
        steps_saved = (max_steps - step_count) / max_steps
        efficiency_bonus = steps_saved * 0.15
    else:
        efficiency_bonus = 0.0

    raw_score = diag_score + intervention_score + efficiency_bonus
    return _strict_score(raw_score)
