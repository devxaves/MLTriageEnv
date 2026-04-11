"""
Grader 3: Pipeline Debugger scoring.

Pure Python, deterministic, no LLM. Returns 0.0–1.0.
"""

from typing import Any, Dict, List


SCORE_MIN = 0.0001
SCORE_MAX = 0.9999


def _strict_score(raw_score: float) -> float:
    return max(SCORE_MIN, min(SCORE_MAX, raw_score))


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
    if scenario.get("mode") == "evidence_triage" or scenario.get("root_cause_service"):
        root_service = scenario.get("root_cause_service", "")
        required_evidence = set(scenario.get("required_evidence", ["logs", "metrics", "dependency_graph"]))
        red_herring = scenario.get("red_herring_service", "")

        root_evidence = {
            i.get("evidence") for i in issues_found
            if i.get("type") == "evidence" and i.get("service") == root_service
        }
        evidence_ratio = len(root_evidence.intersection(required_evidence)) / max(1, len(required_evidence))

        dismissed = any(
            i.get("type") == "dismissal"
            and i.get("service") == red_herring
            and i.get("correct") is True
            for i in issues_found
        )
        triage_correct = any(
            i.get("type") == "triage" and i.get("correct") is True
            for i in issues_found
        )

        score = evidence_ratio * 0.45
        score += 0.20 if dismissed else 0.0
        score += 0.30 if triage_correct else 0.0

        if triage_correct and step_count < max_steps:
            score += ((max_steps - step_count) / max_steps) * 0.05

        return _strict_score(score)

    bugs = scenario.get("bugs", {})
    total_bugs = len(bugs)
    if total_bugs == 0:
        return SCORE_MAX

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

    # Diagnosis score (max 0.35)
    diag_score = (correct_diag / total_bugs) * 0.30 + (partial_diag / total_bugs) * 0.05

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
    return _strict_score(raw_score)
