"""
Grader 1: Config Fixer scoring.

Pure Python, deterministic, no LLM. Returns 0.0–1.0.
"""

from typing import Any, Dict, List


def grade_config_episode(
    scenario: Dict[str, Any],
    issues_found: List[Dict[str, Any]],
    step_count: int,
    max_steps: int = 15,
) -> float:
    """Grade a config fixer episode.

    Scoring breakdown (max 1.0):
        - Correct patches:       0.30 per broken field (max = total_broken * 0.30)
        - Correct diagnoses:     0.10 per broken field
        - Efficiency bonus:      up to 0.10 for finishing under budget
        - False diagnosis:       -0.05 per false positive

    Scores are normalized to [0.0, 1.0].
    """
    broken = scenario.get("broken_fields", {})
    total_broken = sum(
        1 for v in broken.values() if v.get("issue_type", "none") != "none"
    )
    if total_broken == 0:
        return 1.0  # Nothing to fix

    # Count correct patches
    correct_patches = sum(
        1 for i in issues_found
        if i.get("type") == "patch" and i.get("correct") is True
    )

    # Count correct diagnoses (that weren't also patched)
    correct_diagnoses = sum(
        1 for i in issues_found
        if i.get("type") == "diagnosis" and i.get("issue_type") is not None
    )

    # False positives (diagnosis on non-broken fields)
    false_positives = sum(
        1 for i in issues_found
        if i.get("type") == "partial_diagnosis"
    )

    # Patch score (main component)
    patch_score = (correct_patches / total_broken) * 0.60

    # Diagnosis score
    diag_score = min(correct_diagnoses / total_broken, 1.0) * 0.25

    # Efficiency bonus — reward finishing quickly
    if step_count < max_steps and correct_patches == total_broken:
        steps_saved_ratio = (max_steps - step_count) / max_steps
        efficiency_bonus = steps_saved_ratio * 0.15
    else:
        efficiency_bonus = 0.0

    # Penalty for false positives
    penalty = false_positives * 0.02

    raw_score = patch_score + diag_score + efficiency_bonus - penalty
    return max(0.0, min(1.0, raw_score))
