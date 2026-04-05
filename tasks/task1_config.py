"""
Task 1: Config Fixer (Easy)

Agent must inspect broken ML training config YAML, identify broken fields,
and patch them with correct values. Rewards partial fixes.
"""

from typing import Any, Dict, List

from .base import BaseTask


class ConfigFixerTask(BaseTask):
    """Easy task — find and fix broken config fields."""

    scenario_file = "config_scenarios.json"

    def process_action(
        self,
        action_type: str,
        target: str,
        value: str,
        scenario: Dict[str, Any],
        history: List[Dict[str, str]],
        issues_found: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        broken = scenario.get("broken_fields", {})
        valid = scenario.get("valid_fields", [])

        # --- INSPECT action ---
        if action_type == "inspect":
            # Agent wants more info about a field
            if target in broken:
                info = broken[target]
                if info.get("issue_type", "none") != "none":
                    return {
                        "reward": 0.05,
                        "feedback": (
                            f"Field '{target}' currently has value '{info['wrong_value']}'. "
                            f"Issue type: {info['issue_type']}."
                        ),
                        "issue_found": None,
                        "task_complete": False,
                    }
                else:
                    return {
                        "reward": 0.0,
                        "feedback": f"Field '{target}' has value '{info['wrong_value']}'. This field appears correct.",
                        "issue_found": None,
                        "task_complete": False,
                    }
            elif target in valid:
                return {
                    "reward": 0.0,
                    "feedback": f"Field '{target}' is correctly configured. No issues found.",
                    "issue_found": None,
                    "task_complete": False,
                }
            else:
                return {
                    "reward": 0.0,
                    "feedback": f"Field '{target}' not found in this config. Check available fields.",
                    "issue_found": None,
                    "task_complete": False,
                }

        # --- DIAGNOSE action ---
        if action_type == "diagnose":
            if target in broken:
                info = broken[target]
                if info.get("issue_type", "none") != "none":
                    already_found = any(
                        i.get("field") == target for i in issues_found
                    )
                    if already_found:
                        return {
                            "reward": 0.0,
                            "feedback": f"You already identified this issue with '{target}'.",
                            "issue_found": None,
                            "task_complete": False,
                        }
                    # Check if diagnosis is reasonable
                    issue_type = info["issue_type"]
                    diagnosis_match = (
                        issue_type.lower() in value.lower()
                        or any(
                            kw in value.lower()
                            for kw in _issue_keywords(issue_type)
                        )
                    )
                    if diagnosis_match:
                        return {
                            "reward": 0.15,
                            "feedback": (
                                f"Correct diagnosis! Field '{target}' has issue: {issue_type}. "
                                f"Current value: {info['wrong_value']}, expected: {info['correct_value']}."
                            ),
                            "issue_found": {
                                "field": target,
                                "type": "diagnosis",
                                "issue_type": issue_type,
                            },
                            "task_complete": False,
                        }
                    else:
                        return {
                            "reward": 0.05,
                            "feedback": (
                                f"Field '{target}' does have an issue, but your diagnosis "
                                f"'{value}' is not precise. The actual issue is: {issue_type}."
                            ),
                            "issue_found": {
                                "field": target,
                                "type": "partial_diagnosis",
                                "issue_type": issue_type,
                            },
                            "task_complete": False,
                        }
                else:
                    return {
                        "reward": -0.05,
                        "feedback": f"Field '{target}' is actually correct. False diagnosis.",
                        "issue_found": None,
                        "task_complete": False,
                    }
            else:
                return {
                    "reward": -0.05,
                    "feedback": f"Field '{target}' is not broken or does not exist.",
                    "issue_found": None,
                    "task_complete": False,
                }

        # --- PATCH action ---
        if action_type == "patch":
            if target in broken:
                info = broken[target]
                if info.get("issue_type", "none") == "none":
                    return {
                        "reward": -0.1,
                        "feedback": f"Field '{target}' was already correct. Unnecessary patch!",
                        "issue_found": None,
                        "task_complete": False,
                    }
                # Check if already patched
                already_patched = any(
                    i.get("field") == target and i.get("type") == "patch"
                    for i in issues_found
                )
                if already_patched:
                    return {
                        "reward": 0.0,
                        "feedback": f"Field '{target}' has already been patched.",
                        "issue_found": None,
                        "task_complete": False,
                    }
                # Check if patch value is correct
                correct = str(info["correct_value"]).strip().lower()
                proposed = str(value).strip().lower()
                if proposed == correct:
                    return {
                        "reward": 0.25,
                        "feedback": (
                            f"Excellent! Field '{target}' patched from "
                            f"'{info['wrong_value']}' to '{info['correct_value']}' — correct fix!"
                        ),
                        "issue_found": {
                            "field": target,
                            "type": "patch",
                            "correct": True,
                        },
                        "task_complete": False,
                    }
                else:
                    # Partial credit — right field, wrong value
                    return {
                        "reward": 0.08,
                        "feedback": (
                            f"Correct field '{target}', but value '{value}' is not right. "
                            f"Expected '{info['correct_value']}'."
                        ),
                        "issue_found": {
                            "field": target,
                            "type": "patch",
                            "correct": False,
                        },
                        "task_complete": False,
                    }
            else:
                return {
                    "reward": -0.05,
                    "feedback": f"Field '{target}' is not a broken field. No patch needed.",
                    "issue_found": None,
                    "task_complete": False,
                }

        # --- VALIDATE action ---
        if action_type == "validate":
            total_broken = sum(
                1
                for v in broken.values()
                if v.get("issue_type", "none") != "none"
            )
            patches = [
                i for i in issues_found
                if i.get("type") == "patch" and i.get("correct")
            ]
            if len(patches) == total_broken:
                return {
                    "reward": 0.1,
                    "feedback": "All issues fixed! Config validated successfully.",
                    "issue_found": None,
                    "task_complete": True,
                }
            else:
                return {
                    "reward": 0.0,
                    "feedback": (
                        f"Validation: {len(patches)}/{total_broken} issues fixed. "
                        f"Some issues remain."
                    ),
                    "issue_found": None,
                    "task_complete": False,
                }

        # --- DONE action ---
        if action_type == "done":
            total_broken = sum(
                1
                for v in broken.values()
                if v.get("issue_type", "none") != "none"
            )
            patches = [
                i for i in issues_found
                if i.get("type") == "patch" and i.get("correct")
            ]
            completion = len(patches) / max(total_broken, 1)
            return {
                "reward": 0.0,
                "feedback": (
                    f"Episode ended. Fixed {len(patches)}/{total_broken} issues "
                    f"({completion:.0%} complete)."
                ),
                "issue_found": None,
                "task_complete": True,
            }

        return {
            "reward": 0.0,
            "feedback": f"Unknown action type: {action_type}",
            "issue_found": None,
            "task_complete": False,
        }


def _issue_keywords(issue_type: str) -> List[str]:
    """Return fuzzy-match keywords for each issue type."""
    mapping = {
        "typo": ["typo", "misspell", "spelling", "wrong name"],
        "value_out_of_range": ["range", "too high", "too large", "out of range", "invalid value"],
        "negative_value": ["negative", "below zero", "minus", "invalid sign"],
        "zero_value": ["zero", "0", "missing", "not set"],
        "wrong_type": ["type", "string", "wrong format", "cast", "not numeric"],
        "wrong_loss_for_task": ["wrong loss", "mismatch", "incorrect loss", "loss function"],
    }
    return mapping.get(issue_type, [issue_type])
