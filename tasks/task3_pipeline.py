"""
Task 3: Pipeline Debugger (Hard)

Agent must trace silent failures across a multi-stage ML pipeline.
Each scenario has 2 faulty stages the agent must identify and fix.
"""

from typing import Any, Dict, List

from .base import BaseTask


# Bug type aliases for fuzzy matching
BUG_ALIASES: Dict[str, List[str]] = {
    "fit_on_test": ["fit on test", "fitted on full", "scaler on all data", "fit before split", "data leakage via scaler"],
    "metric_mismatch": ["wrong metric", "metric mismatch", "accuracy for imbalanced", "misleading metric"],
    "wrong_split_strategy": ["wrong split", "random split", "shuffle time series", "temporal leak"],
    "target_leakage": ["target leakage", "leaked feature", "post-event variable", "future information"],
    "missing_value_wrong_strategy": ["wrong imputation", "mean imputation skewed", "imputation strategy"],
    "evaluation_on_train": ["eval on train", "evaluated on training", "train set evaluation"],
    "label_encoding_instead_of_onehot": ["label encoding", "ordinal encoding", "false ordering", "nominal encoding"],
    "class_imbalance_ignored": ["class imbalance", "imbalanced", "no class weight", "majority class"],
    "wrong_scaler_applied": ["wrong scaler", "standardscaler on bounded", "scaler mismatch"],
    "wrong_aggregation": ["wrong aggregation", "mean instead of sum", "aggregation error"],
    "off_by_one_in_sequences": ["off by one", "sequence window", "lookback", "include current"],
    "duplicate_rows_not_removed": ["duplicate", "duplicated rows", "dedup", "remove duplicates"],
    "preprocessing_after_split": ["fit after split on both", "fit on train+val", "scaler on combined"],
    "shuffle_before_split": ["shuffle before split", "shuffled time series", "random shuffle temporal"],
}


FIX_KEYWORDS: Dict[str, List[str]] = {
    "fit_on_test": ["fit on train", "training only", "transform val", "transform test"],
    "metric_mismatch": ["f1", "auc", "roc", "precision-recall", "balanced metric"],
    "wrong_split_strategy": ["temporal", "chronological", "time-based", "group"],
    "target_leakage": ["remove", "drop", "exclude", "post-event"],
    "missing_value_wrong_strategy": ["median", "indicator", "missing flag", "informative"],
    "evaluation_on_train": ["validation", "test set", "held-out", "holdout"],
    "label_encoding_instead_of_onehot": ["one-hot", "onehot", "dummy", "get_dummies"],
    "class_imbalance_ignored": ["balanced", "smote", "oversample", "class_weight", "weight"],
    "wrong_scaler_applied": ["minmax", "min-max", "no scale", "separate scaler"],
    "wrong_aggregation": ["sum", "total", "count", "aggregate correctly"],
    "off_by_one_in_sequences": ["exclude current", "t-1", "shift", "remove current"],
    "duplicate_rows_not_removed": ["remove duplicate", "drop duplicate", "deduplicate", "dedup"],
    "preprocessing_after_split": ["training data only", "fit on train", "separate fit"],
    "shuffle_before_split": ["no shuffle", "chronological", "temporal order", "ordered"],
}


class PipelineDebuggerTask(BaseTask):
    """Hard task — trace and fix bugs in multi-stage ML pipeline."""

    scenario_file = "pipeline_scenarios.json"

    def process_action(
        self,
        action_type: str,
        target: str,
        value: str,
        scenario: Dict[str, Any],
        history: List[Dict[str, str]],
        issues_found: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        faulty_stages = scenario.get("faulty_stages", [])
        bugs = scenario.get("bugs", {})
        valid_stages = scenario.get("valid_stages", [])

        # --- INSPECT action ---
        if action_type == "inspect":
            if target in bugs:
                bug = bugs[target]
                return {
                    "reward": 0.05,
                    "feedback": (
                        f"Stage '{target}': {bug['description']}. "
                        f"Bug type: {bug['bug_type']}."
                    ),
                    "issue_found": None,
                    "task_complete": False,
                }
            elif target in valid_stages:
                return {
                    "reward": 0.0,
                    "feedback": f"Stage '{target}' appears to be functioning correctly.",
                    "issue_found": None,
                    "task_complete": False,
                }
            else:
                # Target is a stage name not in bugs or valid_stages
                all_stages = list(bugs.keys()) + valid_stages
                return {
                    "reward": 0.0,
                    "feedback": (
                        f"Stage '{target}' — reviewing. "
                        f"Available stages: {', '.join(all_stages)}."
                    ),
                    "issue_found": None,
                    "task_complete": False,
                }

        # --- DIAGNOSE action ---
        if action_type == "diagnose":
            if target in bugs:
                bug = bugs[target]
                already = any(
                    i.get("stage") == target and i.get("type") == "diagnosis"
                    for i in issues_found
                )
                if already:
                    return {
                        "reward": 0.0,
                        "feedback": f"Stage '{target}' already diagnosed.",
                        "issue_found": None,
                        "task_complete": False,
                    }
                # Check match
                correct_type = bug["bug_type"]
                if _matches_bug(value, correct_type):
                    return {
                        "reward": 0.15,
                        "feedback": (
                            f"Correct! Stage '{target}' has bug: {correct_type}. "
                            f"{bug['description']}"
                        ),
                        "issue_found": {
                            "stage": target,
                            "type": "diagnosis",
                            "bug_type": correct_type,
                            "correct": True,
                        },
                        "task_complete": False,
                    }
                else:
                    partial = _partial_bug_match(value, correct_type)
                    if partial:
                        return {
                            "reward": 0.05,
                            "feedback": (
                                f"Stage '{target}' does have a bug, but your diagnosis is imprecise. "
                                f"Actual bug: {correct_type}."
                            ),
                            "issue_found": {
                                "stage": target,
                                "type": "diagnosis",
                                "bug_type": correct_type,
                                "correct": False,
                            },
                            "task_complete": False,
                        }
                    return {
                        "reward": -0.02,
                        "feedback": (
                            f"Stage '{target}' has a bug, but '{value}' is incorrect. "
                            f"Look more carefully at the stage description."
                        ),
                        "issue_found": None,
                        "task_complete": False,
                    }
            elif target in valid_stages:
                return {
                    "reward": -0.05,
                    "feedback": f"Stage '{target}' is actually correct. False diagnosis.",
                    "issue_found": None,
                    "task_complete": False,
                }
            else:
                return {
                    "reward": -0.02,
                    "feedback": f"Stage '{target}' not recognized in the pipeline.",
                    "issue_found": None,
                    "task_complete": False,
                }

        # --- FIX_STAGE action ---
        if action_type == "fix_stage":
            if target in bugs:
                bug = bugs[target]
                already_fixed = any(
                    i.get("stage") == target and i.get("type") == "fix"
                    for i in issues_found
                )
                if already_fixed:
                    return {
                        "reward": 0.0,
                        "feedback": f"Stage '{target}' already has a proposed fix.",
                        "issue_found": None,
                        "task_complete": False,
                    }

                correct_type = bug["bug_type"]
                fix_kw = FIX_KEYWORDS.get(correct_type, [])
                match = sum(1 for kw in fix_kw if kw.lower() in value.lower())
                if match >= 1:
                    return {
                        "reward": 0.2,
                        "feedback": (
                            f"Good fix for stage '{target}'! "
                            f"Recommended: {bug['fix']}."
                        ),
                        "issue_found": {
                            "stage": target,
                            "type": "fix",
                            "correct": True,
                        },
                        "task_complete": False,
                    }
                else:
                    return {
                        "reward": 0.03,
                        "feedback": (
                            f"Fix for '{target}' doesn't match expected approach. "
                            f"Recommended: {bug['fix']}."
                        ),
                        "issue_found": {
                            "stage": target,
                            "type": "fix",
                            "correct": False,
                        },
                        "task_complete": False,
                    }
            elif target in valid_stages:
                return {
                    "reward": -0.05,
                    "feedback": f"Stage '{target}' doesn't need fixing — it's working correctly.",
                    "issue_found": None,
                    "task_complete": False,
                }
            else:
                return {
                    "reward": -0.02,
                    "feedback": f"Stage '{target}' not recognized.",
                    "issue_found": None,
                    "task_complete": False,
                }

        # --- VALIDATE ---
        if action_type == "validate":
            correct_diags = sum(
                1 for i in issues_found
                if i.get("type") == "diagnosis" and i.get("correct")
            )
            correct_fixes = sum(
                1 for i in issues_found
                if i.get("type") == "fix" and i.get("correct")
            )
            total = len(bugs)
            if correct_diags == total and correct_fixes == total:
                return {
                    "reward": 0.1,
                    "feedback": "All pipeline bugs diagnosed and fixed correctly!",
                    "issue_found": None,
                    "task_complete": True,
                }
            else:
                return {
                    "reward": 0.0,
                    "feedback": (
                        f"Validation: {correct_diags}/{total} diagnosed, "
                        f"{correct_fixes}/{total} fixed correctly."
                    ),
                    "issue_found": None,
                    "task_complete": False,
                }

        # --- DONE ---
        if action_type == "done":
            total = len(bugs)
            correct_diags = sum(
                1 for i in issues_found
                if i.get("type") == "diagnosis" and i.get("correct")
            )
            correct_fixes = sum(
                1 for i in issues_found
                if i.get("type") == "fix" and i.get("correct")
            )
            diag_pct = correct_diags / max(total, 1)
            fix_pct = correct_fixes / max(total, 1)
            overall = (diag_pct + fix_pct) / 2
            return {
                "reward": 0.0,
                "feedback": (
                    f"Episode ended. Diagnosed: {correct_diags}/{total}, "
                    f"Fixed: {correct_fixes}/{total}. Overall: {overall:.0%}."
                ),
                "issue_found": None,
                "task_complete": True,
            }

        return {
            "reward": 0.0,
            "feedback": f"Action '{action_type}' not applicable for pipeline debugging.",
            "issue_found": None,
            "task_complete": False,
        }


def _matches_bug(value: str, correct_type: str) -> bool:
    """Check if value matches the correct bug type."""
    v = value.lower()
    if correct_type.lower() in v or correct_type.replace("_", " ").lower() in v:
        return True
    aliases = BUG_ALIASES.get(correct_type, [])
    return any(alias.lower() in v for alias in aliases)


def _partial_bug_match(value: str, correct_type: str) -> bool:
    """Check for word-level partial match."""
    v = value.lower()
    words = correct_type.replace("_", " ").split()
    return any(w.lower() in v for w in words if len(w) > 3)
