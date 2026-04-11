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
        if scenario.get("mode") == "evidence_triage" or scenario.get("root_cause_service"):
            return self._process_evidence_triage(
                action_type=action_type,
                target=target,
                value=value,
                scenario=scenario,
                history=history,
                issues_found=issues_found,
            )

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

    def _process_evidence_triage(
        self,
        action_type: str,
        target: str,
        value: str,
        scenario: Dict[str, Any],
        history: List[Dict[str, str]],
        issues_found: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        root_service = scenario.get("root_cause_service", "")
        red_herring = scenario.get("red_herring_service", "")
        expected_priority = str(scenario.get("expected_priority", "P1")).upper()
        required_evidence = set(scenario.get("required_evidence", ["logs", "metrics", "dependency_graph"]))

        def _already_has(e_type: str, service: str = "") -> bool:
            return any(
                i.get("type") == e_type and (not service or i.get("service") == service)
                for i in issues_found
            )

        def _action_repeats(a_type: str, tgt: str) -> int:
            return sum(
                1 for h in history
                if h.get("action_type") == a_type and h.get("target") == tgt
            )

        # Investigation actions
        if action_type in {"inspect_logs", "query_metrics", "check_dependency_graph", "inspect"}:
            if action_type == "inspect":
                action_type = "inspect_logs"

            repeats = _action_repeats(action_type, target)
            repeat_penalty = -0.02 if repeats > 2 else 0.0

            if action_type == "inspect_logs":
                if target == root_service and not _already_has("evidence", root_service):
                    return {
                        "reward": 0.10 + repeat_penalty,
                        "feedback": f"Logs indicate correlated failures around '{root_service}'.",
                        "issue_found": {"type": "evidence", "evidence": "logs", "service": root_service, "correct": True},
                        "task_complete": False,
                    }
                if target == red_herring:
                    return {
                        "reward": 0.03 + repeat_penalty,
                        "feedback": f"'{red_herring}' shows noisy warnings but no causal failure signal yet.",
                        "issue_found": {"type": "evidence", "evidence": "logs", "service": red_herring, "correct": True},
                        "task_complete": False,
                    }
                return {
                    "reward": 0.0 + repeat_penalty,
                    "feedback": f"Collected logs for '{target}'. Evidence is inconclusive.",
                    "issue_found": {"type": "evidence", "evidence": "logs", "service": target, "correct": False},
                    "task_complete": False,
                }

            if action_type == "query_metrics":
                if target == root_service:
                    return {
                        "reward": 0.10 + repeat_penalty,
                        "feedback": f"Metrics spike confirms '{root_service}' as upstream trigger.",
                        "issue_found": {"type": "evidence", "evidence": "metrics", "service": root_service, "correct": True},
                        "task_complete": False,
                    }
                if target == red_herring:
                    return {
                        "reward": 0.03 + repeat_penalty,
                        "feedback": f"Metrics for '{red_herring}' are degraded but lag root-cause onset.",
                        "issue_found": {"type": "evidence", "evidence": "metrics", "service": red_herring, "correct": True},
                        "task_complete": False,
                    }
                return {
                    "reward": 0.01 + repeat_penalty,
                    "feedback": f"Metrics queried for '{target}' with limited diagnostic value.",
                    "issue_found": {"type": "evidence", "evidence": "metrics", "service": target, "correct": False},
                    "task_complete": False,
                }

            # check_dependency_graph
            if root_service:
                return {
                    "reward": 0.12 + repeat_penalty,
                    "feedback": f"Dependency graph shows '{root_service}' fans out into failing downstream services.",
                    "issue_found": {"type": "evidence", "evidence": "dependency_graph", "service": root_service, "correct": True},
                    "task_complete": False,
                }

        if action_type == "dismiss_red_herring":
            if target != red_herring:
                return {
                    "reward": -0.03,
                    "feedback": f"'{target}' is not the known red-herring service for this incident.",
                    "issue_found": None,
                    "task_complete": False,
                }

            has_red_herring_evidence = any(
                i.get("type") == "evidence" and i.get("service") == red_herring
                for i in issues_found
            )
            if has_red_herring_evidence:
                return {
                    "reward": 0.12,
                    "feedback": f"Correctly dismissed '{red_herring}' as a non-causal symptom.",
                    "issue_found": {"type": "dismissal", "service": red_herring, "correct": True},
                    "task_complete": False,
                }
            return {
                "reward": 0.02,
                "feedback": f"Dismissal accepted, but gather stronger evidence on '{red_herring}' next time.",
                "issue_found": {"type": "dismissal", "service": red_herring, "correct": False},
                "task_complete": False,
            }

        if action_type in {"finalize_triage", "done", "validate"}:
            normalized_value = value.lower()
            mentions_root = root_service.lower() in normalized_value if root_service else False
            mentions_priority = expected_priority.lower() in normalized_value
            has_dismissal = _already_has("dismissal", red_herring)
            gathered = {
                i.get("evidence") for i in issues_found
                if i.get("type") == "evidence" and i.get("service") == root_service
            }
            has_required_evidence = required_evidence.issubset(gathered)

            if action_type == "validate":
                return {
                    "reward": 0.02,
                    "feedback": (
                        f"Readiness: root_evidence={has_required_evidence}, "
                        f"red_herring_dismissed={has_dismissal}, "
                        f"priority_declared={mentions_priority}."
                    ),
                    "issue_found": None,
                    "task_complete": False,
                }

            if mentions_root and mentions_priority and has_dismissal and has_required_evidence:
                return {
                    "reward": 0.30,
                    "feedback": (
                        f"Triage finalized correctly: root cause='{root_service}', "
                        f"priority={expected_priority}, red herring dismissed='{red_herring}'."
                    ),
                    "issue_found": {
                        "type": "triage",
                        "root_cause": root_service,
                        "priority": expected_priority,
                        "correct": True,
                    },
                    "task_complete": True,
                }

            return {
                "reward": -0.08,
                "feedback": (
                    "Finalization is premature or incomplete. "
                    "Ensure root-cause evidence, red-herring dismissal, and explicit priority declaration."
                ),
                "issue_found": {
                    "type": "triage",
                    "root_cause": root_service,
                    "priority": expected_priority,
                    "correct": False,
                },
                "task_complete": action_type == "done",
            }

        return {
            "reward": -0.01,
            "feedback": f"Action '{action_type}' is unsupported in evidence-triage mode.",
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
