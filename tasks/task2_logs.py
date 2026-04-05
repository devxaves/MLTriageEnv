"""
Task 2: Log Diagnostician (Medium)

Agent reads ML training logs containing a specific failure mode,
must diagnose the failure and propose the correct intervention.
"""

from typing import Any, Dict, List

from .base import BaseTask


# Canonical failure modes the agent must classify
FAILURE_MODES = frozenset([
    "gradient_explosion",
    "gradient_vanishing",
    "learning_rate_too_high",
    "learning_rate_too_low",
    "overfitting",
    "underfitting",
    "data_leakage",
    "class_imbalance",
    "nan_in_data",
    "wrong_loss_function",
    "dead_relu",
    "batch_size_too_large",
    "mode_collapse",
    "lr_schedule_misconfigured",
])

# Aliases for fuzzy matching
FAILURE_ALIASES: Dict[str, List[str]] = {
    "gradient_explosion": ["gradient explosion", "exploding gradient", "grad explod", "inf loss", "nan loss from gradient"],
    "gradient_vanishing": ["gradient vanishing", "vanishing gradient", "zero gradient", "grad vanish"],
    "learning_rate_too_high": ["lr too high", "learning rate too high", "high learning rate", "lr oscillat"],
    "learning_rate_too_low": ["lr too low", "learning rate too low", "low learning rate", "slow convergence", "slow learning"],
    "overfitting": ["overfitting", "overfit", "train val gap", "validation increasing", "val loss increasing"],
    "underfitting": ["underfitting", "underfit", "insufficient capacity", "model too simple"],
    "data_leakage": ["data leakage", "data leak", "leakage", "val higher than test"],
    "class_imbalance": ["class imbalance", "imbalanced", "imbalance", "minority class", "zero recall"],
    "nan_in_data": ["nan in data", "nan data", "nan input", "missing values in data", "null in features"],
    "wrong_loss_function": ["wrong loss", "incorrect loss", "loss mismatch", "loss function wrong"],
    "dead_relu": ["dead relu", "dying relu", "relu dead", "zero activation"],
    "batch_size_too_large": ["batch size too large", "oom", "out of memory", "large batch"],
    "mode_collapse": ["mode collapse", "gan collapse", "generator collapse", "single mode"],
    "lr_schedule_misconfigured": ["lr schedule", "learning rate schedule", "scheduler", "warmup mismatch"],
}


INTERVENTION_KEYWORDS: Dict[str, List[str]] = {
    "gradient_explosion": ["clip", "gradient clip", "reduce lr", "lower learning rate"],
    "gradient_vanishing": ["batch norm", "residual", "skip connection", "relu", "initialization"],
    "learning_rate_too_high": ["reduce", "lower", "decrease", "smaller lr", "10x"],
    "learning_rate_too_low": ["increase", "higher", "raise", "larger lr", "lr finder"],
    "overfitting": ["dropout", "regulariz", "weight decay", "data augment", "early stop"],
    "underfitting": ["capacity", "larger model", "more layers", "more parameters", "reduce regulariz"],
    "data_leakage": ["split", "audit", "separate", "remove leaked", "temporal"],
    "class_imbalance": ["weight", "oversample", "smote", "undersample", "balanced"],
    "nan_in_data": ["clean", "impute", "fill", "nan", "missing", "preprocess"],
    "wrong_loss_function": ["change loss", "cross entropy", "correct loss", "match task"],
    "dead_relu": ["leaky", "elu", "prelu", "reduce lr", "initialization"],
    "batch_size_too_large": ["reduce batch", "smaller batch", "gradient accumulation", "accumulate"],
    "mode_collapse": ["diversity", "wgan", "spectral", "minibatch discrimination"],
    "lr_schedule_misconfigured": ["warmup", "schedule", "decay", "config", "match"],
}


class LogDiagnosticianTask(BaseTask):
    """Medium task — diagnose failure mode from training logs."""

    scenario_file = "log_scenarios.json"

    def process_action(
        self,
        action_type: str,
        target: str,
        value: str,
        scenario: Dict[str, Any],
        history: List[Dict[str, str]],
        issues_found: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        correct_mode = scenario["failure_mode"]
        correct_intervention = scenario["correct_intervention"]
        red_herrings = scenario.get("red_herrings", [])

        # --- INSPECT action ---
        if action_type == "inspect":
            if target.lower() in ["logs", "log", "training_log", "artifact"]:
                return {
                    "reward": 0.02,
                    "feedback": f"Training log loaded. Analyze the output to identify the failure mode.",
                    "issue_found": None,
                    "task_complete": False,
                }
            if target.lower() in ["red_herring", "warning", "warnings"]:
                if red_herrings:
                    return {
                        "reward": 0.0,
                        "feedback": f"Warning messages found: {'; '.join(red_herrings)}. These may or may not be relevant.",
                        "issue_found": None,
                        "task_complete": False,
                    }
            return {
                "reward": 0.0,
                "feedback": f"Inspect target '{target}'. Focus on loss trends, gradient norms, and metric patterns.",
                "issue_found": None,
                "task_complete": False,
            }

        # --- DIAGNOSE action ---
        if action_type == "diagnose":
            already_diagnosed = any(
                i.get("type") == "diagnosis" for i in issues_found
            )
            if already_diagnosed:
                return {
                    "reward": 0.0,
                    "feedback": "You already provided a diagnosis. Use 'fix_stage' to propose an intervention.",
                    "issue_found": None,
                    "task_complete": False,
                }

            # Check if the diagnosis matches the correct failure mode
            is_correct = _matches_failure_mode(target, value, correct_mode)
            if is_correct:
                return {
                    "reward": 0.35,
                    "feedback": (
                        f"Correct diagnosis: '{correct_mode}'. "
                        f"Now propose an intervention using 'fix_stage'."
                    ),
                    "issue_found": {
                        "type": "diagnosis",
                        "mode": correct_mode,
                        "correct": True,
                    },
                    "task_complete": False,
                }
            else:
                # Check if partially correct (right category, wrong specifics)
                partial = _partial_match(target, value, correct_mode)
                if partial:
                    return {
                        "reward": 0.1,
                        "feedback": (
                            f"Your diagnosis is in the right direction but not precise. "
                            f"The actual failure mode is: {correct_mode}."
                        ),
                        "issue_found": {
                            "type": "diagnosis",
                            "mode": correct_mode,
                            "correct": False,
                        },
                        "task_complete": False,
                    }
                return {
                    "reward": -0.05,
                    "feedback": (
                        f"Incorrect diagnosis. '{target}: {value}' does not match. "
                        f"Re-analyze the log patterns carefully."
                    ),
                    "issue_found": None,
                    "task_complete": False,
                }

        # --- FIX_STAGE action (used for proposing intervention) ---
        if action_type == "fix_stage":
            already_fixed = any(
                i.get("type") == "intervention" for i in issues_found
            )
            if already_fixed:
                return {
                    "reward": 0.0,
                    "feedback": "You already proposed an intervention.",
                    "issue_found": None,
                    "task_complete": False,
                }

            # Check if intervention is reasonable
            keywords = INTERVENTION_KEYWORDS.get(correct_mode, [])
            match_score = sum(
                1 for kw in keywords if kw.lower() in value.lower()
            )
            if match_score >= 1:
                return {
                    "reward": 0.3,
                    "feedback": (
                        f"Good intervention! Your fix aligns with the recommended: "
                        f"'{correct_intervention}'."
                    ),
                    "issue_found": {
                        "type": "intervention",
                        "correct": True,
                    },
                    "task_complete": False,
                }
            else:
                return {
                    "reward": 0.05,
                    "feedback": (
                        f"Your proposed intervention doesn't strongly match the expected fix. "
                        f"Recommended: '{correct_intervention}'."
                    ),
                    "issue_found": {
                        "type": "intervention",
                        "correct": False,
                    },
                    "task_complete": False,
                }

        # --- VALIDATE ---
        if action_type == "validate":
            has_diag = any(
                i.get("type") == "diagnosis" and i.get("correct")
                for i in issues_found
            )
            has_fix = any(
                i.get("type") == "intervention" and i.get("correct")
                for i in issues_found
            )
            if has_diag and has_fix:
                return {
                    "reward": 0.1,
                    "feedback": "Validation passed — diagnosis and intervention are both correct!",
                    "issue_found": None,
                    "task_complete": True,
                }
            else:
                missing = []
                if not has_diag:
                    missing.append("correct diagnosis")
                if not has_fix:
                    missing.append("correct intervention")
                return {
                    "reward": 0.0,
                    "feedback": f"Validation incomplete. Still need: {', '.join(missing)}.",
                    "issue_found": None,
                    "task_complete": False,
                }

        # --- DONE ---
        if action_type == "done":
            has_diag = any(
                i.get("type") == "diagnosis" and i.get("correct")
                for i in issues_found
            )
            has_fix = any(
                i.get("type") == "intervention" and i.get("correct")
                for i in issues_found
            )
            score = 0.0
            if has_diag:
                score += 0.5
            if has_fix:
                score += 0.5
            return {
                "reward": 0.0,
                "feedback": f"Episode ended. Score: {score:.1f}/1.0.",
                "issue_found": None,
                "task_complete": True,
            }

        return {
            "reward": 0.0,
            "feedback": f"Action '{action_type}' not applicable for log diagnosis tasks.",
            "issue_found": None,
            "task_complete": False,
        }


def _matches_failure_mode(target: str, value: str, correct: str) -> bool:
    """Check if agent's diagnosis matches the correct failure mode."""
    combined = f"{target} {value}".lower()
    # Exact match
    if correct.lower() in combined or correct.replace("_", " ").lower() in combined:
        return True
    # Alias match
    aliases = FAILURE_ALIASES.get(correct, [])
    return any(alias.lower() in combined for alias in aliases)


def _partial_match(target: str, value: str, correct: str) -> bool:
    """Check for partial match (e.g., gradient-related for gradient explosion)."""
    combined = f"{target} {value}".lower()
    # Check if any word from the correct mode appears
    words = correct.replace("_", " ").split()
    return any(w.lower() in combined for w in words if len(w) > 3)
