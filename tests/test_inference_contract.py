import inference
from models import VALID_ACTION_TYPES


def test_print_end_includes_score(capsys):
    inference._print_end(success=True, rewards=[0.0, 0.25, 0.75])
    out = capsys.readouterr().out.strip()
    assert out.startswith("[END] success=true")
    assert " score=0.75 " in out
    assert out.endswith("rewards=0.00,0.25,0.75")


def test_candidate_env_urls_include_local_fallbacks():
    urls = inference._candidate_env_urls("http://localhost:8000")
    assert "http://localhost:8000" in urls
    assert "http://localhost:7860" in urls
    assert "http://127.0.0.1:8000" in urls
    assert "http://127.0.0.1:7860" in urls


def test_extract_json_object_from_fenced_payload():
    text = """```json
    {\"action_type\": \"inspect\", \"target\": \"x\", \"value\": \"\", \"metadata\": {}}
    ```"""
    obj = inference._extract_json_object(text)
    assert obj["action_type"] == "inspect"
    assert obj["target"] == "x"


def test_fallback_action_is_valid_for_task_types():
    for task_type in ("config", "logs", "pipeline"):
        action = inference._fallback_action({"task_type": task_type, "step_count": 0, "max_steps": 10})
        assert action.action_type in VALID_ACTION_TYPES
