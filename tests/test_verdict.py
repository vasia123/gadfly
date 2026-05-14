from gadfly.verdict import Verdict


def test_silent_ok_has_no_hook_output():
    assert Verdict.silent_ok().to_hook_output() is None


def test_professional_true_emits_no_hook_output():
    v = Verdict(professional=True, reason="", suggestion="")
    assert v.to_hook_output() is None


def test_unprofessional_builds_additional_context():
    v = Verdict(professional=False, reason="adds TODO instead of implementing", suggestion="implement it")
    out = v.to_hook_output()
    assert out is not None
    assert out["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    msg = out["hookSpecificOutput"]["additionalContext"]
    assert msg.startswith("Watchdog (gadfly):")
    assert "TODO" in msg
    assert "Suggestion: implement it" in msg


def test_from_tool_input_handles_missing_fields():
    v = Verdict.from_tool_input({"professional": False})
    assert v.professional is False
    assert v.reason == ""
    assert v.suggestion == ""


def test_from_tool_input_coerces_types():
    v = Verdict.from_tool_input({"professional": "yes-truthy", "reason": None, "suggestion": None})
    assert v.professional is True
    assert v.reason == ""
    assert v.suggestion == ""
