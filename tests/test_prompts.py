from gadfly import prompts


def test_build_user_message_includes_all_sections():
    msg = prompts.build_user_message(
        tool_name="Edit",
        tool_input={"file_path": "auth.py", "old_string": "pass", "new_string": "return None  # TODO"},
        tool_response={"success": True},
        recent_user_requests=["implement validate_token properly"],
        last_assistant_plan="I will edit auth.py",
        recent_actions=["Read(auth.py)", "Bash: pytest -q"],
    )
    assert "User messages in this conversation" in msg
    assert "implement validate_token properly" in msg
    assert "Agent's reasoning immediately before this action" in msg
    assert "Recent prior actions" in msg
    assert "Read(auth.py)" in msg
    assert "tool_name\nEdit" in msg
    assert "auth.py" in msg
    assert "TODO" in msg


def test_build_user_message_handles_missing_context():
    msg = prompts.build_user_message(
        tool_name="Bash",
        tool_input={"command": "git commit --no-verify -m fix"},
        tool_response={"exit_code": 0, "stdout": ""},
        recent_user_requests=[],
        last_assistant_plan=None,
        recent_actions=[],
    )
    assert "(unavailable)" in msg
    assert "--no-verify" in msg


def test_submit_verdict_schema_is_well_formed():
    schema = prompts.SUBMIT_VERDICT_INPUT_SCHEMA
    assert set(schema) == {"professional", "reason", "suggestion"}
    assert schema["professional"] is bool
    assert schema["reason"] is str
    assert schema["suggestion"] is str


def test_submit_verdict_description_mentions_default():
    # The description is the only prompt-engineering knob Haiku sees for the
    # tool itself; it must remind it that "true" is the default.
    assert "default true" in prompts.SUBMIT_VERDICT_DESCRIPTION.lower() or \
        "default" in prompts.SUBMIT_VERDICT_DESCRIPTION.lower()


def test_system_prompt_treats_reasoning_as_part_of_evaluation():
    """Regression for the mini_decoder_layer_forward / proxy false-negative:
    Haiku looked at the innocuous `grep` tool call and ignored the
    rationalization paragraph that came right before it. The prompt must
    spell out that the reasoning itself is in scope."""
    p = prompts.SYSTEM_PROMPT.lower()
    assert "reasoning" in p
    # Concrete rationalization patterns we want Haiku to recognise.
    assert "proxy" in p
    assert "harder than" in p
    assert "rationalization" in p
    # And the marker for the agent.
    assert "rationalization:" in p


def test_user_message_marks_assistant_plan_as_part_of_evaluation():
    msg = prompts.build_user_message(
        tool_name="Bash",
        tool_input={"command": "grep foo bar"},
        tool_response={"stdout": ""},
        recent_user_requests=["implement Tier 2 bench"],
        last_assistant_plan="I'll use mini_decoder as a proxy because the real one is harder than needed.",
        recent_actions=[],
    )
    # The plan section must explicitly tell Haiku the plan is evaluable.
    assert "PART OF what you are evaluating" in msg
    # And the task statement must mention reasoning, not just the action.
    assert "reasoning" in msg.lower()


def test_system_prompt_includes_root_cause_discipline():
    """Regression: the rubric must spell out symptom-vs-cause explicitly,
    not just mention it in passing — otherwise Haiku won't reliably flag
    `except: pass` and `if x is None: x = default` patches as unprofessional."""
    p = prompts.SYSTEM_PROMPT.lower()
    # Top-level concept appears.
    assert "root cause" in p
    assert "symptom" in p
    # Reflective stop-and-think instruction is in.
    assert "pause" in p or "stop" in p or "ask yourself" in p
    # Several of the concrete red-flag patterns we want Haiku to recognise.
    assert "if x is none" in p or "x = default" in p
    assert "except: pass" in p or "except exception" in p
    assert "sleep" in p
    assert "type: ignore" in p or "noqa" in p
    # The output convention so the agent can spot this class of feedback.
    assert 'symptom fix:' in p or "symptom fix" in p
