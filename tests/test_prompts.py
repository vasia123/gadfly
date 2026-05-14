from gadfly import prompts


def test_build_user_message_includes_all_sections():
    msg = prompts.build_user_message(
        tool_name="Edit",
        tool_input={"file_path": "auth.py", "old_string": "pass", "new_string": "return None  # TODO"},
        tool_response={"success": True},
        last_user_request="implement validate_token properly",
        last_assistant_plan="I will edit auth.py",
        recent_actions=["Read(auth.py)", "Bash: pytest -q"],
    )
    assert "Most recent user instruction" in msg
    assert "implement validate_token properly" in msg
    assert "Last thing the agent said" in msg
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
        last_user_request=None,
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
