"""Corpus fixtures are cut from real transcripts, so they carry whatever the
agent happened to read — including config files with live API keys. On
2026-08-21 that put a working OpenRouter key and a Fireworks key into
committed fixtures. These tests guard the scrubber that now stands between a
transcript and the disk, and the fixtures themselves.
"""
import json
import pathlib

from gadfly.pairs import redact_secrets

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


# Samples are assembled at runtime rather than written out as literals: a
# secret-shaped string in a checked-in file is exactly what this module exists
# to prevent, and a scanner cannot tell a test sample from the real thing.
def _sample(prefix: str) -> str:
    return prefix + "b3n1gn" + "0" * 18


def test_redacts_known_provider_prefixes():
    for raw in [
        f'openrouter_api_key: "{_sample("sk-or-v1-")}"',
        f'fireworks_api_key: "{_sample("fw_")}"',
        f'GROQ="{_sample("gsk_")}"',
        f"Authorization: Bearer {_sample('ghp_')}",
    ]:
        assert "REDACTED" in redact_secrets(raw), raw


def test_leaves_ordinary_code_alone():
    for raw in [
        "password: postgres",
        "token = request.token",
        "tokenizer = Tokenizer(model)",
        "GADFLY_API_KEY_ENV=OPENROUTER_API_KEY",
        'password: "string"',
    ]:
        assert redact_secrets(raw) == raw, raw


def test_survives_json_serialised_transcripts():
    secret = _sample("fw_")
    case = {"session_context": {"tool_response": f'api_key: "{secret}"'}}
    cleaned = json.loads(redact_secrets(json.dumps(case)))
    assert secret not in json.dumps(cleaned)


def test_committed_fixtures_carry_no_credentials():
    """The fixtures on disk must stay clean — this is the check that would
    have caught the 2026-08-21 leak before it was committed."""
    import re

    prefixed = re.compile(
        r"\b(?:fw_|gsk_|sk-ant-|sk-or-v1-|sk-proj-|AIza|hf_|ghp_|github_pat_|AKIA)"
        r"[A-Za-z0-9_\-]{16,}"
    )
    offenders = []
    for path in FIXTURES.rglob("*.json"):
        text = path.read_text(errors="ignore")
        for hit in prefixed.findall(text):
            if "REDACTED" not in hit.upper():
                offenders.append(f"{path.name}: {hit[:10]}…")
    assert not offenders, f"credentials in fixtures: {offenders}"
