"""Unit tests for extractor module using mocked responses."""

import json
from unittest.mock import MagicMock
import pytest
from pydantic import ValidationError

from extractor import (
    build_system_prompt,
    build_user_prompt,
    call_llm,
    extract_contact_info,
    parse_json_response,
    validate_extracted_dict,
)
from schemas import ContactInfo


def test_build_prompts():
    """System and user prompts should include necessary instructions and feedback."""
    sys_prompt = build_system_prompt()
    assert "name" in sys_prompt
    assert "email" in sys_prompt
    assert "null" in sys_prompt

    user_prompt_clean = build_user_prompt("My name is John.")
    assert "My name is John." in user_prompt_clean
    assert "ATTENTION - PREVIOUS ATTEMPT FAILED" not in user_prompt_clean

    user_prompt_with_feedback = build_user_prompt("My name is John.", feedback="Invalid email format")
    assert "My name is John." in user_prompt_with_feedback
    assert "ATTENTION - PREVIOUS ATTEMPT FAILED" in user_prompt_with_feedback
    assert "Invalid email format" in user_prompt_with_feedback


def test_parse_json_response_clean_json():
    """Clean JSON object string parses correctly."""
    raw = '{"name": "Alice", "email": "alice@example.com", "phone": null, "job_title": "Engineer"}'
    parsed = parse_json_response(raw)
    assert parsed["name"] == "Alice"
    assert parsed["email"] == "alice@example.com"
    assert parsed["phone"] is None
    assert parsed["job_title"] == "Engineer"


def test_parse_json_response_markdown_fenced():
    """Markdown code fences (```json ... ```) should be stripped safely."""
    raw = "```json\n{\"name\": \"Bob\", \"email\": null, \"phone\": \"555-1234\", \"job_title\": null}\n```"
    parsed = parse_json_response(raw)
    assert parsed["name"] == "Bob"
    assert parsed["phone"] == "555-1234"


def test_parse_json_response_with_surrounding_commentary():
    """JSON enclosed by conversational text should be safely isolated."""
    raw = "Here is the extracted contact info:\n{\"name\": \"Carol\", \"email\": null, \"phone\": null, \"job_title\": null}\nHope this helps!"
    parsed = parse_json_response(raw)
    assert parsed["name"] == "Carol"


def test_parse_json_response_malformed_json_raises_value_error():
    """Malformed JSON syntax must raise ValueError with clear message."""
    raw = '{"name": "Broken", "email": '
    with pytest.raises(ValueError) as exc_info:
        parse_json_response(raw)
    assert "Malformed JSON" in str(exc_info.value)


def test_parse_json_response_empty_raises_value_error():
    """Empty or whitespace response must raise ValueError."""
    with pytest.raises(ValueError):
        parse_json_response("")
    with pytest.raises(ValueError):
        parse_json_response("    ")


def test_parse_json_response_non_dict_raises_value_error():
    """JSON arrays or primitives must raise ValueError (expected object)."""
    with pytest.raises(ValueError) as exc_info:
        parse_json_response('["Alice", "alice@example.com"]')
    assert "Expected JSON object" in str(exc_info.value)


def test_validate_extracted_dict():
    """Valid dictionary converts to ContactInfo, invalid raises ValidationError."""
    valid_dict = {"name": "Dana", "email": "dana@company.org", "phone": None, "job_title": None}
    info = validate_extracted_dict(valid_dict)
    assert isinstance(info, ContactInfo)
    assert info.name == "Dana"

    invalid_dict = {"email": "not-an-email"}
    with pytest.raises(ValidationError):
        validate_extracted_dict(invalid_dict)


def test_call_llm_with_custom_caller():
    """call_llm properly routes through custom_caller function."""
    def mock_caller(sys_p, user_p):
        return '{"name": "Mocked User", "email": null, "phone": null, "job_title": null}'

    res = call_llm("test prompt", custom_caller=mock_caller)
    assert "Mocked User" in res


def test_call_llm_connection_error_handling():
    """Connection errors from Ollama client raise RuntimeError without crashing."""
    mock_client = MagicMock()
    mock_client.chat.side_effect = ConnectionError("Could not reach Ollama server")

    with pytest.raises(RuntimeError) as exc_info:
        call_llm("hello", client=mock_client)
    assert "Ollama model call failed" in str(exc_info.value)


def test_extract_contact_info_end_to_end_mocked():
    """Full extract_contact_info step with valid mocked response."""
    def mock_caller(sys_p, user_p):
        return json.dumps({
            "name": "Dr. Beatrice Dupont",
            "email": "bdupont@healthcare.fr",
            "phone": "06-12-34-56-78",
            "job_title": "Clinical Pharmacologist",
        })

    info, raw = extract_contact_info("Some doctor bio", custom_caller=mock_caller)
    assert info.name == "Dr. Beatrice Dupont"
    assert info.email == "bdupont@healthcare.fr"
    assert info.phone == "06-12-34-56-78"
    assert info.job_title == "Clinical Pharmacologist"
