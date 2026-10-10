"""Unit tests for validator pipeline, retry behavior, and manual review routing."""

import json
from unittest.mock import MagicMock
import pytest

from schemas import ContactInfo, ExtractionStatus
from validator import ExtractionPipeline, process_text


def test_validator_success_all_fields():
    """Pipeline successfully validates extraction when all fields are present."""
    mock_payload = {
        "name": "Alexander Wright",
        "email": "a.wright@cloudnet.org",
        "phone": "+1 (555) 321-9876",
        "job_title": "Senior DevOps Engineer",
    }
    def mock_caller(sys_p, user_p):
        return json.dumps(mock_payload)

    pipeline = ExtractionPipeline(custom_caller=mock_caller)
    result = pipeline.extract_and_validate("Alexander Wright bio...")

    assert result.status == ExtractionStatus.SUCCESS
    assert result.is_success
    assert result.attempts == 1
    assert result.data is not None
    assert result.data.name == "Alexander Wright"
    assert result.data.email == "a.wright@cloudnet.org"
    assert len(result.errors) == 0


def test_validator_success_partial_fields():
    """Missing fields represented by null must succeed."""
    mock_payload = {
        "name": "John Smith",
        "email": "john.smith@company.com",
        "phone": None,
        "job_title": None,
    }
    def mock_caller(sys_p, user_p):
        return json.dumps(mock_payload)

    result = process_text("John Smith contact...", custom_caller=mock_caller)
    assert result.status == ExtractionStatus.SUCCESS
    assert result.data.name == "John Smith"
    assert result.data.email == "john.smith@company.com"
    assert result.data.phone is None
    assert result.data.job_title is None


def test_validator_success_all_fields_null():
    """Documents with no target fields must return SUCCESS with all nulls."""
    mock_payload = {
        "name": None,
        "email": None,
        "phone": None,
        "job_title": None,
    }
    def mock_caller(sys_p, user_p):
        return json.dumps(mock_payload)

    result = process_text("The weather is pleasant today.", custom_caller=mock_caller)
    assert result.status == ExtractionStatus.SUCCESS
    assert result.is_empty_target
    assert result.data.is_empty()
    assert result.attempts == 1


def test_validator_empty_and_whitespace_input():
    """Empty or whitespace-only input must be handled consistently without crashing."""
    result_empty = process_text("")
    assert result_empty.status == ExtractionStatus.SUCCESS
    assert result_empty.is_empty_target
    assert result_empty.attempts == 1

    result_spaces = process_text("   \n\t  ")
    assert result_spaces.status == ExtractionStatus.SUCCESS
    assert result_spaces.is_empty_target
    assert result_spaces.attempts == 1


def test_validator_retry_on_malformed_json_then_success():
    """Model returning malformed JSON on attempt 1, then valid JSON on attempt 2."""
    call_count = 0

    def mock_caller(sys_p, user_p):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return '{"name": "Broken JSON'  # syntax error
        return json.dumps({"name": "Recovered User", "email": None, "phone": None, "job_title": None})

    pipeline = ExtractionPipeline(max_retries=2, custom_caller=mock_caller)
    result = pipeline.extract_and_validate("Some text")

    assert result.status == ExtractionStatus.SUCCESS
    assert result.attempts == 2
    assert result.data.name == "Recovered User"
    assert len(result.errors) == 1
    assert "JSON parsing error" in result.errors[0]


def test_validator_retry_on_schema_validation_failure_then_success():
    """Model returning invalid email on attempt 1, corrected on attempt 2 after feedback."""
    call_count = 0

    def mock_caller(sys_p, user_p):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Invalid email triggers Pydantic ValidationError
            return json.dumps({"name": "Alice", "email": "invalid-email-address", "phone": None, "job_title": None})
        # Verifies feedback was included in prompt for attempt 2
        assert "ATTENTION - PREVIOUS ATTEMPT FAILED" in user_p
        return json.dumps({"name": "Alice", "email": "alice@valid.org", "phone": None, "job_title": None})

    pipeline = ExtractionPipeline(max_retries=2, custom_caller=mock_caller)
    result = pipeline.extract_and_validate("Alice text")

    assert result.status == ExtractionStatus.SUCCESS
    assert result.attempts == 2
    assert result.data.email == "alice@valid.org"
    assert len(result.errors) == 1
    assert "Schema validation failed" in result.errors[0]


def test_validator_exhausted_retries_routes_to_needs_review():
    """Repeated validation failures must exhaust retries and route to needs_review."""
    def always_invalid_caller(sys_p, user_p):
        return json.dumps({"name": "Bob", "email": "bad_email_format", "phone": None, "job_title": None})

    pipeline = ExtractionPipeline(max_retries=2, custom_caller=always_invalid_caller)
    result = pipeline.extract_and_validate("Bob text")

    assert result.status == ExtractionStatus.NEEDS_REVIEW
    assert not result.is_success
    assert result.data is None  # Invalid output must never be marked success or accepted
    assert result.attempts == 3  # 1 initial + 2 retries
    assert len(result.errors) == 3


def test_validator_model_connection_error_routes_to_model_error():
    """Fatal model error routes to MODEL_ERROR immediately."""
    mock_client = MagicMock()
    mock_client.chat.side_effect = ConnectionError("Connection refused by Ollama host")

    pipeline = ExtractionPipeline(client=mock_client)
    result = pipeline.extract_and_validate("Some text")

    assert result.status == ExtractionStatus.MODEL_ERROR
    assert not result.is_success
    assert result.data is None
    assert len(result.errors) == 1
    assert "Connection refused" in result.errors[0]


def test_validator_configurable_retries():
    """When max_retries is 0, exactly 1 attempt is performed before routing to review."""
    def fail_caller(sys_p, user_p):
        return "not even json"

    pipeline = ExtractionPipeline(max_retries=0, custom_caller=fail_caller)
    result = pipeline.extract_and_validate("Input text")

    assert result.status == ExtractionStatus.NEEDS_REVIEW
    assert result.attempts == 1
    assert len(result.errors) == 1
