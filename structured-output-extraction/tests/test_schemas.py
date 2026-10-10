"""Unit tests for Pydantic v2 ContactInfo schema and ExtractionResult."""

import json
import pytest
from pydantic import ValidationError

from schemas import (
    ContactInfo,
    ExtractionAttempt,
    ExtractionResult,
    ExtractionStatus,
)


def test_contact_info_all_fields_valid():
    """All 4 contact fields populated with valid data."""
    info = ContactInfo(
        name="Jane Doe",
        email="jane.doe@example.com",
        phone="+1 (555) 019-2834",
        job_title="Software Architect",
    )
    assert info.name == "Jane Doe"
    assert info.email == "jane.doe@example.com"
    assert info.phone == "+1 (555) 019-2834"
    assert info.job_title == "Software Architect"
    assert not info.is_empty()
    assert len(info.non_null_fields()) == 4


def test_contact_info_missing_fields_default_to_none():
    """Missing fields must default to None and serialize to JSON null."""
    info = ContactInfo(name="Bob Smith")
    assert info.name == "Bob Smith"
    assert info.email is None
    assert info.phone is None
    assert info.job_title is None

    dumped = info.to_dict()
    assert dumped["email"] is None
    assert dumped["phone"] is None
    assert dumped["job_title"] is None
    assert dumped["name"] == "Bob Smith"


def test_contact_info_all_fields_none():
    """A ContactInfo with all fields set to None is completely valid."""
    info = ContactInfo()
    assert info.is_empty()
    assert info.name is None
    assert info.email is None
    assert info.phone is None
    assert info.job_title is None
    assert len(info.non_null_fields()) == 0

    json_str = json.dumps(info.to_dict())
    assert json_str == '{"name": null, "email": null, "phone": null, "job_title": null}'


def test_contact_info_whitespace_normalization():
    """Whitespace should be trimmed and internal multiple spaces collapsed."""
    info = ContactInfo(
        name="   Alice    Marie   Johnson   ",
        email="  alice@example.com  ",
        phone="   +1  555   1234   ",
        job_title="  Senior   Data   Scientist  ",
    )
    assert info.name == "Alice Marie Johnson"
    assert info.email == "alice@example.com"
    assert info.phone == "+1 555 1234"
    assert info.job_title == "Senior Data Scientist"


def test_contact_info_null_placeholders_converted_to_none():
    """String placeholders like 'null', 'none', 'n/a', '' should convert to None."""
    info = ContactInfo(
        name="null",
        email="N/A",
        phone="none",
        job_title="",
    )
    assert info.name is None
    assert info.email is None
    assert info.phone is None
    assert info.job_title is None
    assert info.is_empty()


def test_contact_info_numeric_phone_converted_to_string():
    """Numeric phone values in JSON (int/float) must be converted safely to string."""
    info = ContactInfo(phone=18005551234)
    assert info.phone == "18005551234"
    assert isinstance(info.phone, str)


def test_contact_info_invalid_email_rejected():
    """Invalid email formats must raise Pydantic ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        ContactInfo(email="not-an-email")
    assert "email" in str(exc_info.value)

    with pytest.raises(ValidationError):
        ContactInfo(email="user@")

    with pytest.raises(ValidationError):
        ContactInfo(email="@domain.com")


def test_contact_info_unexpected_fields_rejected():
    """Unexpected extra fields must be rejected per extra='forbid'."""
    with pytest.raises(ValidationError) as exc_info:
        ContactInfo.model_validate({"name": "Eve", "unexpected_field": "disallowed"})
    assert "extra_forbidden" in str(exc_info.value) or "unexpected_field" in str(exc_info.value)


def test_extraction_result_properties():
    """Verify ExtractionResult helper properties and serialization."""
    res_success = ExtractionResult(
        status=ExtractionStatus.SUCCESS,
        data=ContactInfo(name="Sam"),
        original_text="I am Sam",
        attempts=1,
    )
    assert res_success.is_success
    assert not res_success.is_empty_target
    assert res_success.to_dict()["status"] == "success"

    res_empty_target = ExtractionResult(
        status=ExtractionStatus.SUCCESS,
        data=ContactInfo(),
        original_text="Nice weather today",
        attempts=1,
    )
    assert res_empty_target.is_success
    assert res_empty_target.is_empty_target

    res_review = ExtractionResult(
        status=ExtractionStatus.NEEDS_REVIEW,
        data=None,
        original_text="malformed input",
        errors=["Invalid schema"],
        attempts=3,
    )
    assert not res_review.is_success
    assert not res_review.is_empty_target
    assert res_review.to_dict()["data"] is None
