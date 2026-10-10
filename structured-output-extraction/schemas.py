"""Pydantic v2 schemas for structured contact extraction pipeline."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class ContactInfo(BaseModel):
    """Schema representing extracted contact information.

    Fields absent from the source text are represented as None (serialized as JSON null).
    Strict schema validation rejects extraneous fields and validates email formats.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    name: Optional[str] = Field(
        default=None,
        description="Full personal name of the contact. Null if absent.",
    )
    email: Optional[EmailStr] = Field(
        default=None,
        description="Validated RFC-compliant email address. Null if absent.",
    )
    phone: Optional[str] = Field(
        default=None,
        description="Phone number string. Null if absent.",
    )
    job_title: Optional[str] = Field(
        default=None,
        description="Professional title or organizational role. Null if absent.",
    )

    @field_validator("name", "email", "phone", "job_title", mode="before")
    @classmethod
    def normalize_and_filter_empty(cls, value: Any) -> Any:
        """Normalize string fields, collapsing whitespace and mapping placeholder words to None."""
        if value is None:
            return None

        # Convert numeric values (e.g. phone numbers written as raw numbers) to string
        if isinstance(value, (int, float)):
            value = str(value)

        if isinstance(value, str):
            # Normalize whitespace
            cleaned = re.sub(r"\s+", " ", value).strip()
            # If string is empty or common LLM placeholder for null, return None
            null_placeholders = {
                "",
                "null",
                "none",
                "n/a",
                "na",
                "nil",
                "unknown",
                "not provided",
                "not specified",
                "undefined",
            }
            if cleaned.lower() in null_placeholders:
                return None
            return cleaned

        return value

    def is_empty(self) -> bool:
        """Return True if all four contact fields are None."""
        return (
            self.name is None
            and self.email is None
            and self.phone is None
            and self.job_title is None
        )

    def non_null_fields(self) -> dict[str, Any]:
        """Return a dictionary containing only non-None extracted fields."""
        return {k: v for k, v in self.model_dump().items() if v is not None}

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable dictionary with nulls for missing values."""
        return self.model_dump(mode="json")


class ExtractionStatus(str, Enum):
    """Execution status for extraction pipeline results."""

    SUCCESS = "success"
    NEEDS_REVIEW = "needs_review"
    MODEL_ERROR = "model_error"


class ExtractionAttempt(BaseModel):
    """Record of a single extraction attempt during retry cycle."""

    attempt_number: int
    prompt: Optional[str] = None
    raw_output: Optional[str] = None
    error_message: Optional[str] = None
    duration_ms: float = 0.0


class ExtractionResult(BaseModel):
    """Complete structured result returned by the validation pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: ExtractionStatus
    data: Optional[ContactInfo] = None
    original_text: str
    errors: list[str] = Field(default_factory=list)
    attempts: int = 1
    processing_time_ms: float = 0.0
    model_name: Optional[str] = None
    raw_response: Optional[str] = None
    history: list[ExtractionAttempt] = Field(default_factory=list)

    @property
    def is_success(self) -> bool:
        """Return True if extraction concluded with SUCCESS status."""
        return self.status == ExtractionStatus.SUCCESS

    @property
    def is_empty_target(self) -> bool:
        """Return True if extraction succeeded and all extracted fields are null."""
        return self.is_success and self.data is not None and self.data.is_empty()

    def to_dict(self) -> dict[str, Any]:
        """Serialize extraction result to a standard dictionary."""
        return {
            "status": self.status.value,
            "data": self.data.to_dict() if self.data else None,
            "original_text": self.original_text,
            "errors": self.errors,
            "attempts": self.attempts,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "model_name": self.model_name,
            "raw_response": self.raw_response,
        }
