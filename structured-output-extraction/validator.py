"""Validation, retry, and manual-review pipeline for structured extraction."""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import ollama
from pydantic import ValidationError

from extractor import (
    call_llm,
    extract_contact_info,
    parse_json_response,
    validate_extracted_dict,
)
from schemas import (
    ContactInfo,
    ExtractionAttempt,
    ExtractionResult,
    ExtractionStatus,
)

logger = logging.getLogger(__name__)

# Default bounded retry limit
DEFAULT_MAX_RETRIES: int = 2


class ExtractionPipeline:
    """End-to-end extraction pipeline with schema validation, retry loop, and routing to manual review."""

    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        client: Optional[ollama.Client] = None,
        custom_caller: Optional[Callable[[str, str], str]] = None,
    ) -> None:
        """Initialize pipeline with retry limit and optional client/caller overrides."""
        self.max_retries = max(0, max_retries)
        self.client = client
        self.custom_caller = custom_caller

    def extract_and_validate(self, text: str) -> ExtractionResult:
        """Execute extraction pipeline on input text with bounded retry and error feedback.

        Processing Flow:
        1. Handle empty / whitespace-only input: Treat as an empty document with no target fields.
           Returns status SUCCESS with all fields null, bypassing LLM call.
        2. Attempt extraction using LLM.
        3. Parse JSON safely without eval().
        4. Validate against Pydantic ContactInfo schema.
        5. If parsing or validation fails, record specific error details and retry up to max_retries.
           Retry prompts include explicit validation feedback.
        6. If LLM connection fails fatally, return MODEL_ERROR immediately.
        7. If retries are exhausted without valid output, return NEEDS_REVIEW.
        8. A valid extraction with all null fields is considered a SUCCESS, not an error.

        Returns:
            ExtractionResult containing status, data, attempt count, errors, and timing.
        """
        start_time = time.perf_counter()
        original_text = text

        # Step 1: Handle empty or whitespace-only input consistently
        if not text or not text.strip():
            logger.info("Empty or whitespace-only input detected; returning empty ContactInfo.")
            empty_info = ContactInfo()
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ExtractionResult(
                status=ExtractionStatus.SUCCESS,
                data=empty_info,
                original_text=original_text,
                errors=[],
                attempts=1,
                processing_time_ms=duration_ms,
                model_name="local_bypass_empty_input",
                raw_response='{"name": null, "email": null, "phone": null, "job_title": null}',
            )

        total_allowed_attempts = 1 + self.max_retries
        attempts_history: list[ExtractionAttempt] = []
        collected_errors: list[str] = []
        feedback: Optional[str] = None
        last_raw_response: Optional[str] = None

        for attempt_idx in range(1, total_allowed_attempts + 1):
            attempt_start = time.perf_counter()
            attempt_error: Optional[str] = None
            current_raw: Optional[str] = None

            try:
                # Steps 2-4: Call extractor (build prompt, invoke LLM, parse JSON, validate)
                validated_info, current_raw = extract_contact_info(
                    text=text,
                    feedback=feedback,
                    client=self.client,
                    custom_caller=self.custom_caller,
                )
                last_raw_response = current_raw
                attempt_duration = (time.perf_counter() - attempt_start) * 1000

                attempts_history.append(
                    ExtractionAttempt(
                        attempt_number=attempt_idx,
                        raw_output=current_raw,
                        duration_ms=attempt_duration,
                    )
                )

                # Schema validation succeeded
                total_duration = (time.perf_counter() - start_time) * 1000
                logger.info(
                    "Extraction succeeded on attempt %d/%d (took %.1fms). Empty target: %s",
                    attempt_idx,
                    total_allowed_attempts,
                    total_duration,
                    validated_info.is_empty(),
                )

                return ExtractionResult(
                    status=ExtractionStatus.SUCCESS,
                    data=validated_info,
                    original_text=original_text,
                    errors=collected_errors,
                    attempts=attempt_idx,
                    processing_time_ms=total_duration,
                    raw_response=current_raw,
                    history=attempts_history,
                )

            except RuntimeError as exc:
                # LLM connection or model communication failure
                attempt_duration = (time.perf_counter() - attempt_start) * 1000
                attempt_error = f"Model execution error: {exc}"
                collected_errors.append(attempt_error)
                logger.error("Attempt %d failed due to model error: %s", attempt_idx, exc)

                attempts_history.append(
                    ExtractionAttempt(
                        attempt_number=attempt_idx,
                        error_message=attempt_error,
                        duration_ms=attempt_duration,
                    )
                )

                total_duration = (time.perf_counter() - start_time) * 1000
                # Connection or engine failures are fatal model errors
                return ExtractionResult(
                    status=ExtractionStatus.MODEL_ERROR,
                    data=None,
                    original_text=original_text,
                    errors=collected_errors,
                    attempts=attempt_idx,
                    processing_time_ms=total_duration,
                    raw_response=current_raw,
                    history=attempts_history,
                )

            except (ValueError, ValidationError) as exc:
                # Parsing or schema validation failure
                attempt_duration = (time.perf_counter() - attempt_start) * 1000

                if isinstance(exc, ValidationError):
                    err_lines = []
                    for err in exc.errors():
                        loc = ".".join(str(l) for l in err.get("loc", []))
                        msg = err.get("msg", "Validation error")
                        inp = err.get("input", "")
                        err_lines.append(f"Field '{loc}': {msg} (received: {inp!r})")
                    err_desc = "Schema validation failed:\n" + "\n".join(err_lines)
                else:
                    err_desc = f"JSON parsing error: {exc}"

                collected_errors.append(f"Attempt {attempt_idx}: {err_desc}")
                logger.warning("Attempt %d validation failure: %s", attempt_idx, err_desc)

                attempts_history.append(
                    ExtractionAttempt(
                        attempt_number=attempt_idx,
                        raw_output=current_raw,
                        error_message=err_desc,
                        duration_ms=attempt_duration,
                    )
                )

                # Formulate feedback for next attempt
                feedback = (
                    f"Your response on attempt {attempt_idx} failed validation:\n"
                    f"{err_desc}\n"
                    "Ensure your output strictly adheres to the schema: "
                    "name (string or null), email (valid email string or null), phone (string or null), job_title (string or null)."
                )

        # All retries exhausted -> Route to manual review
        total_duration = (time.perf_counter() - start_time) * 1000
        logger.warning(
            "Extraction exhausted all %d attempts without valid schema adherence. Routing to manual review.",
            total_allowed_attempts,
        )

        return ExtractionResult(
            status=ExtractionStatus.NEEDS_REVIEW,
            data=None,
            original_text=original_text,
            errors=collected_errors,
            attempts=total_allowed_attempts,
            processing_time_ms=total_duration,
            raw_response=last_raw_response,
            history=attempts_history,
        )


def process_text(
    text: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    client: Optional[ollama.Client] = None,
    custom_caller: Optional[Callable[[str, str], str]] = None,
) -> ExtractionResult:
    """Convenience helper to extract and validate contact info with bounded retries."""
    pipeline = ExtractionPipeline(
        max_retries=max_retries,
        client=client,
        custom_caller=custom_caller,
    )
    return pipeline.extract_and_validate(text)
