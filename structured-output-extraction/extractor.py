"""Extraction module interfacing with Ollama for structured contact extraction."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable, Optional, Tuple

import ollama
from pydantic import ValidationError

from schemas import ContactInfo

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_MODEL: str = "llama3.2"
DEFAULT_BASE_URL: str = "http://localhost:11434"
DEFAULT_TIMEOUT_SECONDS: float = 30.0


def get_model_name() -> str:
    """Return configured model name from environment or default."""
    return os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)


def get_base_url() -> str:
    """Return configured Ollama base URL from environment or default."""
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)


def get_ollama_client(base_url: Optional[str] = None) -> ollama.Client:
    """Create an Ollama client instance with the configured host."""
    host = base_url or get_base_url()
    return ollama.Client(host=host)


def check_ollama_availability(
    client: Optional[ollama.Client] = None,
    model_name: Optional[str] = None,
) -> Tuple[bool, str]:
    """Verify whether Ollama is reachable and the required model is available.

    Returns:
        (is_available, message): Status boolean and informative description.
    """
    model = model_name or get_model_name()
    c = client or get_ollama_client()
    try:
        models_response = c.list()
        # In ollama python library, models_response can be a ListResponse or dict
        available_models = []
        if hasattr(models_response, "models"):
            for m in models_response.models:
                name = getattr(m, "model", None) or getattr(m, "name", "")
                available_models.append(name)
        elif isinstance(models_response, dict) and "models" in models_response:
            for m in models_response["models"]:
                name = m.get("model") or m.get("name", "")
                available_models.append(name)

        # Check exact or prefix match (e.g. 'llama3.2' matches 'llama3.2:latest')
        model_found = any(
            m == model or m.startswith(f"{model}:") or model.startswith(f"{m}:")
            for m in available_models
        )

        if not model_found:
            return (
                False,
                f"Ollama is running, but model '{model}' was not found. "
                f"Available models: {', '.join(available_models) or 'none'}. "
                f"Run 'ollama pull {model}' to download it.",
            )

        return True, f"Ollama is running with model '{model}' available."

    except Exception as exc:
        return (
            False,
            f"Cannot connect to Ollama at {get_base_url()}. "
            f"Ensure Ollama is installed and running (`ollama serve`). Error: {exc}",
        )


def build_system_prompt() -> str:
    """Construct instructions for the model enforcing schema adherence and truthfulness."""
    return (
        "You are an expert structured data extraction engine specializing in contact information.\n"
        "Your task is to extract exactly four fields from the provided text:\n"
        "- name: (string or null) The full name of the person.\n"
        "- email: (string or null) A valid RFC-compliant email address.\n"
        "- phone: (string or null) The contact phone number string.\n"
        "- job_title: (string or null) The professional job title or role.\n\n"
        "STRICT EXTRACTION RULES:\n"
        "1. Extract ONLY information that is explicitly stated in the input text.\n"
        "2. If a field is not present, missing, or uncertain in the text, you MUST set its value to null.\n"
        "3. If the text does NOT contain any contact information (e.g. notes about the weather, unrelated text), "
        "set ALL four fields to null.\n"
        "4. NEVER invent, hallucinate, or assume missing values.\n"
        "5. Output MUST strictly conform to the JSON schema provided.\n"
        "6. Do not include extra fields or Markdown commentary outside the JSON object."
    )


def build_user_prompt(text: str, feedback: Optional[str] = None) -> str:
    """Build the extraction prompt, incorporating previous validation error feedback if retrying."""
    prompt_parts = [
        "Please extract contact information from the following text according to the schema.",
        f'"""\n{text}\n"""',
    ]

    if feedback:
        prompt_parts.append(
            "\nATTENTION - PREVIOUS ATTEMPT FAILED VALIDATION:\n"
            f"{feedback}\n"
            "Correct the issues identified above. Return a valid JSON object strictly complying with the schema."
        )

    return "\n".join(prompt_parts)


def parse_json_response(raw_text: str) -> dict[str, Any]:
    """Safely parse model output into a Python dictionary.

    Extracts JSON blocks and strips potential Markdown fences.
    Never uses eval().

    Raises:
        ValueError: If raw_text cannot be parsed into a valid JSON object dictionary.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Model returned an empty response.")

    text = raw_text.strip()

    # Strip markdown code blocks ```json ... ``` or ``` ... ```
    md_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if md_match:
        text = md_match.group(1).strip()

    # If the text still contains extra commentary around braces, extract the outer-most JSON object
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace : last_brace + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON returned by model: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected JSON object (dictionary), but received {type(data).__name__}."
        )

    return data


def validate_extracted_dict(data: dict[str, Any]) -> ContactInfo:
    """Validate dictionary against ContactInfo Pydantic schema.

    Raises:
        ValidationError: If schema constraints or email formats are violated.
    """
    return ContactInfo.model_validate(data)


def call_llm(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    client: Optional[ollama.Client] = None,
    custom_caller: Optional[Callable[[str, str], str]] = None,
) -> str:
    """Send request to Ollama and return the raw generated text response.

    Args:
        user_prompt: User instruction containing text to extract.
        system_prompt: System instructions enforcing schema.
        timeout: Request timeout in seconds.
        client: Optional custom Ollama client.
        custom_caller: Optional custom callable (used for mocking during tests).

    Raises:
        RuntimeError: On connection failures, timeouts, or unexpected API errors.
    """
    if custom_caller is not None:
        return custom_caller(system_prompt or build_system_prompt(), user_prompt)

    sys_p = system_prompt or build_system_prompt()
    model = get_model_name()
    c = client or get_ollama_client()

    messages = [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = c.chat(
            model=model,
            messages=messages,
            format=ContactInfo.model_json_schema(),
            options={"temperature": 0.0},  # Deterministic output
        )
        # Handle Ollama response object or dict
        if hasattr(response, "message"):
            content = getattr(response.message, "content", "")
        elif isinstance(response, dict) and "message" in response:
            content = response["message"].get("content", "")
        else:
            raise RuntimeError(f"Unexpected response format from Ollama: {response}")

        return content

    except Exception as exc:
        logger.error("Error communicating with Ollama: %s", exc)
        raise RuntimeError(f"Ollama model call failed: {exc}") from exc


def extract_contact_info(
    text: str,
    feedback: Optional[str] = None,
    client: Optional[ollama.Client] = None,
    custom_caller: Optional[Callable[[str, str], str]] = None,
) -> Tuple[ContactInfo, str]:
    """Execute single extraction step: prompt construction, LLM call, parsing, and validation.

    Returns:
        (contact_info, raw_response): Validated ContactInfo object and raw response string.

    Raises:
        RuntimeError: If communication with the model fails.
        ValueError: If JSON response is malformed.
        ValidationError: If response fails Pydantic schema validation.
    """
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(text, feedback=feedback)

    raw_response = call_llm(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        client=client,
        custom_caller=custom_caller,
    )

    parsed_dict = parse_json_response(raw_response)
    validated_info = validate_extracted_dict(parsed_dict)

    return validated_info, raw_response
