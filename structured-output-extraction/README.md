# Structured Output Extraction Pipeline

A production-grade Python and LLM pipeline that extracts validated, typed structured data from messy, unstructured text using local Large Language Models (via [Ollama](https://ollama.com/)) and [Pydantic v2](https://docs.pydantic.dev/).

Every extracted record strictly adheres to a typed contact information schema (`ContactInfo`). Invalid or malformed model outputs are never accepted silently: they are retried with corrective schema feedback or routed to manual review.

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture & Pipeline Flow](#system-architecture--pipeline-flow)
- [Project Directory Structure](#project-directory-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
  - [1. Environment Setup (Windows PowerShell / CMD / Linux)](#1-environment-setup)
  - [2. Install Dependencies](#2-install-dependencies)
  - [3. Ollama & Local Model Setup](#3-ollama--local-model-setup)
- [Running the Streamlit UI](#running-the-streamlit-ui)
- [Running Automated Tests](#running-automated-tests)
- [Evaluation Dataset & Benchmarking](#evaluation-dataset--benchmarking)
  - [Generating the Dataset](#generating-the-dataset)
  - [Running Offline Mock Evaluation](#running-offline-mock-evaluation)
  - [Running Live LLM Evaluation](#running-live-llm-evaluation)
- [Typed Schema Details](#typed-schema-details)
- [Validation, Retry, and Manual Review Pipeline](#validation-retry-and-manual-review-pipeline)
- [Handling Documents with No Target Fields](#handling-documents-with-no-target-fields)
- [Evaluation Metrics & Mathematical Definitions](#evaluation-metrics--mathematical-definitions)
- [Troubleshooting](#troubleshooting)
- [Limitations & Future Enhancements](#limitations--future-enhancements)

---

## Overview

Unstructured documents—such as informal notes, email signatures, resume snippets, customer support messages, and invoices—often contain vital contact information mixed with irrelevant text. 

Traditional approaches either rely on brittle regular expressions or unconstrained LLM generation that produces hallucinated keys, invalid email formats, or unpredictable JSON.

This project implements an end-to-end extraction and validation pipeline:
1. **Local & Free:** Runs completely on your machine using Ollama (`llama3.2`). Zero cloud API keys or costs.
2. **Schema Enforcement:** Pydantic v2 defines and guarantees data shapes, email RFC compliance, and whitespace normalization.
3. **Resilient Retry Loop:** Model validation errors trigger an automated corrective feedback loop before routing persistently invalid responses to manual review.
4. **Honest Evaluation:** Provides both a deterministic offline evaluation mode and live LLM benchmark suite over 70 labeled test cases.

---

## Key Features

- **Strict Typed Schema:** Pydantic v2 `ContactInfo` model enforcing fields: `name`, `email` (RFC-validated via `EmailStr`), `phone`, and `job_title`. Extra fields are strictly forbidden (`extra="forbid"`).
- **Graceful Null Handling:** Fields absent from the input are serialized as `null` (`None`). The model never fabricates missing values.
- **Zero-Target Document Safety:** Inputs with no contact info (e.g. weather notes, system logs) succeed with all-null fields (`is_empty() == True`).
- **Bounded Retry with Feedback:** Configurable retries (default: 2 retries, 3 total attempts). Failed attempts pass validation error messages back to the model.
- **Manual Review Routing:** If validation fails across all retries, the output receives status `needs_review` and preserves debugging error logs. Model connectivity issues receive `model_error`.
- **Interactive Streamlit UI:** Live extraction interface with preset examples, status badges, field inspection cards, raw JSON viewer, and conditional download buttons.
- **Comprehensive Benchmark Suite:** 70 reproducible evaluation samples across 4 categories (`all_fields`, `partial_fields`, `no_target_fields`, `ambiguous`).

---

## System Architecture & Pipeline Flow

```
                      [ Unstructured Input Text ]
                                   │
                                   ▼
               ┌───────────────────────────────────────┐
               │    Is input empty or whitespace?      │
               └───────────────────┬───────────────────┘
                       YES         │         NO
         ┌─────────────────────────┘         ▼
         │                         ┌───────────────────┐
         │                         │ Construct Prompts │
         │                         │ (Include feedback │
         │                         │  if on retry)     │
         │                         └─────────┬─────────┘
         │                                   │
         │                                   ▼
         │                         ┌───────────────────┐
         │                         │ Call Ollama LLM   │◄────────────────┐
         │                         │ (llama3.2)        │                 │
         │                         └─────────┬─────────┘                 │
         │                                   │                           │
         │                                   ▼                           │
         │                         ┌───────────────────┐                 │
         │                         │ Safe JSON Parsing │                 │
         │                         │ (No eval())       │                 │
         │                         └─────────┬─────────┘                 │
         │                                   │                           │
         │                                   ▼                           │
         │                         ┌───────────────────┐                 │
         │                         │ Validate against  │                 │
         │                         │ Pydantic Schema   │                 │
         │                         └─────────┬─────────┘                 │
         │                                   │                           │
         │                        ┌──────────┴──────────┐                │
         │                    VALID                   INVALID            │
         │                        │                     │                │
         │                        │             Retries < Max?           │
         │                        │             ┌───────┴───────┐        │
         │                        │           YES               NO       │
         │                        │            │                 │       │
         │                        │      Build Feedback          │       │
         │                        │      and Loop Again          │       │
         │                        │            │                 │       │
         │                        │            └─────────────────┘       │
         ▼                        ▼                              ▼       │
┌──────────────────┐    ┌──────────────────┐           ┌──────────────────┐
│ Status: SUCCESS  │    │ Status: SUCCESS  │           │ Status:          │
│ All fields: null │    │ Validated Data   │           │ NEEDS_REVIEW     │
│ (Bypass LLM)     │    │ ContactInfo      │           │ Log error trace  │
└──────────────────┘    └──────────────────┘           └──────────────────┘
```

---

## Project Directory Structure

```text
structured-output-extraction/
├── app.py                      # Interactive Streamlit Web Application
├── extractor.py                # Ollama communication, prompt building, and safe JSON parsing
├── schemas.py                  # Pydantic v2 schemas: ContactInfo, ExtractionResult, ExtractionStatus
├── validator.py                # Validation pipeline, retry loop with feedback, and review routing
├── evaluate.py                 # Evaluation benchmark runner supporting mock and live LLM modes
├── dataset.py                  # Evaluation dataset generator (70 labeled samples)
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation
├── .gitignore                  # Git exclusions for venv, caches, and test artifacts
├── data/
│   └── evaluation.jsonl        # 70 labeled evaluation samples in JSONL format
├── results/
│   ├── .gitkeep                # Git tracking marker
│   ├── evaluation.json         # Evaluation metrics summary (generated by evaluate.py)
│   └── per_sample_results.csv  # Detailed per-sample evaluation breakdown (generated)
└── tests/
    ├── conftest.py             # Pytest root configuration
    ├── test_schemas.py         # Schema validation, email checks, extra forbid tests
    ├── test_extractor.py       # Prompt construction, safe parsing, and mock LLM tests
    ├── test_validator.py       # Retry loops, error feedback, empty text, review routing
    └── test_evaluation.py      # Metric calculations, zero-division, dataset integrity tests
```

---

## Prerequisites

- **Python:** 3.11 or higher (Python 3.11, 3.12, 3.13, 3.14 supported).
- **Ollama:** Installed locally from [https://ollama.com](https://ollama.com).
- **Model:** `llama3.2` downloaded in Ollama (`ollama pull llama3.2`).

> **Offline Note:** If Ollama is not installed or running, the pipeline, test suite, and evaluation utilities still support full offline execution using deterministic mock responses.

---

## Installation & Setup

### 1. Environment Setup

#### Windows (PowerShell)
```powershell
# Navigate into project directory
cd structured-output-extraction

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

#### Windows (Command Prompt)
```cmd
cd structured-output-extraction
python -m venv .venv
.\.venv\Scripts\activate.bat
```

#### macOS / Linux
```bash
cd structured-output-extraction
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Ollama & Local Model Setup
Ensure the Ollama background daemon is running, then pull the default model:
```bash
# Pull the default llama3.2 model (approx 2.0 GB)
ollama pull llama3.2
```

To configure a custom model or Ollama host, set environment variables:
```bash
# Optional configuration
set OLLAMA_MODEL=llama3.2
set OLLAMA_BASE_URL=http://localhost:11434
```

---

## Running the Streamlit UI

Launch the interactive web application:
```bash
streamlit run app.py
```
This opens `http://localhost:8501` in your browser.

### Key UI Features:
1. **Ollama Health Indicator:** Real-time badge in the sidebar verifying connectivity to Ollama and presence of `llama3.2`.
2. **Preset Examples:** One-click dropdown loading realistic samples (all fields, partial, weather notes, messy resumes).
3. **Configurable Retries:** Adjust maximum allowed retry attempts dynamically (0 to 5).
4. **Offline Mock Toggle:** Enables testing the interface even when Ollama is unavailable.
5. **Field Cards:** Visual status for each field (`Extracted` vs `null / absent`).
6. **Safe Download Button:** Only enables downloading when output is valid JSON; disabled for failed extractions.

---

## Running Automated Tests

The test suite runs completely offline without requiring internet access or a running Ollama server:

```bash
# Run pytest quietly
pytest -q

# Run pytest with detailed verbosity
pytest -v
```

### Test Coverage Highlights:
- `test_schemas.py`: Schema instantiation, email RFC validity, null conversion for placeholder tokens (`null`, `none`, `N/A`), whitespace collapsing, rejection of unexpected fields (`extra="forbid"`).
- `test_extractor.py`: Prompt synthesis, feedback integration, safe JSON extraction (markdown fences, conversational prefixes), malformed JSON error catching, non-dict validation, no `eval()`.
- `test_validator.py`: Successful extraction, missing fields, all-null records, empty/whitespace document handling, retry on malformed JSON, retry on validation failure with feedback, exhaustion of retries routing to `needs_review`, model connection errors routing to `model_error`.
- `test_evaluation.py`: Mathematical precision/recall/F1 verification, zero-division resilience, string normalization rules, dataset size (>=60 samples) and uniqueness check, end-to-end evaluator execution.

---

## Evaluation Dataset & Benchmarking

### Generating the Dataset
The dataset generator produces `data/evaluation.jsonl` containing 70 reproducible, hand-crafted labeled samples:
```bash
python dataset.py
```

### Dataset Categories:
| Category | Sample Count | Description |
| :--- | :--- | :--- |
| `all_fields` | 16 | All 4 fields (`name`, `email`, `phone`, `job_title`) clearly present. |
| `partial_fields` | 24 | Subsets of 1, 2, or 3 fields present; absent fields strictly `null`. |
| `no_target_fields` | 15 | Text with zero contact targets (weather, recipes, system logs, notes). |
| `ambiguous` | 15 | Distractors: order IDs, version strings, dates, Twitter handles, URLs. |
| **Total** | **70** | **Comprehensive benchmark** |

---

### Running Offline Mock Evaluation
Runs fast, deterministic evaluation against the ground truth without invoking the LLM:
```bash
python evaluate.py --mode mock
```
Outputs:
- Terminal summary table
- `results/evaluation.json` (Structured metrics)
- `results/per_sample_results.csv` (Per-sample comparison table)

---

### Running Live LLM Evaluation
Runs live inference using Ollama and the local `llama3.2` model:
```bash
# Evaluate entire dataset
python evaluate.py --mode live

# Or evaluate a quick subset (e.g. 10 samples)
python evaluate.py --mode live --limit 10
```

---

## Typed Schema Details

The schema is defined in [schemas.py](file:///d:/Projects/Virtual%20Internship/Structured-Output-Extraction-Pipeline/structured-output-extraction/schemas.py) using Pydantic v2:

```python
class ContactInfo(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    name: Optional[str] = Field(default=None, description="Full personal name.")
    email: Optional[EmailStr] = Field(default=None, description="Validated RFC email.")
    phone: Optional[str] = Field(default=None, description="Phone number string.")
    job_title: Optional[str] = Field(default=None, description="Professional title.")
```

### Normalization and Cleansing Rules:
- **Whitespace Collapsing:** Strings like `"   Jane    Doe  "` are normalized to `"Jane Doe"`.
- **Placeholder Nullification:** Values like `"null"`, `"none"`, `"N/A"`, `"unknown"`, and `""` are automatically coerced to Python `None` (JSON `null`).
- **Numeric Phone Conversion:** Numbers emitted by LLMs (e.g. `18005551234`) are safely converted to string without failing validation.
- **Extra Field Rejection:** Any unexpected key (e.g. `{"company": "Acme"}`) raises a `ValidationError` per `extra="forbid"`.

---

## Validation, Retry, and Manual Review Pipeline

### Execution Flow:
1. **Empty Input Guard:** If input text is empty or whitespace-only, the pipeline immediately returns `ExtractionStatus.SUCCESS` with all fields `None`, avoiding unnecessary LLM latency.
2. **Initial Attempt:** Ollama is invoked with JSON Schema formatting and temperature 0.0.
3. **Safe Parsing:** Markdown fences (` ```json `) and conversational wrappers are stripped. Output is parsed via `json.loads()`. **`eval()` is never used.**
4. **Schema Validation:** Parsed dictionary is validated against `ContactInfo`.
5. **Bounded Retry Loop:**
   - If a `ValueError` (malformed JSON) or `ValidationError` (e.g. bad email) occurs, the error details are recorded.
   - If `attempts <= max_retries`, a follow-up prompt is sent to the model containing exact validation feedback (e.g. `Field 'email': value is not a valid email address`).
6. **Manual Review Routing:**
   - If retries are exhausted without a valid response, status is set to `ExtractionStatus.NEEDS_REVIEW`.
   - The original text, attempt count, and complete error history are preserved for audit.
   - If model connection fails catastrophically, status is set to `ExtractionStatus.MODEL_ERROR`.
   - **Invalid or unvalidated data is never returned with a success status.**

---

## Handling Documents with No Target Fields

A core design requirement is correctly processing documents that contain no contact information.

### Example Input:
```text
The weather is pleasant today. I went for a walk in the park.
```

### Expected & Actual Output:
```json
{
  "name": null,
  "email": null,
  "phone": null,
  "job_title": null
}
```

- Status: `success`
- The system does not invent names, phone numbers, or titles.
- The evaluation suite specifically isolates this subset and measures `no_target_detection_accuracy_pct`.

---

## Evaluation Metrics & Mathematical Definitions

The evaluator reports the following metrics, handling zero denominators safely:

### 1. Schema Validity Rate
$$\text{Schema Validity Rate} = \left(\frac{N_{\text{valid}}}{N_{\text{total}}}\right) \times 100$$
Where $N_{\text{valid}}$ is the number of extractions concluding with status `success`.

### 2. Exact-Record Accuracy
$$\text{Exact-Record Accuracy} = \left(\frac{N_{\text{exact}}}{N_{\text{total}}}\right) \times 100$$
Where $N_{\text{exact}}$ is the number of samples where all 4 extracted fields match the normalized ground truth exactly and extraction status is `success`.

### 3. Non-Null Field-Level Precision, Recall, and F1
For each field $f \in \{\text{name}, \text{email}, \text{phone}, \text{job\_title}\}$:
- **True Positive ($TP$):** Expected is non-null, predicted is non-null, and normalized values match.
- **False Positive ($FP$):** Predicted is non-null, but expected is null OR normalized values do not match.
- **False Negative ($FN$):** Expected is non-null, but predicted is null OR normalized values do not match.
- **True Negative ($TN$):** Both expected and predicted are `null` (not counted in non-null precision/recall).

$$\text{Precision} = \frac{TP}{TP + FP} \quad (\text{if } TP + FP = 0 \implies 1.0 \text{ if } FN = 0 \text{ else } 0.0)$$
$$\text{Recall} = \frac{TP}{TP + FN} \quad (\text{if } TP + FN = 0 \implies 1.0 \text{ if } FP = 0 \text{ else } 0.0)$$
$$F_1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \quad (\text{if } P + R = 0 \implies 0.0)$$

### 4. No-Target Detection Accuracy
$$\text{No-Target Accuracy} = \left(\frac{N_{\text{no\_target\_correct}}}{N_{\text{no\_target\_total}}}\right) \times 100$$
Evaluated exclusively on samples where ground-truth has all 4 fields `null`. Correctness requires all 4 predicted fields to be `null` **and** extraction status to be `success`.

### 5. Retry Rate & Manual Review Rate
$$\text{Retry Rate} = \left(\frac{N_{\text{attempts} > 1}}{N_{\text{total}}}\right) \times 100$$
$$\text{Manual Review Rate} = \left(\frac{N_{\text{needs\_review}} + N_{\text{model\_error}}}{N_{\text{total}}}\right) \times 100$$

---

## Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `Cannot connect to Ollama at http://localhost:11434` | Ollama service is not running | Run `ollama serve` in a terminal or launch Ollama app. |
| `Model 'llama3.2' was not found` | The model has not been downloaded | Run `ollama pull llama3.2`. |
| `ModuleNotFoundError: No module named 'schemas'` | Running python without directory in `PYTHONPATH` | Run commands from inside `structured-output-extraction/` or set `PYTHONPATH=.`. |
| Port 8501 already in use | Previous Streamlit instance running | Run `streamlit run app.py --server.port 8502`. |
| PowerShell Execution Policy error on venv activate | Script execution restricted | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before activating. |

---

## Limitations & Future Enhancements

- **International Phone Standardization:** While the current normalizer removes non-digits, integrating Google's `phonenumbers` library would enable E.164 standardization and country-code inference.
- **Multi-Contact Extraction:** The current schema targets single-contact entities. Future iterations can support `List[ContactInfo]` for documents referencing multiple individuals.
- **Asynchronous Batch Processing:** For large corporate datasets, implementing an `asyncio` batch worker pool would increase processing throughput across multi-core systems.
- **Active Learning Feedback:** Storing cases routed to `needs_review` in a human-in-the-loop review database to periodically fine-tune or evaluate model performance.
