"""Unit tests for evaluation metrics, dataset validation, and benchmark execution."""

import json
from pathlib import Path
import pytest

from dataset import DATASET_PATH, load_dataset, save_dataset
from evaluate import (
    Evaluator,
    compute_field_metrics,
    normalize_value,
)
from schemas import ContactInfo


def test_compute_field_metrics_normal():
    """Verify standard precision, recall, F1 computation."""
    # TP=8, FP=2, FN=0 -> Prec = 8/10 = 0.8, Rec = 8/8 = 1.0, F1 = 2*0.8*1.0 / 1.8 = 0.8889
    p, r, f1 = compute_field_metrics(8, 2, 0)
    assert p == 0.8
    assert r == 1.0
    assert f1 == 0.8889


def test_compute_field_metrics_zero_division_safety():
    """Verify zero division returns 0.0 or 1.0 safely without crashing."""
    # Denominators are 0
    p, r, f1 = compute_field_metrics(0, 0, 0)
    assert p == 1.0
    assert r == 1.0
    assert f1 == 1.0

    # TP=0, FP=5, FN=0 -> Prec = 0/5 = 0.0
    p2, r2, f2 = compute_field_metrics(0, 5, 0)
    assert p2 == 0.0
    assert f2 == 0.0


def test_normalize_value_rules():
    """Verify normalization for email, phone, name, and job_title."""
    # Email: lowercased and stripped
    assert normalize_value("email", "  John.DOE@Example.COM ") == "john.doe@example.com"
    assert normalize_value("email", None) is None
    assert normalize_value("email", "") is None

    # Phone: non-digits stripped
    assert normalize_value("phone", "+1 (555) 321-9876") == "15553219876"
    assert normalize_value("phone", "555-0199") == "5550199"

    # Name and job title: lowercase, collapsed whitespace
    assert normalize_value("name", "  Dr.   Jane    Doe  ") == "dr. jane doe"
    assert normalize_value("job_title", "Senior   Data Analyst ") == "senior data analyst"


def test_dataset_size_and_uniqueness():
    """Evaluation dataset must contain at least 60 samples with unique IDs."""
    samples = load_dataset(DATASET_PATH)
    assert len(samples) >= 60, f"Expected at least 60 samples, got {len(samples)}"

    sample_ids = [s["sample_id"] for s in samples]
    assert len(sample_ids) == len(set(sample_ids)), "Duplicate sample_id found in dataset"

    # Check categories coverage
    categories = {s["category"] for s in samples}
    expected_categories = {"all_fields", "partial_fields", "no_target_fields", "ambiguous"}
    assert expected_categories.issubset(categories), f"Missing categories: {expected_categories - categories}"


def test_dataset_ground_truth_schema_conformance():
    """Every ground truth sample must conform to the ContactInfo schema."""
    samples = load_dataset(DATASET_PATH)
    for sample in samples:
        expected = sample["expected_output"]
        # Must validate against ContactInfo without raising ValidationError
        info = ContactInfo.model_validate(expected)
        assert isinstance(info, ContactInfo)


def test_evaluator_mock_mode_end_to_end(tmp_path):
    """Evaluator mock mode runs end-to-end, computing metrics and generating output files."""
    results_dir = tmp_path / "results"
    evaluator = Evaluator(
        mode="mock",
        dataset_file=DATASET_PATH,
        results_dir=results_dir,
    )
    # Evaluate a subset of 10 samples
    metrics = evaluator.run_evaluation(limit=10)

    assert metrics["evaluation_mode"] == "mock"
    assert metrics["total_samples"] == 10
    assert metrics["successful_extractions"] == 10
    assert metrics["schema_validity_rate_pct"] == 100.0
    assert (results_dir / "evaluation.json").exists()
    assert (results_dir / "per_sample_results.csv").exists()

    # Verify JSON content
    with open(results_dir / "evaluation.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["total_samples"] == 10
