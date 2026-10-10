"""Evaluation module for structured contact extraction pipeline.

Calculates schema validity, field-level precision/recall/F1, exact record match,
no-target accuracy, retry rate, manual review rate, and processing latency.
Supports offline mock mode and live Ollama extraction.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dataset import DATASET_PATH, load_dataset
from schemas import ContactInfo, ExtractionResult, ExtractionStatus
from validator import ExtractionPipeline, process_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Default output locations
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
EVALUATION_JSON_PATH = DEFAULT_RESULTS_DIR / "evaluation.json"
PER_SAMPLE_CSV_PATH = DEFAULT_RESULTS_DIR / "per_sample_results.csv"

FIELDS = ["name", "email", "phone", "job_title"]


def normalize_value(field_name: str, val: Any) -> Optional[str]:
    """Conservative, documented normalization for string comparison.

    - Whitespace: Stripped and multiple spaces collapsed.
    - None / empty: Returns None.
    - email: Lowercased, stripped.
    - phone: Strips all formatting characters (spaces, dashes, parens, plus).
    - name & job_title: Lowercased, stripped, punctuation cleaned.
    """
    if val is None:
        return None

    s = str(val).strip()
    if not s:
        return None

    if field_name == "email":
        return s.lower()

    if field_name == "phone":
        # Keep only digits for phone comparison
        digits = re.sub(r"\D", "", s)
        return digits if digits else s.lower()

    # For name and job_title: normalize whitespace and lowercase
    s_clean = re.sub(r"\s+", " ", s).lower()
    return s_clean


def compute_field_metrics(
    true_positives: int, false_positives: int, false_negatives: int
) -> Tuple[float, float, float]:
    """Calculate precision, recall, and F1 with safe zero-division handling."""
    # Precision = TP / (TP + FP)
    prec_denom = true_positives + false_positives
    precision = (true_positives / prec_denom) if prec_denom > 0 else (1.0 if false_negatives == 0 else 0.0)

    # Recall = TP / (TP + FN)
    rec_denom = true_positives + false_negatives
    recall = (true_positives / rec_denom) if rec_denom > 0 else (1.0 if false_positives == 0 else 0.0)

    # F1 = 2 * (P * R) / (P + R)
    f1_denom = precision + recall
    f1 = (2 * precision * recall / f1_denom) if f1_denom > 0 else 0.0

    return round(precision, 4), round(recall, 4), round(f1, 4)


def mock_custom_caller_factory(samples: List[Dict[str, Any]]) -> Callable[[str, str], str]:
    """Create a deterministic mock LLM caller for offline test runs.

    Matches input text to labeled sample expected output.
    """
    sample_lookup = {s["input_text"].strip(): s["expected_output"] for s in samples}

    def _mock_caller(system_prompt: str, user_prompt: str) -> str:
        # Extract the input text from user prompt
        match = re.search(r'"""\s*([\s\S]*?)\s*"""', user_prompt)
        text = match.group(1).strip() if match else user_prompt.strip()

        # Find matching sample
        expected = sample_lookup.get(text)
        if expected is not None:
            return json.dumps(expected)

        # Fallback empty JSON object
        return json.dumps({f: None for f in FIELDS})

    return _mock_caller


class Evaluator:
    """Evaluates extraction performance against labeled ground truth."""

    def __init__(
        self,
        mode: str = "mock",
        dataset_file: Path = DATASET_PATH,
        results_dir: Path = DEFAULT_RESULTS_DIR,
        max_retries: int = 2,
    ) -> None:
        self.mode = mode.lower()
        self.dataset_file = dataset_file
        self.results_dir = results_dir
        self.max_retries = max_retries

    def run_evaluation(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Execute evaluation over dataset and calculate all performance metrics."""
        samples = load_dataset(self.dataset_file)
        if limit is not None and limit > 0:
            samples = samples[:limit]

        total_samples = len(samples)
        logger.info(
            "Starting evaluation: mode=%s, total_samples=%d, max_retries=%d",
            self.mode,
            total_samples,
            self.max_retries,
        )

        custom_caller = None
        if self.mode == "mock":
            custom_caller = mock_custom_caller_factory(samples)

        pipeline = ExtractionPipeline(
            max_retries=self.max_retries,
            custom_caller=custom_caller,
        )

        per_sample_results: List[Dict[str, Any]] = []

        # Metric accumulators
        successful_extractions = 0
        needs_review_count = 0
        model_error_count = 0
        retried_count = 0
        exact_record_matches = 0

        # No-target subset tracking
        no_target_total = 0
        no_target_correct = 0

        # Per-field TP, FP, FN accumulators
        field_tp = {f: 0 for f in FIELDS}
        field_fp = {f: 0 for f in FIELDS}
        field_fn = {f: 0 for f in FIELDS}

        latencies_ms: List[float] = []

        for idx, sample in enumerate(samples, start=1):
            sample_id = sample.get("sample_id", f"SAMPLE-{idx:03d}")
            category = sample.get("category", "unspecified")
            input_text = sample.get("input_text", "")
            expected = sample.get("expected_output", {})

            # Check if expected is all null
            is_expected_no_target = all(expected.get(f) is None for f in FIELDS)
            if is_expected_no_target:
                no_target_total += 1

            # Run extraction pipeline
            result: ExtractionResult = pipeline.extract_and_validate(input_text)
            latencies_ms.append(result.processing_time_ms)

            if result.attempts > 1:
                retried_count += 1

            if result.status == ExtractionStatus.SUCCESS:
                successful_extractions += 1
            elif result.status == ExtractionStatus.NEEDS_REVIEW:
                needs_review_count += 1
            elif result.status == ExtractionStatus.MODEL_ERROR:
                model_error_count += 1

            pred_dict = result.data.to_dict() if result.data else {f: None for f in FIELDS}

            # Field-by-field comparison
            sample_all_fields_match = (result.status == ExtractionStatus.SUCCESS)

            for field in FIELDS:
                exp_raw = expected.get(field)
                pred_raw = pred_dict.get(field)

                exp_norm = normalize_value(field, exp_raw)
                pred_norm = normalize_value(field, pred_raw)

                # Check exact field equivalence
                if exp_norm != pred_norm:
                    sample_all_fields_match = False

                # Evaluate non-null classification
                if exp_norm is not None and pred_norm is not None:
                    if exp_norm == pred_norm:
                        field_tp[field] += 1
                    else:
                        field_fp[field] += 1
                        field_fn[field] += 1
                elif exp_norm is not None and pred_norm is None:
                    field_fn[field] += 1
                elif exp_norm is None and pred_norm is not None:
                    field_fp[field] += 1
                # Both None: True Negative, not counted in non-null precision/recall

            if sample_all_fields_match:
                exact_record_matches += 1

            # No-target subset evaluation: all fields must be null and status SUCCESS
            if is_expected_no_target:
                pred_all_null = all(pred_dict.get(f) is None for f in FIELDS)
                if result.status == ExtractionStatus.SUCCESS and pred_all_null:
                    no_target_correct += 1

            # Record per-sample CSV row
            per_sample_results.append({
                "sample_id": sample_id,
                "category": category,
                "status": result.status.value,
                "attempts": result.attempts,
                "latency_ms": round(result.processing_time_ms, 2),
                "expected_name": expected.get("name"),
                "predicted_name": pred_dict.get("name"),
                "expected_email": expected.get("email"),
                "predicted_email": pred_dict.get("email"),
                "expected_phone": expected.get("phone"),
                "predicted_phone": pred_dict.get("phone"),
                "expected_job_title": expected.get("job_title"),
                "predicted_job_title": pred_dict.get("job_title"),
                "exact_match": sample_all_fields_match,
                "errors": "; ".join(result.errors) if result.errors else "",
            })

            if idx % 10 == 0 or idx == total_samples:
                logger.info("Processed %d/%d samples...", idx, total_samples)

        # Compute aggregate rates
        schema_validity_rate = round((successful_extractions / total_samples) * 100, 2) if total_samples > 0 else 0.0
        exact_record_accuracy = round((exact_record_matches / total_samples) * 100, 2) if total_samples > 0 else 0.0
        no_target_accuracy = round((no_target_correct / no_target_total) * 100, 2) if no_target_total > 0 else 0.0
        retry_rate = round((retried_count / total_samples) * 100, 2) if total_samples > 0 else 0.0
        manual_review_rate = round(((needs_review_count + model_error_count) / total_samples) * 100, 2) if total_samples > 0 else 0.0
        avg_latency_ms = round(sum(latencies_ms) / len(latencies_ms), 2) if latencies_ms else 0.0

        # Per-field and macro/micro precision, recall, F1
        per_field_metrics = {}
        total_tp = sum(field_tp.values())
        total_fp = sum(field_fp.values())
        total_fn = sum(field_fn.values())

        for f in FIELDS:
            p, r, f1 = compute_field_metrics(field_tp[f], field_fp[f], field_fn[f])
            per_field_metrics[f] = {
                "true_positives": field_tp[f],
                "false_positives": field_fp[f],
                "false_negatives": field_fn[f],
                "precision": p,
                "recall": r,
                "f1_score": f1,
            }

        overall_p, overall_r, overall_f1 = compute_field_metrics(total_tp, total_fp, total_fn)

        metrics_summary = {
            "evaluation_mode": self.mode,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_samples": total_samples,
            "successful_extractions": successful_extractions,
            "schema_validity_rate_pct": schema_validity_rate,
            "exact_record_accuracy_pct": exact_record_accuracy,
            "no_target_samples_count": no_target_total,
            "no_target_detection_accuracy_pct": no_target_accuracy,
            "retried_samples_count": retried_count,
            "retry_rate_pct": retry_rate,
            "manual_review_samples_count": needs_review_count + model_error_count,
            "manual_review_rate_pct": manual_review_rate,
            "average_latency_ms": avg_latency_ms,
            "overall_non_null_field_metrics": {
                "precision": overall_p,
                "recall": overall_r,
                "f1_score": overall_f1,
            },
            "per_field_metrics": per_field_metrics,
        }

        self._save_results(metrics_summary, per_sample_results)
        return metrics_summary

    def _save_results(
        self,
        metrics: Dict[str, Any],
        per_sample_results: List[Dict[str, Any]],
    ) -> None:
        """Write evaluation metrics to JSON and per-sample table to CSV."""
        self.results_dir.mkdir(parents=True, exist_ok=True)

        json_out = self.results_dir / "evaluation.json"
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info("Saved evaluation metrics to %s", json_out)

        csv_out = self.results_dir / "per_sample_results.csv"
        if per_sample_results:
            fieldnames = list(per_sample_results[0].keys())
            with open(csv_out, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(per_sample_results)
            logger.info("Saved per-sample results to %s", csv_out)


def print_summary_table(metrics: Dict[str, Any]) -> None:
    """Print clean terminal summary of evaluation results."""
    mode_tag = metrics.get("evaluation_mode", "mock").upper()
    print("\n" + "=" * 65)
    print(f"   EXTRACTION EVALUATION RESULTS [{mode_tag} MODE]")
    print("=" * 65)
    print(f"Total Samples Evaluated:      {metrics['total_samples']}")
    print(f"Successful Extractions:       {metrics['successful_extractions']}")
    print(f"Schema Validity Rate:         {metrics['schema_validity_rate_pct']}%")
    print(f"Exact-Record Accuracy:        {metrics['exact_record_accuracy_pct']}%")
    print(f"No-Target Detection Accuracy: {metrics['no_target_detection_accuracy_pct']}% ({metrics['no_target_samples_count']} samples)")
    print(f"Retry Rate:                   {metrics['retry_rate_pct']}% ({metrics['retried_samples_count']} samples)")
    print(f"Manual Review Rate:           {metrics['manual_review_rate_pct']}% ({metrics['manual_review_samples_count']} samples)")
    print(f"Average Latency:              {metrics['average_latency_ms']} ms")
    print("-" * 65)
    overall = metrics["overall_non_null_field_metrics"]
    print(f"OVERALL NON-NULL METRICS:     P: {overall['precision']} | R: {overall['recall']} | F1: {overall['f1_score']}")
    print("-" * 65)
    print(f"{'Field':<15}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}")
    print("-" * 65)
    for field, m in metrics["per_field_metrics"].items():
        print(f"{field:<15}{m['precision']:<12}{m['recall']:<12}{m['f1_score']:<12}")
    print("=" * 65 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate contact extraction pipeline.")
    parser.add_argument(
        "--mode",
        choices=["mock", "live"],
        default="mock",
        help="Evaluation mode: 'mock' (offline deterministic) or 'live' (local Ollama LLM).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATASET_PATH,
        help="Path to evaluation dataset JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory to save evaluation.json and per_sample_results.csv.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Max retries allowed per sample (default: 2).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of samples to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator = Evaluator(
        mode=args.mode,
        dataset_file=args.data,
        results_dir=args.output_dir,
        max_retries=args.retries,
    )
    metrics = evaluator.run_evaluation(limit=args.limit)
    print_summary_table(metrics)


if __name__ == "__main__":
    main()
