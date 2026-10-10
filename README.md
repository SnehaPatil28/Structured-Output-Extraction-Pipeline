# Structured Output Extraction Pipeline

Turn messy, unstructured text into validated structured contact data using local LLMs (Ollama `llama3.2`) and Pydantic v2 schemas.

This project was built for the **AgenticX Internship Assessment**.

The complete project implementation and documentation reside in the [`structured-output-extraction`](./structured-output-extraction) directory.

---

## Quick Start

### 1. Run Automated Tests
```bash
# From workspace root:
pytest -v

# Or from inside the project directory:
cd structured-output-extraction
pytest -q
```

### 2. Generate Evaluation Dataset
```bash
cd structured-output-extraction
python dataset.py
```

### 3. Run Offline Mock Evaluation
```bash
cd structured-output-extraction
python evaluate.py --mode mock
```

### 4. Run Live LLM Evaluation
```bash
cd structured-output-extraction
python evaluate.py --mode live
```

### 5. Launch the Streamlit Web Application
```bash
cd structured-output-extraction
streamlit run app.py
```

---

## Project Structure
```text
structured-output-extraction/
├── app.py                      # Streamlit interactive user interface
├── extractor.py                # Ollama integration, prompt synthesis, safe parsing
├── schemas.py                  # Pydantic v2 schemas (ContactInfo, ExtractionResult)
├── validator.py                # Validation pipeline, retry loop with feedback, review routing
├── evaluate.py                 # Evaluation benchmark runner (mock & live LLM modes)
├── dataset.py                  # Benchmark dataset generator (70 labeled samples)
├── requirements.txt            # Python dependencies
├── README.md                   # Full documentation & metric formulas
├── .gitignore                  # Git exclusions
├── data/
│   └── evaluation.jsonl        # 70 labeled test cases
├── results/
│   ├── .gitkeep
│   ├── evaluation.json         # Evaluation results summary
│   └── per_sample_results.csv  # Detailed per-sample outputs
└── tests/
    ├── conftest.py
    ├── test_schemas.py         # Schema tests
    ├── test_extractor.py       # Extractor tests
    ├── test_validator.py       # Validator tests
    └── test_evaluation.py      # Evaluator tests
```

For full architecture details, schema definitions, mathematical metrics, and troubleshooting, please read [structured-output-extraction/README.md](./structured-output-extraction/README.md).
