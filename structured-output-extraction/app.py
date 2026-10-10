"""Streamlit web interface for the Structured Output Extraction Pipeline."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st
from extractor import check_ollama_availability, get_base_url, get_model_name
from schemas import ContactInfo, ExtractionResult, ExtractionStatus
from validator import DEFAULT_MAX_RETRIES, ExtractionPipeline

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Contact Extraction Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Dark theme CSS  – every element explicitly styled so nothing is invisible
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ===== BASE / BACKGROUND ===== */
    html, body, [class*="stApp"] {
        background-color: #0F1117 !important;
        color: #E2E8F0 !important;
    }
    .main .block-container {
        background-color: #0F1117 !important;
        padding-top: 1.5rem;
    }

    /* ===== TYPOGRAPHY ===== */
    h1,h2,h3,h4,h5,h6,p,li,label,span,div {
        color: #E2E8F0 !important;
    }

    /* ===== HEADER ===== */
    .main-header {
        font-size: 2rem;
        font-weight: 800;
        color: #A78BFA !important;
        margin-bottom: 0;
        line-height: 1.2;
    }
    .sub-header {
        font-size: 0.93rem;
        color: #94A3B8 !important;
        margin-bottom: 1rem;
    }

    /* ===== STATUS BADGES ===== */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 0.28rem 0.85rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .badge-success { background: #14532D; color: #86EFAC !important; border: 1px solid #22C55E; }
    .badge-review  { background: #78350F; color: #FCD34D !important; border: 1px solid #F59E0B; }
    .badge-error   { background: #7F1D1D; color: #FCA5A5 !important; border: 1px solid #EF4444; }

    /* ===== CARDS ===== */
    .dark-card {
        background: #1E2130;
        border: 1px solid #2D3748;
        border-radius: 0.75rem;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.75rem;
    }

    /* ===== FIELD CARDS ===== */
    .field-card {
        padding: 0.65rem 1rem;
        border-radius: 0.55rem;
        border: 1px solid #2D3748;
        background: #1A1F2E;
        margin-bottom: 0.4rem;
        font-size: 0.87rem;
        color: #E2E8F0 !important;
        transition: border-color 0.15s;
    }
    .field-card:hover { border-color: #6366F1; }
    .field-present { border-left: 4px solid #10B981; }
    .field-absent  { border-left: 4px solid #475569; opacity: 0.7; }
    .field-card strong { color: #C4B5FD !important; }
    .field-card code {
        background: #1E3A5F !important;
        color: #93C5FD !important;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .field-card em { color: #64748B !important; }

    /* ===== PIPELINE TRACE STEPS ===== */
    .step-row {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.5rem 0.9rem;
        border-radius: 0.45rem;
        margin-bottom: 0.3rem;
        font-size: 0.84rem;
    }
    .step-ok   { background: #14532D33; border-left: 3px solid #22C55E; color: #86EFAC !important; }
    .step-fail { background: #7F1D1D33; border-left: 3px solid #EF4444; color: #FCA5A5 !important; }
    .step-skip { background: #1E293B;   border-left: 3px solid #475569; color: #94A3B8 !important; }

    /* ===== STREAMLIT WIDGET OVERRIDES ===== */
    /* Text areas + inputs */
    textarea, input[type="text"], input[type="number"] {
        background-color: #1E2130 !important;
        color: #E2E8F0 !important;
        border-color: #374151 !important;
    }
    /* Select boxes */
    div[data-baseweb="select"] > div {
        background-color: #1E2130 !important;
        color: #E2E8F0 !important;
        border-color: #374151 !important;
    }
    /* Dropdown options */
    ul[data-baseweb="menu"] li {
        background-color: #1E2130 !important;
        color: #E2E8F0 !important;
    }
    ul[data-baseweb="menu"] li:hover {
        background-color: #2D3748 !important;
    }
    /* Metric tiles */
    div[data-testid="metric-container"] {
        background: #1E2130 !important;
        border: 1px solid #2D3748 !important;
        border-radius: 0.65rem !important;
        padding: 0.75rem 1rem !important;
    }
    div[data-testid="metric-container"] label {
        color: #94A3B8 !important;
        font-size: 0.8rem !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #E2E8F0 !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    /* Tabs */
    div[data-baseweb="tab-list"] {
        background-color: #1A1F2E !important;
        border-radius: 0.5rem;
        padding: 4px;
        gap: 4px;
    }
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        color: #94A3B8 !important;
        font-weight: 600 !important;
        border-radius: 0.35rem !important;
        padding: 0.4rem 1rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #4F46E5 !important;
        color: #FFFFFF !important;
    }
    /* Expander */
    div[data-testid="stExpander"] > details {
        background: #1A1F2E !important;
        border: 1px solid #2D3748 !important;
        border-radius: 0.55rem !important;
    }
    div[data-testid="stExpander"] summary {
        color: #C4B5FD !important;
        font-weight: 600 !important;
    }
    /* Dataframe */
    div[data-testid="stDataFrame"] {
        background: #1A1F2E !important;
        border-radius: 0.5rem;
    }
    /* Buttons */
    button[kind="primary"] {
        background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
        color: #fff !important;
        border: none !important;
        font-weight: 600 !important;
    }
    button[kind="secondary"] {
        background: #1E2130 !important;
        color: #C4B5FD !important;
        border: 1px solid #4F46E5 !important;
    }
    /* st.json */
    div[data-testid="stJson"] {
        background: #1A1F2E !important;
        border: 1px solid #2D3748 !important;
        border-radius: 0.5rem;
        color: #93C5FD !important;
    }
    /* Code blocks */
    pre, code {
        background: #1A1F2E !important;
        color: #93C5FD !important;
    }
    /* Divider */
    hr { border-color: #2D3748 !important; }
    /* Caption */
    p.caption, small, div[data-testid="stCaptionContainer"] {
        color: #64748B !important;
    }
    /* Progress bar */
    div[data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #4F46E5, #7C3AED) !important;
    }
    /* Alerts */
    div[data-testid="stAlert"] {
        border-radius: 0.5rem !important;
    }
    /* Radio / checkbox labels */
    div[data-testid="stRadio"] label,
    div[data-testid="stCheckbox"] label {
        color: #E2E8F0 !important;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161B2C !important;
    }
    section[data-testid="stSidebar"] * {
        color: #CBD5E1 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown strong {
        color: #A78BFA !important;
    }
    /* Spinner */
    div[data-testid="stSpinner"] { color: #A78BFA !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_preset_examples() -> dict[str, str]:
    return {
        "All Fields Present": (
            "Hi, I'm Priya Sharma. You can reach me at priya@example.com or call "
            "+91 98765 43210. I work as a senior data analyst at Acme Insights."
        ),
        "Partial Fields (Name & Email)": (
            "Hello, my name is John Smith. Please email me back at "
            "john.smith@company.com at your convenience."
        ),
        "No Target Fields (Weather note)": (
            "The weather is pleasant today. I went for a walk in the park "
            "and watched the ducks by the pond."
        ),
        "Messy Resume Header": (
            "Alexander Wright | Chief Technology Officer\n"
            "Mobile: +1 (555) 321-9876\n"
            "GitHub: github.com/awright\n"
            "Contact: a.wright@cloudnet.org"
        ),
        "Ambiguous Text (Order numbers)": (
            "Your shipment #98214-4321 has departed the facility. "
            "Tracking code: 1234-5678-9012. Delivering to 742 Evergreen Terrace."
        ),
        "Email Signature": (
            "Best regards,\nDr. Elena Vasquez\nHead of Research & Development\n"
            "NovaBio Sciences | evasquez@novabio.com | +1 (617) 444-7890"
        ),
        "Business Card": (
            "JAMES THORNTON\nVice President, Global Strategy\nThornton & Associates\n"
            "Tel: +44 20 7123 4567 | j.thornton@thorntonassoc.com"
        ),
    }


def make_mock_caller(presets: dict[str, str]):
    mock_map = {
        presets["All Fields Present"]: {
            "name": "Priya Sharma", "email": "priya@example.com",
            "phone": "+91 98765 43210", "job_title": "senior data analyst",
        },
        presets["Partial Fields (Name & Email)"]: {
            "name": "John Smith", "email": "john.smith@company.com",
            "phone": None, "job_title": None,
        },
        presets["No Target Fields (Weather note)"]: {
            "name": None, "email": None, "phone": None, "job_title": None,
        },
        presets["Messy Resume Header"]: {
            "name": "Alexander Wright", "email": "a.wright@cloudnet.org",
            "phone": "+1 (555) 321-9876", "job_title": "Chief Technology Officer",
        },
        presets["Ambiguous Text (Order numbers)"]: {
            "name": None, "email": None, "phone": None, "job_title": None,
        },
        presets["Email Signature"]: {
            "name": "Dr. Elena Vasquez", "email": "evasquez@novabio.com",
            "phone": "+1 (617) 444-7890", "job_title": "Head of Research & Development",
        },
        presets["Business Card"]: {
            "name": "James Thornton", "email": "j.thornton@thorntonassoc.com",
            "phone": "+44 20 7123 4567", "job_title": "Vice President, Global Strategy",
        },
    }

    def _caller(sys_p: str, user_p: str) -> str:
        for key, val in mock_map.items():
            if key.strip() in user_p:
                return json.dumps(val)
        return json.dumps({"name": None, "email": None, "phone": None, "job_title": None})

    return _caller


def status_badge_html(status: ExtractionStatus) -> str:
    if status == ExtractionStatus.SUCCESS:
        return '<span class="badge badge-success">&#10003; SUCCESS</span>'
    elif status == ExtractionStatus.NEEDS_REVIEW:
        return '<span class="badge badge-review">&#9888; NEEDS REVIEW</span>'
    return '<span class="badge badge-error">&#10007; MODEL ERROR</span>'


def render_pipeline_trace(result: ExtractionResult, max_retries: int) -> None:
    total_attempts = 1 + max_retries
    is_empty = not result.original_text.strip()

    st.markdown(
        '<div class="step-row step-ok">'
        '<strong>Step 1 &mdash; Input guard:</strong>&nbsp;'
        + ("Empty document &mdash; bypass LLM, return null record."
           if is_empty else "Non-empty input, proceeding to LLM.")
        + "</div>",
        unsafe_allow_html=True,
    )

    if not is_empty:
        for i in range(1, result.attempts + 1):
            is_last = (i == result.attempts)
            attempt_errors = [e for e in result.errors if e.startswith(f"Attempt {i}")]

            if result.status == ExtractionStatus.SUCCESS and is_last:
                cls = "step-ok"
                msg = (
                    f"<strong>Attempt {i}/{total_attempts}</strong> &mdash; "
                    "LLM call &rarr; JSON parse &rarr; Schema validation: <strong>PASSED</strong>"
                )
            elif result.status == ExtractionStatus.MODEL_ERROR and is_last:
                cls = "step-fail"
                msg = (
                    f"<strong>Attempt {i}/{total_attempts}</strong> &mdash; "
                    "LLM call: <strong>MODEL ERROR</strong>"
                )
            elif attempt_errors:
                cls = "step-fail"
                msg = (
                    f"<strong>Attempt {i}/{total_attempts}</strong> &mdash; "
                    "Validation failed &rarr; error feedback injected &rarr; retry"
                )
            else:
                cls = "step-ok"
                msg = f"<strong>Attempt {i}/{total_attempts}</strong> &mdash; Passed"

            st.markdown(f'<div class="step-row {cls}">{msg}</div>', unsafe_allow_html=True)

        if result.status == ExtractionStatus.NEEDS_REVIEW:
            st.markdown(
                '<div class="step-row step-fail">'
                "<strong>All retries exhausted</strong> &mdash; Routed to manual review."
                "</div>",
                unsafe_allow_html=True,
            )
        elif result.status == ExtractionStatus.SUCCESS:
            st.markdown(
                '<div class="step-row step-ok">'
                "<strong>Result</strong> &mdash; Valid ContactInfo object emitted."
                "</div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[int, bool]:
    st.sidebar.markdown("## ⚙️ Pipeline Settings")

    model = get_model_name()
    url = get_base_url()
    st.sidebar.markdown(f"**Model:** `{model}`")
    st.sidebar.markdown(f"**Host:** `{url}`")

    is_available, status_msg = check_ollama_availability()
    if is_available:
        st.sidebar.success(f"🟢 {status_msg}")
    else:
        st.sidebar.error(f"🔴 {status_msg}")
        st.sidebar.info(
            "**Setup:**\n"
            "1. Install → https://ollama.com\n"
            "2. `ollama serve`\n"
            f"3. `ollama pull {model}`"
        )

    mock_mode = st.sidebar.checkbox(
        "🧪 Offline Mock Mode",
        value=not is_available,
        help="Use deterministic mock extractor — no Ollama needed.",
    )

    max_retries = st.sidebar.slider(
        "Max Retries on Failure",
        min_value=0, max_value=5,
        value=DEFAULT_MAX_RETRIES,
        help="How many times to retry on validation failure.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📜 Session History")
    count = len(st.session_state.history)
    st.sidebar.caption(f"{count} extraction(s) this session")
    if count > 0 and st.sidebar.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "### Schema\n"
        "- `name` — full name or `null`\n"
        "- `email` — RFC email or `null`\n"
        "- `phone` — phone string or `null`\n"
        "- `job_title` — role/title or `null`\n\n"
        "Invalid → retry with feedback → manual review"
    )

    return max_retries, mock_mode


# ---------------------------------------------------------------------------
# Tab 1 — Single Extraction
# ---------------------------------------------------------------------------

def tab_single(max_retries: int, mock_mode: bool) -> None:
    presets = get_preset_examples()

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown("#### 📝 Input Text")
        selected = st.selectbox(
            "Load a preset example:",
            ["— choose an example —"] + list(presets.keys()),
            key="preset_select",
        )
        initial_text = presets.get(selected, "")

        input_text = st.text_area(
            "Paste unstructured text:",
            value=initial_text,
            height=175,
            placeholder="An email signature, resume snippet, business card, or any free-form note…",
            key="input_text_area",
        )
        st.caption(f"Characters: {len(input_text)}")

        btn_col, clr_col = st.columns([4, 1])
        with btn_col:
            run = st.button("🚀 Extract Contact Info", type="primary", key="run_btn")
        with clr_col:
            if st.button("✕ Clear", key="clr_btn"):
                st.session_state["input_text_area"] = ""
                st.rerun()

    with col_r:
        st.markdown("#### ℹ️ How it works")
        st.markdown(
            '<div class="dark-card">'
            '<ol style="margin:0;padding-left:1.2rem;line-height:2;font-size:0.88rem;">'
            "<li>Empty input check (bypass LLM if blank)</li>"
            "<li>LLM called to extract 4 fields as JSON</li>"
            "<li>Pydantic v2 validates schema &amp; email format</li>"
            "<li>On failure: retry with validation error feedback</li>"
            "<li>Retries exhausted: route to manual review</li>"
            "</ol>"
            "</div>",
            unsafe_allow_html=True,
        )
        if st.session_state.last_result:
            r = st.session_state.last_result
            st.markdown("**Last run:**")
            st.markdown(status_badge_html(r.status), unsafe_allow_html=True)
            st.caption(f"{r.attempts} attempt(s) · {r.processing_time_ms:.1f} ms")

    if run:
        if not input_text.strip():
            st.info("ℹ️ Empty input — returns all-null ContactInfo with status SUCCESS.")

        caller = make_mock_caller(presets) if mock_mode else None
        pipeline = ExtractionPipeline(max_retries=max_retries, custom_caller=caller)

        with st.spinner("⚙️ Running extraction pipeline…"):
            result: ExtractionResult = pipeline.extract_and_validate(input_text)

        st.session_state.last_result = result
        st.session_state.history.append({
            "ts": time.strftime("%H:%M:%S"),
            "preview": input_text[:60].replace("\n", " ") + ("…" if len(input_text) > 60 else ""),
            "status": result.status.value,
            "attempts": result.attempts,
            "latency_ms": round(result.processing_time_ms, 1),
            "fields": result.data.non_null_fields() if result.data else {},
            "result": result,
        })

        _render_result(result, max_retries)


def _render_result(result: ExtractionResult, max_retries: int) -> None:
    st.markdown("---")
    st.markdown("### 📊 Extraction Results")

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(status_badge_html(result.status), unsafe_allow_html=True)
        st.write("")
    with m2:
        st.metric("Attempts", f"{result.attempts} / {1 + max_retries}")
    with m3:
        st.metric("Latency", f"{result.processing_time_ms:.1f} ms")
    with m4:
        if result.is_empty_target:
            st.metric("Fields Found", "0 / 4")
        elif result.is_success and result.data:
            st.metric("Fields Found", f"{len(result.data.non_null_fields())} / 4")
        else:
            st.metric("Fields Found", "—")

    st.write("")

    col_f, col_j = st.columns(2, gap="large")

    with col_f:
        st.markdown("#### 🗂️ Extracted Fields")
        if result.status == ExtractionStatus.SUCCESS and result.data:
            for key, label, icon in [
                ("name",      "Name",      "👤"),
                ("email",     "Email",     "✉️"),
                ("phone",     "Phone",     "📞"),
                ("job_title", "Job Title", "💼"),
            ]:
                val = getattr(result.data, key)
                if val:
                    st.markdown(
                        f'<div class="field-card field-present">'
                        f'{icon} <strong>{label}</strong>&nbsp;&nbsp;'
                        f'<code>{val}</code></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="field-card field-absent">'
                        f'{icon} <strong>{label}</strong>&nbsp;&nbsp;'
                        f'<em>null — not present in text</em></div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.warning("No validated fields to display.")

    with col_j:
        st.markdown("#### 📦 JSON Output")
        if result.status == ExtractionStatus.SUCCESS and result.data:
            json_data = result.data.to_dict()
            st.json(json_data)
            st.download_button(
                "💾 Download JSON",
                data=json.dumps(json_data, indent=2),
                file_name="extracted_contact.json",
                mime="application/json",
            )
        else:
            st.error("❌ No valid JSON — extraction failed or needs review.")
            if result.raw_response:
                with st.expander("🔍 Raw model response"):
                    st.code(result.raw_response, language="text")

    with st.expander("🔬 Pipeline Execution Trace", expanded=(result.status != ExtractionStatus.SUCCESS)):
        render_pipeline_trace(result, max_retries)

    if result.status == ExtractionStatus.NEEDS_REVIEW:
        st.error(
            "🚨 **Manual Review Required** — Output failed schema validation across all retry attempts. "
            "Invalid data is never silently accepted."
        )
        with st.expander("📋 Validation Error Log"):
            for err in result.errors:
                st.code(err, language="text")
    elif result.status == ExtractionStatus.MODEL_ERROR:
        st.error("💥 **Model Error** — Could not reach Ollama. Ensure it is running.")
        for err in result.errors:
            st.code(err, language="text")


# ---------------------------------------------------------------------------
# Tab 2 — Batch Extraction
# ---------------------------------------------------------------------------

def tab_batch(max_retries: int, mock_mode: bool) -> None:
    st.markdown("#### 📋 Batch Extraction")
    st.caption("Process multiple texts at once. Enter one per line or upload a `.txt` / `.jsonl` file.")

    mode = st.radio(
        "Input method:",
        ["Text area (one per line)", "Upload file (.txt or .jsonl)"],
        horizontal=True,
    )

    texts: list[str] = []

    if mode == "Text area (one per line)":
        raw = st.text_area(
            "One text per line:",
            height=200,
            placeholder="Hi, I'm Alice — alice@corp.com\nCall Bob at 555-111-2222, CTO\nThe meeting is Monday at 10 AM",
        )
        if raw.strip():
            texts = [t.strip() for t in raw.splitlines() if t.strip()]
    else:
        uploaded = st.file_uploader("Upload .txt or .jsonl", type=["txt", "jsonl"])
        if uploaded:
            content = uploaded.read().decode("utf-8")
            if uploaded.name.endswith(".jsonl"):
                for line in content.splitlines():
                    try:
                        obj = json.loads(line)
                        texts.append(obj.get("input_text", str(obj)))
                    except Exception:
                        texts.append(line)
            else:
                texts = [t.strip() for t in content.splitlines() if t.strip()]

    if texts:
        st.caption(f"**{len(texts)} text(s) ready.**")

    if st.button("🚀 Run Batch", type="primary", disabled=not texts):
        presets = get_preset_examples()
        caller = make_mock_caller(presets) if mock_mode else None
        pipeline = ExtractionPipeline(max_retries=max_retries, custom_caller=caller)

        rows = []
        bar = st.progress(0)
        info = st.empty()

        for i, text in enumerate(texts):
            info.caption(f"Processing {i + 1} / {len(texts)}…")
            res = pipeline.extract_and_validate(text)
            d = res.data.to_dict() if res.data else {
                "name": None, "email": None, "phone": None, "job_title": None
            }
            rows.append({
                "Input": text[:80] + ("…" if len(text) > 80 else ""),
                "Status": res.status.value,
                "Name": d.get("name"),
                "Email": d.get("email"),
                "Phone": d.get("phone"),
                "Job Title": d.get("job_title"),
                "Attempts": res.attempts,
                "Latency (ms)": round(res.processing_time_ms, 1),
            })
            bar.progress((i + 1) / len(texts))

        bar.empty()
        info.empty()

        df = pd.DataFrame(rows)
        total = len(df)
        ok = (df["Status"] == "success").sum()
        rv = (df["Status"] == "needs_review").sum()
        er = (df["Status"] == "model_error").sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", total)
        c2.metric("✅ Success", int(ok))
        c3.metric("⚠️ Review", int(rv))
        c4.metric("💥 Error", int(er))

        st.markdown("#### Results")

        def _col_status(v):
            if v == "success":
                return "background-color:#14532D;color:#86EFAC"
            if v == "needs_review":
                return "background-color:#78350F;color:#FCD34D"
            return "background-color:#7F1D1D;color:#FCA5A5"

        st.dataframe(
            df.style.map(_col_status, subset=["Status"]),
            height=min(420, 50 + 36 * len(df)),
        )

        c_dl1, c_dl2 = st.columns(2)
        with c_dl1:
            st.download_button("💾 CSV", df.to_csv(index=False), "batch_results.csv", "text/csv")
        with c_dl2:
            st.download_button("💾 JSON", json.dumps(rows, indent=2), "batch_results.json", "application/json")


# ---------------------------------------------------------------------------
# Tab 3 — Evaluation Dashboard
# ---------------------------------------------------------------------------

def tab_dashboard() -> None:
    st.markdown("#### 📈 Evaluation Dashboard")

    results_path = Path(__file__).resolve().parent / "results" / "evaluation.json"
    csv_path = Path(__file__).resolve().parent / "results" / "per_sample_results.csv"

    if not results_path.exists():
        st.warning("No evaluation results found. Run `python evaluate.py --mode mock` first.")
        return

    with open(results_path, encoding="utf-8") as f:
        metrics = json.load(f)

    mode_tag = metrics.get("evaluation_mode", "mock").upper()
    ts = metrics.get("timestamp", "—")
    st.caption(f"Mode: **{mode_tag}** · Timestamp: {ts}")

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Samples",         metrics["total_samples"])
    k2.metric("Schema Validity",       f"{metrics['schema_validity_rate_pct']}%")
    k3.metric("Exact-Record Accuracy", f"{metrics['exact_record_accuracy_pct']}%")
    k4.metric("No-Target Accuracy",    f"{metrics['no_target_detection_accuracy_pct']}%")
    k5.metric("Avg Latency",           f"{metrics['average_latency_ms']} ms")

    st.write("")
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("##### Per-Field P / R / F1")
        pf = metrics["per_field_metrics"]
        field_df = pd.DataFrame({
            "Field":     list(pf.keys()),
            "Precision": [pf[f]["precision"] for f in pf],
            "Recall":    [pf[f]["recall"]    for f in pf],
            "F1 Score":  [pf[f]["f1_score"]  for f in pf],
        }).set_index("Field")
        st.bar_chart(field_df, height=270)

    with col_b:
        st.markdown("##### Pipeline Outcome Breakdown")
        s_n = metrics["successful_extractions"]
        r_n = metrics["manual_review_samples_count"]
        e_n = max(0, metrics["total_samples"] - s_n - r_n)
        outcome_df = pd.DataFrame({
            "Outcome": ["Success", "Needs Review", "Error"],
            "Count":   [s_n, r_n, e_n],
        }).set_index("Outcome")
        st.bar_chart(outcome_df, height=270)

    with st.expander("📋 Full Metrics Table"):
        rows = []
        for fname, fm in metrics["per_field_metrics"].items():
            rows.append({
                "Field": fname,
                "TP": fm["true_positives"],
                "FP": fm["false_positives"],
                "FN": fm["false_negatives"],
                "Precision": fm["precision"],
                "Recall":    fm["recall"],
                "F1 Score":  fm["f1_score"],
            })
        st.dataframe(pd.DataFrame(rows).set_index("Field"))

    if csv_path.exists():
        with st.expander("🔍 Per-Sample Results"):
            sample_df = pd.read_csv(csv_path)
            cats = ["All"] + sorted(sample_df["category"].unique().tolist())
            cat_f = st.selectbox("Filter by category:", cats, key="cat_filter")
            filtered = sample_df if cat_f == "All" else sample_df[sample_df["category"] == cat_f]

            def _col_s(v):
                if v == "success":       return "background-color:#14532D;color:#86EFAC"
                if v == "needs_review":  return "background-color:#78350F;color:#FCD34D"
                return "background-color:#7F1D1D;color:#FCA5A5"

            cols = ["sample_id", "category", "status", "attempts",
                    "latency_ms", "expected_name", "predicted_name", "exact_match"]
            avail = [c for c in cols if c in filtered.columns]
            st.dataframe(
                filtered[avail].style.map(_col_s, subset=["status"]),
                height=400,
            )
            st.caption(f"{len(filtered)} rows")


# ---------------------------------------------------------------------------
# Tab 4 — Session History
# ---------------------------------------------------------------------------

def tab_history() -> None:
    st.markdown("#### 📜 Session History")

    history = st.session_state.history
    if not history:
        st.info("No extractions yet. Run extractions in the Single or Batch tab.")
        return

    total = len(history)
    ok_n = sum(1 for h in history if h["status"] == "success")
    avg_lat = sum(h["latency_ms"] for h in history) / total

    h1, h2, h3 = st.columns(3)
    h1.metric("Total", total)
    h2.metric("Success Rate", f"{ok_n / total * 100:.0f}%")
    h3.metric("Avg Latency", f"{avg_lat:.1f} ms")

    st.write("")

    icons = {"success": "✅", "needs_review": "⚠️", "model_error": "💥"}

    for i, entry in enumerate(reversed(history)):
        idx = total - i
        icon = icons.get(entry["status"], "❓")
        label = f"{icon} #{idx} [{entry['ts']}]  —  {entry['preview']}"
        with st.expander(label, expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Status:** `{entry['status']}`")
            c2.markdown(f"**Attempts:** {entry['attempts']}")
            c3.markdown(f"**Latency:** {entry['latency_ms']} ms")

            fields_str = (
                ", ".join(f"{k}: {v}" for k, v in entry["fields"].items())
                if entry["fields"] else "all null"
            )
            st.markdown(f"**Extracted:** {fields_str}")

            r = entry.get("result")
            if r and r.status == ExtractionStatus.SUCCESS and r.data:
                js = json.dumps(r.data.to_dict(), indent=2)
                st.code(js, language="json")
                st.download_button(
                    f"💾 Download #{idx}",
                    js,
                    f"extraction_{idx}.json",
                    "application/json",
                    key=f"dl_h_{idx}",
                )

    st.markdown("---")
    rows = []
    for entry in history:
        r = entry.get("result")
        d = (r.data.to_dict() if (r and r.data)
             else {"name": None, "email": None, "phone": None, "job_title": None})
        rows.append({
            "time": entry["ts"],
            "preview": entry["preview"],
            "status": entry["status"],
            "attempts": entry["attempts"],
            "latency_ms": entry["latency_ms"],
            **{f"field_{k}": v for k, v in d.items()},
        })
    st.download_button(
        "💾 Export Full Session CSV",
        pd.DataFrame(rows).to_csv(index=False),
        "session_history.csv",
        "text/csv",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.markdown(
        '<div class="main-header"> Structured Output Extraction Pipeline</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">'
        "Extract strictly validated contact data from unstructured text "
        "using local LLMs (Ollama) + Pydantic v2."
        "</div>",
        unsafe_allow_html=True,
    )

    max_retries, mock_mode = render_sidebar()

    mode_label = "🧪 Offline Mock" if mock_mode else "🤖 Live Ollama"
    st.caption(f"Mode: **{mode_label}** &nbsp;·&nbsp; Max retries: **{max_retries}**")

    t1, t2, t3, t4 = st.tabs([
        "🧪 Single Extraction",
        "📋 Batch Extraction",
        "📈 Evaluation Dashboard",
        "📜 Session History",
    ])

    with t1:
        tab_single(max_retries, mock_mode)
    with t2:
        tab_batch(max_retries, mock_mode)
    with t3:
        tab_dashboard()
    with t4:
        tab_history()


if __name__ == "__main__":
    main()
