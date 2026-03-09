# NLLB + Llama 4 + Document Reconstruction Pipeline

**Layout-preserving English → Spanish legal document translation for the Massachusetts Trial Court**

This Google Colab notebook implements the document translation pipeline of the CourtAccess AI system. It takes a Massachusetts court PDF, translates all text from English to Spanish while preserving the original layout (fonts, sizes, alignment, table structure, form fields), and produces a translated PDF ready for download.

---

## Overview

The pipeline chains four systems in sequence:

1. **NLLB-200 1.3B** (Facebook/Meta) — neural machine translation
2. **Legal glossary enforcement** — 741-term EN→ES court terminology database
3. **Llama 4 Maverick** (via Vertex AI) — legal accuracy verification
4. **PyMuPDF** — redaction of original text and layout-preserving reinsertion of translated text

Pages are classified as DIGITAL, SCANNED, or BLANK before processing, and each type is handled differently.

---

## Requirements

### Runtime
- **Google Colab** with GPU (T4 or better, High-RAM recommended)
- Python 3.10+

### Cloud / Credentials
| Credential | Used For | Where to Store |
|---|---|---|
| `GCP_SERVICE_ACCOUNT_JSON` | Vertex AI Llama 4 access | Colab Secrets |

The service account needs the `roles/aiplatform.user` IAM role on your GCP project.

### Key Python Dependencies
| Package | Version | Purpose |
|---|---|---|
| `pymupdf` | 1.27.x | PDF parsing, redaction, text insertion |
| `transformers` | 4.38.2 | NLLB-200 model loading |
| `paddlepaddle` | 2.6.2 | PaddleOCR backend |
| `paddleocr` | 2.8.1 | OCR for scanned pages |
| `torch` | 2.2.2 | GPU inference |
| `sentencepiece` | latest | NLLB tokenizer |
| `presidio-analyzer` | latest | PII detection |
| `presidio-anonymizer` | latest | PII redaction |
| `spacy` + `en_core_web_lg` | latest | Named entity recognition |
| `google-cloud-aiplatform` | latest | Vertex AI client |
| `openai` | latest | OpenAI-compatible Vertex AI client |
| `sacrebleu`, `bert-score` | latest | Translation evaluation metrics |

> ⚠️ **Dependency order matters.** Cell 1 installs packages in a specific sequence to avoid numpy/torch/PaddleOCR version conflicts. Do not reorder or combine the install steps.

---

## Notebook Structure

### Cell 0 — Vertex AI Setup
Authenticates to Google Cloud using a service account JSON stored in Colab Secrets. Creates an OpenAI-compatible client pointing at the Vertex AI endpoint for Llama 4 Maverick (`meta/llama-4-maverick-17b-128e-instruct-maas`).

**Key values to configure:**
```python
PROJECT_ID    = "your-gcp-project-id"
LOCATION      = "us-east5"
VERTEX_MODEL_ID = "meta/llama-4-maverick-17b-128e-instruct-maas"
```

If you get a `401 UNAUTHENTICATED` error mid-run (tokens expire after ~1 hour in Colab), call:
```python
refresh_vertex_credentials()
```

---

### Cell 1 — Install Dependencies
Installs all packages in a dependency-safe order. Must be run once per Colab session. After this cell, **restart the runtime** before running Cell 2+.

---

### Cell 2 — Imports & Global Configuration
Loads all libraries and sets global constants:

| Constant | Default | Description |
|---|---|---|
| `OCR_DPI` | 300 | DPI for rasterizing scanned pages |
| `OCR_CONFIDENCE` | 0.50 | Minimum PaddleOCR confidence threshold |
| `WORK_DIR` | `/content/courtaccess_output` | Output directory |
| `device` | `cuda` (if available) | Inference device |

---

### Cell 3 — PDF Upload
Provides a Colab file picker. The uploaded PDF is saved to `WORK_DIR` and stored in `INPUT_PDF`. Only run this once per document — re-running will overwrite.

---

### Cell 4 — Font Utilities
Provides helper functions used throughout the reconstruction phase:

| Function | Purpose |
|---|---|
| `get_font_code(span)` | Maps PDF font name to PyMuPDF built-in code (used in insert_text fallback only) |
| `get_css_font(span)` | Maps PDF font name directly to CSS `font-family` + bold/italic flags (used by `insert_htmlbox`) |
| `get_font_size(span)` | Returns original span font size with a 6pt floor |
| `_safe_color(span)` | Decodes packed 24-bit integer color to `(r, g, b)` float tuple |

---

### Cell 5 — Page Classifier
Classifies each page before processing. Classification uses three signals in priority order:

| Signal | Threshold | Result |
|---|---|---|
| Image coverage | > 40% | SCANNED |
| Images present + no text spans | — | SCANNED |
| Span count | > 5 | DIGITAL |
| Drawing count | > 10 | DIGITAL (vector form structure) |
| Fallback | — | SCANNED |

**Why drawings matter:** A mostly-blank form page with table boxes and checkbox outlines has few text spans but many vector drawings. The drawing threshold prevents these from being misclassified as scanned.

---

### Cell 6 — NLLB Translation Pipeline
Loads `facebook/nllb-200-distilled-1.3B` and `en_core_web_lg` (spaCy NER). Implements `translate_one(text)` which processes text through these steps in order:

1. **Blank/fill-line bypass** — preserve `___` fill lines as-is
2. **Known labels bypass** — short labels like `DATE`, `SIGNATURE`, `BBO NO.` are looked up directly from `_KNOWN_LABELS` without calling NLLB
3. **Very short text bypass** — ≤ 2 non-space chars preserved as-is
4. **Form footer detection** — standardized footer lines (e.g., `Standardized (BMC) – Civil – TC0061`) get minimal word substitution only
5. **Citation protection** — legal citations (`G.L. c. 209A`, `§ 1A`, URLs, form codes) are replaced with `RFCTnRF` placeholders before NLLB
6. **NER protection** — proper nouns (people, places, organizations) are replaced with `RFPNnRF` placeholders, except institutional terms (`court`, `department`) which flow through to NLLB for translation
7. **NLLB translation** — 4-beam search, max 512 tokens
8. **Placeholder restoration** — citations and proper nouns restored from placeholders
9. **Glossary enforcement** — post-NLLB corrections using `GLOSSARY_ES` terms
10. **Hallucination guard** — output/input length ratio must be between 0.1 and 2.5
11. **Deduplication** — repeated comma-separated clauses collapsed

**Configuration:**
```python
SOURCE_LANG = "eng_Latn"
TARGET_LANG = "spa_Latn"   # change to translate to a different language
```

---

### Cell 7A — Legal Glossary
Downloads the NJ Courts official Spanish legal glossary PDF, parses it into a 741-term EN→ES dictionary (`GLOSSARY_ES`), and applies Massachusetts-specific overrides (`MA_OVERRIDES_ES`).

**Two tiers of terms:**

| Tier | Count | Behavior |
|---|---|---|
| Strict (`MA_OVERRIDES_ES`) | 25 | Always auto-enforced. Single unambiguous translation (e.g., `child` → `menor`) |
| Context-dependent (`CONTEXT_DEPENDENT_ES`) | 10 | Injected into Llama's prompt as reference only. Llama chooses based on document context (e.g., `defendant` → `acusado` criminal / `demandado` civil) |

The glossary JSON is cached at `WORK_DIR/glossary_es.json`. To force a re-parse:
```python
GLOSSARY_ES = load_glossary_es(force_reparse=True)
```

---

### Cell 7 — Llama 4 Legal Verification
Sends batches of NLLB-translated spans to Llama 4 Maverick on Vertex AI for legal accuracy review. Llama's responsibilities:

- Correct context-dependent terms Cell 6 cannot handle automatically
- Fix any meaning errors NLLB introduced
- Verify glossary enforcement didn't break grammar
- Leave already-correct translations unchanged

**Configuration:**
```python
VERIFICATION_MODE = "document"  # "document" = verify all spans
                                # "audio"    = only spans containing glossary terms
```

Batches of 16 spans are sent per API call. The prompt injects up to 12 strict glossary terms and 6 context-dependent alternatives found in the batch text.

---

### Cell 8 — Presidio PII Redaction
Detects and redacts PII from translated text before PDF output.

**Configuration:**
```python
REDACT_PII   = True   # Set False for personal-use translation
REDACT_DATES = False  # Dates are kept visible by default
```

**Entities detected:** `PERSON`, `PHONE_NUMBER`, `EMAIL_ADDRESS`, `US_SSN`, `CREDIT_CARD`, `US_PASSPORT`, `US_DRIVER_LICENSE`, `IP_ADDRESS`, `DATE_TIME`

Legal citations and statutory references are always preserved and never redacted.

---

### Cell 9 — Layout-Preserving Reconstruction
The core PDF reconstruction engine. Processes each DIGITAL page through:

1. **Drawing classification** — identifies checkboxes, underlines, table cells, and page borders from vector drawings
2. **Page classifier** — decides between DIGITAL / SCANNED / BLANK
3. **Block and span extraction** — reads all text from PyMuPDF's `get_text("dict")`
4. **Paragraph grouping** — groups lines into paragraphs using gap size, font size changes, bold/centered style breaks
5. **Column detection** — splits spans into horizontal columns using gap analysis, table cell boundaries, and checkbox positions
6. **Fill-field detection** — separates `___` fill lines from translatable text within the same span
7. **Unit construction** — each translatable cluster becomes a `unit` dict with rect, style groups, container rect, and an `in_table_cell` flag
8. **Redaction** — original text spans are redacted while preserving background color and drawings
9. **Translation insertion** — translated text inserted via `insert_htmlbox` (rich HTML with CSS font/size/color) with fallback to `insert_text`

**Key layout behaviors:**

| Behavior | Mechanism |
|---|---|
| Table cell text never escapes drawn boundary | `in_table_cell=True` → hard clip to container rect, iterative font shrink |
| Free-text blocks reflow around expansions | `in_table_cell=False` → column-aware y-push accumulation |
| Original font style preserved | CSS `font-family`, `font-weight`, `font-style` from span flags |
| Original font size preserved | No shrinking except inside table cells where it's necessary to fit |
| Uppercase headings stay uppercase | Detected from original span, applied to translated output |
| Checkboxes preserved | Vector drawings retained; text positioned relative to checkbox rect |

---

### Diagnostics Cell — Pipeline Tracing
Provides live trace capture of the full pipeline. Enable before running Cell 10:

```python
TRACE_ENABLED = True
traces.clear()
```

After Cell 10 completes:

```python
df = build_trace_report()        # Print summary stats + return DataFrame
show_changes_only(df)            # Show only units modified by glossary or Llama
show_glossary_analysis()         # Per-term NLLB accuracy across all pages
compare_pages(1)                 # Show all units for a specific page
save_trace_html(df)              # Export color-coded HTML report
```

The trace HTML is color-coded: green = unchanged, yellow = glossary corrected, blue = Llama corrected.

---

### Cell 10 — Main Pipeline Loop
Iterates over all pages of the input PDF, classifies each, and routes to the appropriate reconstruction function:

| Page Type | Handler |
|---|---|
| `DIGITAL` | `reconstruct_digital_page()` |
| `SCANNED` (PaddleOCR available) | `reconstruct_scanned_page_paddle()` |
| `SCANNED` (fallback) | `reconstruct_scanned_page_tesseract()` |
| `BLANK` | Skipped |
| Image-only (> 80% image coverage) | Preserved as-is |

Output is saved to `WORK_DIR/{input_stem}_translated.pdf` and automatically downloaded.

---

### Cell 11 — Visual Comparison
Renders a side-by-side PNG comparison of a selected page from the original and translated PDFs for a quick visual sanity check.

```python
show_comparison(INPUT_PDF, OUTPUT_PDF, page_num=0)
```

---

## Running the Full Pipeline

1. Open the notebook in **Google Colab** with GPU runtime
2. Add your GCP service account JSON to Colab Secrets as `GCP_SERVICE_ACCOUNT_JSON`
3. Run **Cell 0** — authenticate to Vertex AI
4. Run **Cell 1** — install dependencies, then **restart the runtime**
5. Run **Cells 2–8** in order to load models and utilities
6. Run the **Diagnostics Cell** and set `TRACE_ENABLED = True` if you want tracing
7. Run **Cell 9** to load the reconstruction engine
8. Run **Cell 3** to upload your PDF
9. Run **Cell 10** — the translated PDF will download automatically
10. Optionally run **Cell 11** for a visual comparison

---

## Language Configuration

The pipeline currently translates **English → Spanish**. To change the target language, update these values in Cell 6:

```python
SOURCE_LANG = "eng_Latn"
TARGET_LANG = "spa_Latn"   # e.g., "por_Latn" for Portuguese
```

NLLB-200 supports 200 languages. See the [NLLB language code list](https://github.com/facebookresearch/flores/blob/main/flores200/README.md) for valid values.

You will also need to update `_KNOWN_LABELS`, `GLOSSARY_ES`, `MA_OVERRIDES_ES`, and the Llama prompt in Cell 7 for the target language.

---

## Evaluation Tools

### Glossary Verification (inline test cell after Cell 6)
Runs 7 representative legal sentences through `translate_one()` and checks for expected Spanish terms. Reports pass/fail per sentence.

### Translation Validation (cell between Cell 6 and Cell 7A)
Compares NLLB-only vs. NLLB+Llama output against the glossary:

```python
validate_translation("The defendant waived the right to bail.")
validate_batch(sentences, save_csv="report.csv", save_html="report.html")
glossary_coverage_report()
quick_test()   # Runs 8 representative legal sentences
```

### RTT + BERTScore (Llama test cell)
Round-trips translations back to English via NLLB and computes BERTScore F1 to compare NLLB-alone vs. NLLB+Llama semantic accuracy. Also runs term-level and citation-preservation accuracy tests.

---

## Known Limitations

- **Token expiry:** Vertex AI access tokens expire after ~1 hour in Colab. Call `refresh_vertex_credentials()` if you get a 401 error mid-run.
- **Scanned pages:** PaddleOCR output quality depends heavily on scan resolution. Results on low-quality scans may need manual review.
- **Handwritten text:** Not supported. Handwritten regions are preserved as-is.
- **Right-to-left languages:** Not tested. PyMuPDF `insert_htmlbox` may not handle RTL correctly without additional CSS.
- **Very long documents:** NLLB has a 512-token input limit. Longer text blocks are split at sentence boundaries before translation (`_split_long_text`).
- **Complex table layouts:** Nested tables or overlapping drawing rects may cause incorrect column assignment.

---

## Output

The output PDF is saved as:
```
{WORK_DIR}/{original_filename}_translated.pdf
```

It is automatically downloaded at the end of Cell 10. The file is saved with `garbage=4` (removes unreferenced objects from redactions) and `deflate=True` (recompresses content streams).

---

## Project Context

This notebook implements the `reconstruct_pdf.py` / `translate_text.py` / `legal_review.py` components of the broader **CourtAccess AI** system — a full MLOps platform for real-time courtroom interpretation and legal document translation for the Massachusetts Trial Court. See the main project README for the full system architecture, CI/CD pipeline, Kubernetes deployment, and Airflow DAG structure.
