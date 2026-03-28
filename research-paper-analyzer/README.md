# 📊 Research Paper Analyzer

> **AI-Powered Multi-Agent Document Intelligence Platform** — Plagiarism Detection · AI Content Analysis · Novelty Scoring · Summarization · Entity Extraction · Metadata Validation · Search & Recommendations

A production-grade Streamlit application that uses **10 specialized AI agents** running in parallel to deeply analyze academic research papers. Built with a free-tier-first philosophy — works out of the box with Wikipedia and DuckDuckGo, with optional Gemini API for enhanced AI capabilities.

---

## ✨ Features

### 🔍 Core Analysis (10 Agents)

| Agent | What It Does |
|---|---|
| **Plagiarism Agent** | Multi-source plagiarism detection via DuckDuckGo exact-match search, local heuristic analysis, and uploaded reference paper comparison |
| **AI Content Agent** | Detects AI-generated text using Gemini API or local heuristics (readability scoring, sentence variance, pattern matching) |
| **Novelty Agent** | Scores research uniqueness (0–10) by analyzing terminology innovation, methodology depth, conceptual novelty, and citation patterns |
| **References Agent** | Discovers related work via Semantic Scholar, Wikipedia, DuckDuckGo, and extracts existing citations from the paper |
| **Summarization Agent** | Generates multi-level summaries (TL;DR, Executive, Detailed) with section-aware extraction and hallucination guards |
| **Structure Analyzer** | Detects document sections (Abstract, Methods, Results, etc.) using regex heuristics + optional Gemini refinement |
| **Metadata Extractor** | Extracts title, authors, emails, DOI, keywords, venue, and publication date with email validation and author–email pairing |
| **Entity Classifier** | Identifies methods, datasets, metrics, tools, organizations, and acronyms via rule-based + Gemini extraction |
| **Validation Agent** | Cross-references metadata against CrossRef (DOI) and Semantic Scholar; validates document structure and reference quality |
| **Recommendation Agent** | Suggests related papers from Semantic Scholar, local content similarity, and topic-based DuckDuckGo search |

### 📚 Document Management
- **Indexing Service** — File-based JSON index with TF-IDF sparse embeddings for keyword + semantic search
- **Document Library** — Browse, search, and find similar documents across your indexed collection
- **Search Page** — Full-text search with novelty and plagiarism filters

### 🎨 UI/UX
- **3 Switchable Neon Themes** — Cyber Cyan, Neon Magenta, Toxic Lime
- **Dark mode** with WCAG AA contrast compliance
- **10-tab analysis dashboard** with detailed breakdowns
- **Neon glow effects**, JetBrains Mono metrics, gradient dividers
- **Responsive design** with Inter font family

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <repository-url>
cd research-paper-analyzer

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key (Optional but recommended)

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

> **Without Gemini:** The app still works using local heuristic analysis for all agents. Gemini enhances AI content detection, summarization, entity extraction, and structure analysis.

Or enter the key directly in the sidebar when the app is running.

### 3. Run the Application

```bash
streamlit run streamlit_app.py
```

The app opens at **http://localhost:8501**.

---

## 📁 Project Structure

```
research-paper-analyzer/
├── streamlit_app.py              # Main application (UI + orchestrator)
├── requirements.txt              # Python dependencies
├── .env                          # API keys (create this yourself)
├── README.md                     # This file
│
├── agents/                       # 10 specialized analysis agents
│   ├── __init__.py
│   ├── base_agent.py             # Abstract base class with retry logic
│   ├── plagiarism_agent.py       # DuckDuckGo + heuristic plagiarism check
│   ├── ai_content_agent.py       # AI content detection (Gemini + local)
│   ├── novelty_agent.py          # Research novelty scoring (0-10)
│   ├── references_agent.py       # Related work discovery
│   ├── summarization_agent.py    # Multi-level summarization
│   ├── structure_analyzer_agent.py  # Section boundary detection
│   ├── metadata_extractor_agent.py  # Title/author/DOI extraction
│   ├── entity_classifier_agent.py   # Named entity recognition
│   ├── validation_agent.py       # Metadata cross-referencing
│   └── recommendation_agent.py   # Paper recommendations
│
├── services/
│   └── indexing_service.py       # File-based document index
│
├── utils/
│   └── text_extractor.py         # PDF/TXT text extraction
│
├── config/
│   └── sample_keys.json          # Sample API key configuration
│
└── data/
    └── index/                    # Document index storage (auto-created)
        └── document_index.json
```

---

## 🏗️ Architecture

### Multi-Agent Orchestration

```
                    ┌─────────────────┐
                    │   Streamlit UI   │
                    │  (streamlit_app) │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Orchestrator   │
                    │ (analyze_paper) │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │      Phase 1: Parallel       │
              │   ThreadPoolExecutor (8)     │
    ┌─────────┼─────────┬─────────┬─────────┤
    ▼         ▼         ▼         ▼         ▼
Plagiarism  AI Det.  Novelty  References  Summary
Structure  Metadata  Entities
              │
              ├──── Phase 2: Validation ────▶ (depends on metadata)
              │
              ├──── Phase 3: Recommendations ▶ (uses index)
              │
              └──── Phase 4: Indexing ──────▶ (stores results)
```

### Agent Design Pattern

Every agent inherits from `BaseAgent`, which provides:
- **Retry-enabled HTTP session** (3 retries, backoff for 429/5xx)
- **Unified API request method** with timeout handling
- **Fallback strategy** — Gemini API first → local heuristic fallback
- **Abstract `analyze()` method** — consistent interface across all agents

### Data Flow

1. **Upload** → PDF/TXT extracted via `pdfplumber` (primary) or `PyPDF2` (fallback)
2. **Phase 1** → 8 agents run in parallel via `ThreadPoolExecutor`
3. **Phase 2** → Validation agent cross-references extracted metadata
4. **Phase 3** → Recommendation agent queries Semantic Scholar + local index
5. **Phase 4** → Document indexed with TF-IDF sparse embedding
6. **Display** → Results shown in 10-tab dashboard with neon theme

---

## 🔌 API & Service Dependencies

| Service | Required? | Purpose | Free Tier |
|---|---|---|---|
| **Gemini API** | Optional | AI detection, summarization, entity extraction, structure analysis | ✅ Free with Google account |
| **Semantic Scholar** | Auto | Paper validation, recommendations, reference search | ✅ Free, no key needed |
| **CrossRef** | Auto | DOI validation and metadata retrieval | ✅ Free, no key needed |
| **Wikipedia** | Toggle | Related work discovery | ✅ Free |
| **DuckDuckGo** | Toggle | Plagiarism checking, reference search | ✅ Free |

### Getting a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Click "Create API Key"
3. Copy the key and paste it in the sidebar or `.env` file

---

## 📋 Dependencies

```
streamlit>=1.28.0          # Web UI framework
PyPDF2>=3.0.1              # PDF fallback reader
pdfplumber>=0.9.0          # Primary PDF reader
requests>=2.31.0           # HTTP client
beautifulsoup4>=4.12.0     # HTML parsing
wikipedia>=1.4.0           # Wikipedia search
duckduckgo-search>=3.9.0   # DuckDuckGo search
textstat>=0.7.3            # Readability scoring
nltk>=3.8.1                # NLP tokenization
scikit-learn>=1.3.0        # TF-IDF vectorization
pandas>=2.0.3              # Data handling
numpy>=1.24.3              # Numerical ops
python-dotenv>=1.0.0       # Environment variable loading
```

---

## 🎨 Neon Theme System

The app includes **3 switchable neon themes** selectable from the sidebar:

| Theme | Primary Color | Background | Vibe |
|---|---|---|---|
| ⚡ **Cyber Cyan** | `#00e5ff` | `#0a0e17` | Electric blue sci-fi |
| 💜 **Neon Magenta** | `#ff00e5` | `#0d0a14` | Purple synthwave |
| 🟢 **Toxic Lime** | `#76ff03` | `#080c08` | Matrix hacker green |

### Design System

- **CSS Custom Properties** — All colors as `--neon-*` tokens
- **WCAG AA compliance** — Body text ≥ 4.5:1 contrast ratio
- **Keyboard focus** — 3px solid ring for accessibility
- **Reduced motion** — Respects `prefers-reduced-motion` media query
- **Fonts** — Inter (UI) + JetBrains Mono (metrics/code)

---

## 📊 Understanding Results

### Plagiarism Score
| Range | Status | Meaning |
|---|---|---|
| 0–5% | ✅ Acceptable | Standard academic overlap |
| 5–15% | ⚠️ Moderate | May need review |
| 15%+ | ❌ High | Significant matches found |

### AI Content Score
| Range | Status | Meaning |
|---|---|---|
| 0% | ✅ None | No AI patterns detected |
| 1–20% | ⚠️ Low | Minor AI-typical patterns |
| 20%+ | ❌ High | Strong AI generation indicators |

### Novelty Score (0–10)
| Range | Status | Meaning |
|---|---|---|
| 7–10 | ✅ High | Strong innovation indicators |
| 5–6.9 | ℹ️ Moderate | Some novel elements |
| 0–4.9 | ⚠️ Low | Limited novelty signals |

### Trust Score (Validation)
| Range | Status | Meaning |
|---|---|---|
| 80–100% | ✅ Verified | Metadata cross-referenced successfully |
| 50–79% | ⚠️ Partial | Some checks passed |
| 0–49% | ❌ Low | Metadata could not be verified |

---

## 🔍 How Each Agent Works

### Plagiarism Agent
1. Extracts key sentences (longest, most unique)
2. Searches DuckDuckGo with exact-match queries (`"sentence"`)
3. Runs local heuristic checks (common phrase density, formatting anomalies, style variance)
4. Compares against uploaded reference papers (TF-IDF cosine similarity)
5. Deduplicates and scores

### AI Content Agent
1. **Gemini path:** Sends text to Gemini with specific detection prompts → parses JSON response
2. **Local fallback:** Analyzes readability (Flesch score), transition word frequency, sentence length variance, personal voice absence, and grammar perfection

### Summarization Agent
1. Detects available sections (Abstract, Methods, Results, Conclusion, etc.)
2. **Gemini path:** Sends section-aware prompt with hallucination guards → parses structured response (TL;DR, Executive, Detailed, Key Findings)
3. **Extractive fallback:** Scores sentences by position (abstract/conclusion zones get bonus), importance keywords, and length; assembles summaries from top-scored sentences

### Metadata Extractor
1. Regex extraction for title, authors, emails, DOI, keywords, venue, date
2. Gemini-enhanced extraction for fields regex missed
3. Email format validation (RFC-like checks)
4. Author ↔ Email pairing with confidence levels
5. Integrity notes for discrepancies
6. Unavailable field flagging with verification instructions

---

## 🛠️ Configuration

### Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key for enhanced AI features |

### Sidebar Controls

- **🎨 Neon Theme** — Switch between Cyber Cyan, Neon Magenta, Toxic Lime
- **🔑 Gemini API Key** — Enter or update API key at runtime
- **🌐 Search Services** — Toggle Wikipedia and DuckDuckGo on/off
- **🔍 Test API Connections** — Verify all configured services
- **📁 Document Index** — View count of indexed documents

---

## 📄 Supported File Formats

| Format | Extension | Extraction Method |
|---|---|---|
| PDF | `.pdf` | pdfplumber (primary) → PyPDF2 (fallback) |
| Plain Text | `.txt` | Direct UTF-8 read |

> **Note:** Image-based PDFs (scanned documents) are not supported. The PDF must contain extractable text layers.

---

## 🚀 Deployment

### Local Development
```bash
streamlit run streamlit_app.py
```

### Streamlit Cloud
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set `GEMINI_API_KEY` in Secrets management
5. Deploy

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501"]
```

---

## 📝 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgments

- **Google Gemini** — AI-powered analysis capabilities
- **Semantic Scholar** — Academic paper search and validation
- **CrossRef** — DOI resolution and metadata verification
- **Streamlit** — Web application framework
- **scikit-learn** — TF-IDF vectorization for document indexing
