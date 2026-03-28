import streamlit as st
import os
import json
import logging
import hashlib
from typing import Dict, Any
import concurrent.futures
from datetime import datetime

from agents.plagiarism_agent import PlagiarismAgent
from agents.ai_content_agent import AIContentAgent
from agents.novelty_agent import NoveltyAgent
from agents.references_agent import ReferencesAgent
from agents.summarization_agent import SummarizationAgent
from agents.structure_analyzer_agent import StructureAnalyzerAgent
from agents.metadata_extractor_agent import MetadataExtractorAgent
from agents.entity_classifier_agent import EntityClassifierAgent
from agents.validation_agent import ValidationAgent
from agents.recommendation_agent import RecommendationAgent
from services.indexing_service import IndexingService
from utils.text_extractor import extract_text_from_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Research Paper Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


class ResearchPaperAnalyzer:
    def __init__(self):
        self.agents = {
            'plagiarism': PlagiarismAgent(),
            'ai_content': AIContentAgent(),
            'novelty': NoveltyAgent(),
            'references': ReferencesAgent(),
            'summarization': SummarizationAgent(),
            'structure': StructureAnalyzerAgent(),
            'metadata': MetadataExtractorAgent(),
            'entities': EntityClassifierAgent(),
            'validation': ValidationAgent(),
            'recommendations': RecommendationAgent(),
        }
        self.indexing_service = IndexingService()

    def analyze_paper(self, text: str, api_keys: Dict[str, str],
                      paper_title: str = "", reference_texts: list = None) -> Dict[str, Any]:
        """Orchestrate multi-agent analysis with parallel execution."""

        results = {}

        # Phase 1: Run independent agents in parallel
        phase1_agents = ['plagiarism', 'ai_content', 'novelty', 'references',
                         'summarization', 'structure', 'metadata', 'entities']

        def run_agent(agent_name: str, agent):
            try:
                logger.info(f"Running {agent_name} agent")
                if agent_name == 'references':
                    result = agent.analyze(text, api_keys, paper_title)
                elif agent_name == 'plagiarism':
                    result = agent.analyze(text, api_keys, reference_texts=reference_texts)
                elif agent_name == 'recommendations':
                    indexed_docs = self.indexing_service.get_all_documents()
                    result = agent.analyze(text, api_keys, paper_title=paper_title,
                                           indexed_documents=indexed_docs)
                else:
                    result = agent.analyze(text, api_keys)
                result['status'] = 'completed'
                return agent_name, result
            except Exception as e:
                logger.error(f"{agent_name} agent failed: {str(e)}")
                return agent_name, {'status': 'failed', 'error': str(e)}

        # Phase 1: Independent agents
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(run_agent, name, self.agents[name]): name
                for name in phase1_agents
            }
            for future in concurrent.futures.as_completed(futures):
                agent_name, result = future.result()
                results[agent_name] = result

        # Phase 2: Validation (depends on metadata)
        try:
            metadata = results.get('metadata', {})
            validation_result = self.agents['validation'].analyze(
                text, api_keys, metadata=metadata, paper_title=paper_title
            )
            validation_result['status'] = 'completed'
            results['validation'] = validation_result
        except Exception as e:
            results['validation'] = {'status': 'failed', 'error': str(e)}

        # Phase 3: Recommendations
        try:
            indexed_docs = self.indexing_service.get_all_documents()
            rec_result = self.agents['recommendations'].analyze(
                text, api_keys, paper_title=paper_title, indexed_documents=indexed_docs
            )
            rec_result['status'] = 'completed'
            results['recommendations'] = rec_result
        except Exception as e:
            results['recommendations'] = {'status': 'failed', 'error': str(e)}

        # Phase 4: Index the document
        try:
            doc_id = hashlib.md5(text[:500].encode()).hexdigest()[:12]
            self.indexing_service.add_document(
                doc_id=doc_id,
                title=paper_title or results.get('metadata', {}).get('title', 'Untitled'),
                text=text,
                metadata=results.get('metadata', {}),
                analysis_results={
                    'plagiarism': results.get('plagiarism', {}),
                    'ai_content': results.get('ai_content', {}),
                    'novelty': results.get('novelty', {}),
                }
            )
            results['indexing'] = {'status': 'completed', 'doc_id': doc_id}
        except Exception as e:
            results['indexing'] = {'status': 'failed', 'error': str(e)}

        return results


def load_sample_config():
    """Load sample API key configuration."""
    try:
        with open('config/sample_keys.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "gemini_api_key": "",
            "wikipedia_enabled": True,
            "duckduckgo_enabled": True
        }


# ────────────────────────────────────────────
# Neon Theme Definitions
# ────────────────────────────────────────────
NEON_THEMES = {
    "⚡ Cyber Cyan": {
        "id": "cyber_cyan",
        "bg": "#0a0e17",
        "surface": "#111827",
        "surface_2": "#1a2235",
        "primary": "#00e5ff",
        "primary_dim": "#00b8d4",
        "secondary": "#7c4dff",
        "accent": "#00e676",
        "text": "#e8eaed",
        "text_muted": "#8b95a5",
        "border": "rgba(0, 229, 255, 0.2)",
        "glow": "0 0 15px rgba(0, 229, 255, 0.3), 0 0 30px rgba(0, 229, 255, 0.1)",
        "glow_sm": "0 0 8px rgba(0, 229, 255, 0.25)",
        "hover_bg": "#1e3a5f",
        "focus_ring": "0 0 0 3px rgba(0, 229, 255, 0.5)",
        "success": "#00e676",
        "warning": "#ffab00",
        "error": "#ff5252",
        "info": "#40c4ff",
        # Entity tag neon colors
        "tag_method": {"bg": "rgba(0, 229, 255, 0.15)", "fg": "#00e5ff"},
        "tag_dataset": {"bg": "rgba(0, 230, 118, 0.15)", "fg": "#00e676"},
        "tag_metric": {"bg": "rgba(255, 171, 0, 0.15)", "fg": "#ffab00"},
        "tag_tool": {"bg": "rgba(124, 77, 255, 0.15)", "fg": "#b388ff"},
        "tag_acronym": {"bg": "rgba(64, 196, 255, 0.15)", "fg": "#40c4ff"},
        "tag_org": {"bg": "rgba(255, 82, 82, 0.15)", "fg": "#ff5252"},
        "tag_person": {"bg": "rgba(234, 128, 252, 0.15)", "fg": "#ea80fc"},
    },
    "💜 Neon Magenta": {
        "id": "neon_magenta",
        "bg": "#0d0a14",
        "surface": "#1a1028",
        "surface_2": "#241838",
        "primary": "#ff00e5",
        "primary_dim": "#cc00b8",
        "secondary": "#00e5ff",
        "accent": "#ffea00",
        "text": "#f0e6f6",
        "text_muted": "#9b8aad",
        "border": "rgba(255, 0, 229, 0.2)",
        "glow": "0 0 15px rgba(255, 0, 229, 0.3), 0 0 30px rgba(255, 0, 229, 0.1)",
        "glow_sm": "0 0 8px rgba(255, 0, 229, 0.25)",
        "hover_bg": "#3d1a5c",
        "focus_ring": "0 0 0 3px rgba(255, 0, 229, 0.5)",
        "success": "#00e676",
        "warning": "#ffea00",
        "error": "#ff5252",
        "info": "#e040fb",
        "tag_method": {"bg": "rgba(255, 0, 229, 0.15)", "fg": "#ff00e5"},
        "tag_dataset": {"bg": "rgba(0, 230, 118, 0.15)", "fg": "#00e676"},
        "tag_metric": {"bg": "rgba(255, 234, 0, 0.15)", "fg": "#ffea00"},
        "tag_tool": {"bg": "rgba(0, 229, 255, 0.15)", "fg": "#00e5ff"},
        "tag_acronym": {"bg": "rgba(124, 77, 255, 0.15)", "fg": "#b388ff"},
        "tag_org": {"bg": "rgba(255, 82, 82, 0.15)", "fg": "#ff5252"},
        "tag_person": {"bg": "rgba(64, 196, 255, 0.15)", "fg": "#40c4ff"},
    },
    "🟢 Toxic Lime": {
        "id": "toxic_lime",
        "bg": "#080c08",
        "surface": "#0f1a0f",
        "surface_2": "#1a2e1a",
        "primary": "#76ff03",
        "primary_dim": "#64dd17",
        "secondary": "#00e5ff",
        "accent": "#ff6d00",
        "text": "#e8f5e9",
        "text_muted": "#81a784",
        "border": "rgba(118, 255, 3, 0.2)",
        "glow": "0 0 15px rgba(118, 255, 3, 0.3), 0 0 30px rgba(118, 255, 3, 0.1)",
        "glow_sm": "0 0 8px rgba(118, 255, 3, 0.25)",
        "hover_bg": "#1a3d1a",
        "focus_ring": "0 0 0 3px rgba(118, 255, 3, 0.5)",
        "success": "#76ff03",
        "warning": "#ff6d00",
        "error": "#ff1744",
        "info": "#00e5ff",
        "tag_method": {"bg": "rgba(118, 255, 3, 0.15)", "fg": "#76ff03"},
        "tag_dataset": {"bg": "rgba(0, 229, 255, 0.15)", "fg": "#00e5ff"},
        "tag_metric": {"bg": "rgba(255, 109, 0, 0.15)", "fg": "#ff6d00"},
        "tag_tool": {"bg": "rgba(234, 128, 252, 0.15)", "fg": "#ea80fc"},
        "tag_acronym": {"bg": "rgba(255, 234, 0, 0.15)", "fg": "#ffea00"},
        "tag_org": {"bg": "rgba(255, 23, 68, 0.15)", "fg": "#ff1744"},
        "tag_person": {"bg": "rgba(64, 196, 255, 0.15)", "fg": "#40c4ff"},
    },
}


def inject_css(theme_name: str = "⚡ Cyber Cyan"):
    """Inject neon-themed CSS with full dark mode and glow effects."""
    t = NEON_THEMES.get(theme_name, NEON_THEMES["⚡ Cyber Cyan"])

    st.markdown(f"""
        <style>
        /* ═══════════════════════════════════════════
           NEON THEME: {theme_name}
           WCAG AA: body text ≥ 4.5:1, large text ≥ 3:1
           Focus indicators: 3px solid ring
           ═══════════════════════════════════════════ */

        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* ── CSS Custom Properties (Design Tokens) ── */
        :root {{
            --neon-bg: {t['bg']};
            --neon-surface: {t['surface']};
            --neon-surface-2: {t['surface_2']};
            --neon-primary: {t['primary']};
            --neon-primary-dim: {t['primary_dim']};
            --neon-secondary: {t['secondary']};
            --neon-accent: {t['accent']};
            --neon-text: {t['text']};
            --neon-text-muted: {t['text_muted']};
            --neon-border: {t['border']};
            --neon-glow: {t['glow']};
            --neon-glow-sm: {t['glow_sm']};
            --neon-hover-bg: {t['hover_bg']};
            --neon-focus-ring: {t['focus_ring']};
            --neon-success: {t['success']};
            --neon-warning: {t['warning']};
            --neon-error: {t['error']};
            --neon-info: {t['info']};
        }}

        /* ── Global Reset ── */
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, sans-serif !important;
            color: var(--neon-text) !important;
        }}

        /* ── Main Background ── */
        .stApp, .stApp > header {{
            background-color: var(--neon-bg) !important;
        }}

        section[data-testid="stSidebar"] {{
            background-color: var(--neon-surface) !important;
            border-right: 1px solid var(--neon-border) !important;
        }}
        section[data-testid="stSidebar"] * {{
            color: var(--neon-text) !important;
        }}

        .block-container {{
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            max-width: 1200px !important;
        }}

        /* ── Headings ── */
        h1, h2, h3, h4, h5, h6 {{
            font-weight: 800 !important;
            letter-spacing: -0.03em;
            color: var(--neon-primary) !important;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 0.5rem;
            text-shadow: var(--neon-glow-sm);
            font-size: 2.5rem !important;
        }}
        h2 {{
            border-bottom: 1px solid var(--neon-border);
            padding-bottom: 0.5rem;
        }}

        .subtitle {{
            text-align: center;
            font-size: 1.05rem;
            color: var(--neon-text-muted) !important;
            margin-bottom: 1.5rem;
            font-weight: 500;
        }}

        /* ── Body text and paragraphs ── */
        p, span, label, .stMarkdown {{
            color: var(--neon-text) !important;
        }}
        .stCaption, [data-testid="stCaptionContainer"] {{
            color: var(--neon-text-muted) !important;
        }}

        /* ── Buttons — Neon Glow ── */
        .stButton > button {{
            border-radius: 8px;
            height: 3.2rem;
            background: var(--neon-surface) !important;
            color: var(--neon-primary) !important;
            font-weight: 700;
            font-family: 'Inter', sans-serif !important;
            border: 1px solid var(--neon-primary) !important;
            box-shadow: var(--neon-glow-sm);
            transition: all 0.25s ease;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 0.85rem;
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            background: var(--neon-hover-bg) !important;
            box-shadow: var(--neon-glow);
            color: var(--neon-primary) !important;
        }}
        .stButton > button:focus-visible {{
            outline: none !important;
            box-shadow: var(--neon-focus-ring) !important;
        }}
        .stButton > button:active {{
            transform: translateY(0px);
            box-shadow: var(--neon-glow-sm);
        }}

        /* ── Text Inputs ── */
        div[data-testid="stTextInput"] input,
        textarea,
        .stSelectbox > div > div {{
            background-color: var(--neon-surface-2) !important;
            color: var(--neon-text) !important;
            border: 1px solid var(--neon-border) !important;
            border-radius: 8px !important;
            transition: border-color 0.2s, box-shadow 0.2s;
        }}
        div[data-testid="stTextInput"] input:focus,
        textarea:focus {{
            border-color: var(--neon-primary) !important;
            box-shadow: var(--neon-focus-ring) !important;
            outline: none !important;
        }}

        /* ── Metric Cards — Neon Surface ── */
        div[data-testid="metric-container"] {{
            background-color: var(--neon-surface) !important;
            border: 1px solid var(--neon-border) !important;
            border-radius: 12px;
            padding: 1.2rem !important;
            text-align: center;
            box-shadow: var(--neon-glow-sm);
            transition: all 0.3s ease;
        }}
        div[data-testid="metric-container"]:hover {{
            box-shadow: var(--neon-glow);
            border-color: var(--neon-primary) !important;
        }}
        [data-testid="stMetricValue"] {{
            font-size: 2.2rem !important;
            font-weight: 900;
            color: var(--neon-primary) !important;
            text-shadow: var(--neon-glow-sm);
            font-family: 'JetBrains Mono', monospace !important;
        }}
        [data-testid="stMetricLabel"] {{
            font-size: 0.8rem !important;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--neon-text-muted) !important;
        }}

        /* ── File Uploaders — Neon Dashed ── */
        div[data-testid="stFileUploader"] {{
            border: 2px dashed var(--neon-border) !important;
            border-radius: 12px;
            padding: 1.2rem 1rem;
            max-width: 360px;
            min-height: 180px;
            margin: 0 auto;
            background-color: var(--neon-surface) !important;
            transition: all 0.3s ease;
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        div[data-testid="stFileUploader"]:hover {{
            border-color: var(--neon-primary) !important;
            box-shadow: var(--neon-glow-sm);
        }}
        div[data-testid="stFileUploader"] * {{
            color: var(--neon-text-muted) !important;
        }}

        /* ── Tabs — Neon Underline ── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0px;
            background-color: var(--neon-surface) !important;
            border-radius: 8px;
            padding: 4px;
        }}
        .stTabs [data-baseweb="tab"] {{
            padding: 8px 14px;
            font-size: 0.82rem;
            color: var(--neon-text-muted) !important;
            font-weight: 600;
            border-radius: 6px;
            transition: all 0.2s ease;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            color: var(--neon-primary) !important;
            background-color: var(--neon-surface-2) !important;
        }}
        .stTabs [aria-selected="true"] {{
            color: var(--neon-primary) !important;
            background-color: var(--neon-surface-2) !important;
            box-shadow: var(--neon-glow-sm);
        }}
        .stTabs [data-baseweb="tab-highlight"] {{
            background-color: var(--neon-primary) !important;
        }}

        /* ── Expanders ── */
        details[data-testid="stExpander"] {{
            background-color: var(--neon-surface) !important;
            border: 1px solid var(--neon-border) !important;
            border-radius: 10px !important;
        }}
        details[data-testid="stExpander"] summary {{
            color: var(--neon-text) !important;
        }}
        details[data-testid="stExpander"] summary:hover {{
            color: var(--neon-primary) !important;
        }}

        /* ── Progress Bars ── */
        .stProgress > div > div > div {{
            background-color: var(--neon-primary) !important;
            box-shadow: var(--neon-glow-sm);
        }}
        .stProgress > div > div {{
            background-color: var(--neon-surface-2) !important;
        }}

        /* ── Alerts — Neon Variants ── */
        div[data-testid="stAlert"] {{
            border-radius: 10px !important;
            border-left: 4px solid var(--neon-primary) !important;
            background-color: var(--neon-surface) !important;
        }}
        .stSuccess {{
            border-left-color: var(--neon-success) !important;
        }}
        .stWarning {{
            border-left-color: var(--neon-warning) !important;
        }}
        .stError {{
            border-left-color: var(--neon-error) !important;
        }}

        /* ── Checkboxes & Radio ── */
        .stCheckbox label span {{
            color: var(--neon-text) !important;
        }}
        .stCheckbox [data-testid="stCheckbox"] {{
            accent-color: var(--neon-primary);
        }}
        input[type="checkbox"]:checked {{
            background-color: var(--neon-primary) !important;
        }}

        /* ── Sliders ── */
        .stSlider [data-testid="stThumbValue"] {{
            color: var(--neon-primary) !important;
            font-family: 'JetBrains Mono', monospace !important;
        }}

        /* ── Dividers & Horizontal rules ── */
        hr {{
            border-color: var(--neon-border) !important;
        }}

        /* ── Entity tags — Neon ── */
        .entity-tag {{
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            margin: 0.2rem;
        }}
        .entity-method {{ background: {t['tag_method']['bg']}; color: {t['tag_method']['fg']}; }}
        .entity-dataset {{ background: {t['tag_dataset']['bg']}; color: {t['tag_dataset']['fg']}; }}
        .entity-metric {{ background: {t['tag_metric']['bg']}; color: {t['tag_metric']['fg']}; }}
        .entity-tool {{ background: {t['tag_tool']['bg']}; color: {t['tag_tool']['fg']}; }}
        .entity-acronym {{ background: {t['tag_acronym']['bg']}; color: {t['tag_acronym']['fg']}; }}
        .entity-organization {{ background: {t['tag_org']['bg']}; color: {t['tag_org']['fg']}; }}
        .entity-person {{ background: {t['tag_person']['bg']}; color: {t['tag_person']['fg']}; }}

        /* ── Status Badges — Neon ── */
        .badge-pass {{ color: var(--neon-success); font-weight: 700; text-shadow: 0 0 6px {t['success']}44; }}
        .badge-warn {{ color: var(--neon-warning); font-weight: 700; text-shadow: 0 0 6px {t['warning']}44; }}
        .badge-fail {{ color: var(--neon-error); font-weight: 700; text-shadow: 0 0 6px {t['error']}44; }}

        /* ── Recommendation cards ── */
        .rec-card {{
            background: var(--neon-surface) !important;
            border: 1px solid var(--neon-border);
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 0.8rem;
            transition: all 0.3s ease;
        }}
        .rec-card:hover {{
            border-color: var(--neon-primary);
            box-shadow: var(--neon-glow-sm);
        }}

        /* ── Code blocks (monospace) ── */
        code, pre, .stCodeBlock {{
            font-family: 'JetBrains Mono', monospace !important;
            background-color: var(--neon-surface-2) !important;
            color: var(--neon-accent) !important;
            border: 1px solid var(--neon-border) !important;
            border-radius: 6px;
        }}

        /* ── Inline code in markdown ── */
        .stMarkdown code {{
            background-color: {t['tag_method']['bg']} !important;
            color: var(--neon-primary) !important;
            padding: 0.15em 0.4em;
            border-radius: 4px;
            font-size: 0.85em;
        }}

        /* ── Scrollbar — Neon ── */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: var(--neon-bg);
        }}
        ::-webkit-scrollbar-thumb {{
            background: var(--neon-border);
            border-radius: 4px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: var(--neon-primary-dim);
        }}

        /* ── Links ── */
        a {{
            color: var(--neon-primary) !important;
            text-decoration: none;
            transition: all 0.2s ease;
        }}
        a:hover {{
            text-shadow: var(--neon-glow-sm);
            text-decoration: underline;
        }}

        /* ── Sidebar selectbox for theme ── */
        .theme-label {{
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--neon-text-muted);
            margin-bottom: 0.3rem;
        }}

        /* ── Keyboard Focus — Accessibility ── */
        *:focus-visible {{
            outline: 2px solid var(--neon-primary) !important;
            outline-offset: 2px;
        }}

        /* ── Reduced motion preference ── */
        @media (prefers-reduced-motion: reduce) {{
            * {{
                animation: none !important;
                transition-duration: 0.01ms !important;
            }}
        }}

        /* ── Neon divider ── */
        .neon-divider {{
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--neon-primary), transparent);
            border: none;
            margin: 1.5rem 0;
            box-shadow: var(--neon-glow-sm);
        }}
        </style>
    """, unsafe_allow_html=True)


# ────────────────────────────────────────────
# Main Application
# ────────────────────────────────────────────
def main():
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'analysis_running' not in st.session_state:
        st.session_state.analysis_running = False
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ResearchPaperAnalyzer()
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'analyze'
    if 'neon_theme' not in st.session_state:
        st.session_state.neon_theme = "⚡ Cyber Cyan"

    # Sidebar — Theme selector FIRST (so CSS applies immediately)
    with st.sidebar:
        st.markdown('<p class="theme-label">🎨 NEON THEME</p>', unsafe_allow_html=True)
        selected_theme = st.selectbox(
            "Theme",
            options=list(NEON_THEMES.keys()),
            index=list(NEON_THEMES.keys()).index(st.session_state.neon_theme),
            label_visibility="collapsed",
        )
        if selected_theme != st.session_state.neon_theme:
            st.session_state.neon_theme = selected_theme
            st.rerun()

        st.markdown("---")

    # Inject CSS with selected theme
    inject_css(st.session_state.neon_theme)

    st.title("Research Paper Analyzer")
    st.markdown('<p class="subtitle">AI-Powered Multi-Agent System — Plagiarism · AI Detection · Novelty · Summarization · Search</p>', unsafe_allow_html=True)

    # Sidebar continued
    with st.sidebar:
        st.header("🔑 API Configuration")
        st.markdown("Configure AI models and search engines.")

        api_keys = {}
        api_keys['gemini_api_key'] = st.text_input(
            "Gemini API Key",
            value=os.getenv('GEMINI_API_KEY', ''),
            type="password",
            help="For AI content detection, summarization, and entity extraction"
        )

        st.markdown("---")
        st.subheader("🌐 Search Services")
        api_keys['wikipedia_enabled'] = st.checkbox("Wikipedia Search", value=True)
        api_keys['duckduckgo_enabled'] = st.checkbox("DuckDuckGo Search", value=True)

        st.markdown("---")
        st.info("💡 At least one API key or search service must be enabled.")

        if st.button("🔍 Test API Connections"):
            test_apis(api_keys)

        st.markdown("---")

        # Index stats
        st.subheader("📁 Document Index")
        stats = st.session_state.analyzer.indexing_service.get_stats()
        st.metric("Indexed Documents", stats['total_documents'])
        if stats['total_documents'] > 0:
            st.caption(f"Last updated: {stats['last_updated'][:19]}")

    # Navigation
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    with nav_col1:
        if st.button("📄 Analyze Paper", use_container_width=True):
            st.session_state.current_page = 'analyze'
    with nav_col2:
        if st.button("🔎 Search Documents", use_container_width=True):
            st.session_state.current_page = 'search'
    with nav_col3:
        if st.button("📚 Document Library", use_container_width=True):
            st.session_state.current_page = 'library'

    # Neon gradient divider
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # Route pages
    if st.session_state.current_page == 'analyze':
        show_analyze_page(api_keys)
    elif st.session_state.current_page == 'search':
        show_search_page()
    elif st.session_state.current_page == 'library':
        show_library_page()


# ────────────────────────────────────────────
# Page: Analyze
# ────────────────────────────────────────────
def show_analyze_page(api_keys):
    st.subheader("Document Upload & Configuration")

    up_col1, up_col2 = st.columns(2)

    with up_col1:
        uploaded_file = st.file_uploader(
            "Upload Main Research Paper",
            type=['pdf', 'txt'],
            help="Upload the primary document to analyze (PDF or TXT up to 200MB)"
        )

    with up_col2:
        reference_files = st.file_uploader(
            "Upload Reference Papers (Optional)",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload reference papers to check plagiarism against"
        )

    col_title, col_action = st.columns([2, 1])

    with col_title:
        paper_title = st.text_input(
            "Paper Title (Optional but recommended)",
            placeholder="e.g. Attention Is All You Need",
            help="Improves reference matching, validation, and recommendations."
        )

    if uploaded_file is not None:
        st.success(f"✅ Successfully loaded **{uploaded_file.name}**")

        try:
            text_content = extract_text_from_file(uploaded_file)
            word_count = len(text_content.split())

            reference_texts = []
            if reference_files:
                for ref_file in reference_files:
                    try:
                        ref_text = extract_text_from_file(ref_file)
                        if ref_text:
                            reference_texts.append(ref_text)
                    except Exception as e:
                        st.warning(f"Could not read reference file {ref_file.name}: {e}")

            with col_action:
                st.write("")
                st.write("")
                if st.button("🚀 Start Full Analysis", disabled=st.session_state.analysis_running,
                             use_container_width=True):
                    if any(api_keys.values()):
                        run_analysis(text_content, api_keys,
                                     paper_title or uploaded_file.name, reference_texts)
                    else:
                        st.error("Please configure at least one API key or enable search services")

            st.metric("Total Words (Main Paper)", f"{word_count:,}")
            with st.expander("📋 Click to Preview Extracted Text"):
                st.text_area("Content",
                             text_content[:1500] + "..." if len(text_content) > 1500 else text_content,
                             height=250, disabled=True)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

    st.markdown("---")
    st.subheader("Analysis Dashboard")

    if st.session_state.analysis_running:
        show_analysis_progress()
    elif st.session_state.analysis_results:
        show_results(st.session_state.analysis_results)
    else:
        show_placeholder_dashboard()


# ────────────────────────────────────────────
# Page: Search
# ────────────────────────────────────────────
def show_search_page():
    st.subheader("🔎 Search Indexed Documents")

    search_col, filter_col = st.columns([3, 1])

    with search_col:
        query = st.text_input("Search query", placeholder="Enter keywords or natural language query...",
                                label_visibility="collapsed")

    with filter_col:
        min_novelty = st.slider("Min Novelty", 0.0, 10.0, 0.0, 0.5)

    if query:
        indexing = st.session_state.analyzer.indexing_service
        filters = {}
        if min_novelty > 0:
            filters['min_novelty'] = min_novelty

        results = indexing.search(query, filters=filters, top_k=10)

        if results:
            st.success(f"Found **{len(results)}** matching documents")
            for r in results:
                with st.container():
                    st.markdown(f"### {r['title']}")
                    meta = r.get('metadata', {})
                    analysis = r.get('analysis', {})

                    info_parts = []
                    if r.get('word_count'):
                        info_parts.append(f"📝 {r['word_count']:,} words")
                    if analysis.get('novelty', {}).get('score'):
                        info_parts.append(f"💡 Novelty: {analysis['novelty']['score']}/10")
                    if analysis.get('plagiarism', {}).get('percentage') is not None:
                        info_parts.append(f"🔍 Plagiarism: {analysis['plagiarism']['percentage']}%")
                    info_parts.append(f"📊 Relevance: {r['score']}")

                    st.caption(" · ".join(info_parts))

                    if r.get('preview'):
                        st.markdown(f"_{r['preview'][:200]}..._")

                    st.markdown("---")
        else:
            st.info("No documents match your search. Try different keywords or index more documents.")
    else:
        stats = st.session_state.analyzer.indexing_service.get_stats()
        st.info(f"📁 **{stats['total_documents']}** documents indexed. Enter a search query above.")


# ────────────────────────────────────────────
# Page: Document Library
# ────────────────────────────────────────────
def show_library_page():
    st.subheader("📚 Document Library")

    indexing = st.session_state.analyzer.indexing_service
    documents = indexing.get_all_documents()

    if not documents:
        st.info("No documents indexed yet. Upload and analyze a paper to add it to the library.")
        return

    st.success(f"**{len(documents)}** documents in library")

    for doc in documents:
        with st.expander(f"📄 {doc['title']}", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"📝 {doc.get('word_count', 0):,} words")
            with col2:
                st.caption(f"📅 Indexed: {doc.get('indexed_at', '')[:10]}")
            with col3:
                analysis = doc.get('analysis', {})
                if analysis.get('novelty', {}).get('score'):
                    st.caption(f"💡 Novelty: {analysis['novelty']['score']}/10")

            # Show similar documents
            similar = indexing.get_similar_documents(doc['id'], top_k=3)
            if similar:
                st.markdown("**Similar Documents:**")
                for sim in similar:
                    st.markdown(f"- {sim['title']} (similarity: {sim['similarity']:.0%})")


# ────────────────────────────────────────────
# Analysis Functions
# ────────────────────────────────────────────
def test_apis(api_keys: Dict[str, str]):
    status_container = st.empty()
    with status_container.container():
        st.subheader("API Status")
        services = [
            ('Gemini', api_keys.get('gemini_api_key')),
            ('Wikipedia', api_keys.get('wikipedia_enabled')),
            ('DuckDuckGo', api_keys.get('duckduckgo_enabled'))
        ]
        for service, key in services:
            if key:
                st.success(f"✅ {service}: Configured")
            else:
                st.warning(f"⚠️ {service}: Not configured")


def run_analysis(text: str, api_keys: Dict[str, str], title: str, reference_texts: list = None):
    st.session_state.analysis_running = True
    analyzer = st.session_state.analyzer

    with st.spinner("🔍 Running full multi-agent analysis... This may take 30–60 seconds."):
        results = analyzer.analyze_paper(text, api_keys, title, reference_texts)

    st.session_state.analysis_results = results
    st.session_state.analysis_running = False
    st.rerun()


def show_analysis_progress():
    st.info("🔄 Analysis in progress...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    stages = ["Extracting text", "Detecting structure", "Extracting metadata",
              "Checking plagiarism", "Detecting AI content", "Calculating novelty",
              "Finding references", "Generating summaries", "Classifying entities",
              "Validating metadata", "Generating recommendations", "Indexing document"]
    for i, stage in enumerate(stages):
        progress_bar.progress((i + 1) / len(stages))
        status_text.text(f"{stage}...")


def show_results(results: Dict[str, Any]):
    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        plag = results.get('plagiarism', {}).get('percentage', 0)
        st.metric("🔍 Plagiarism", f"{plag:.1f}%", help="Target: ≤ 5%")
        if plag <= 5:
            st.success("✅ Acceptable")
        else:
            st.error("❌ High")

    with col2:
        ai_pct = results.get('ai_content', {}).get('percentage', 0)
        st.metric("🤖 AI Content", f"{ai_pct:.1f}%", help="Target: ≤ 0%")
        if ai_pct == 0:
            st.success("✅ None")
        elif ai_pct < 20:
            st.warning("⚠️ Low")
        else:
            st.error("❌ High")

    with col3:
        novelty = results.get('novelty', {}).get('score', 0)
        st.metric("💡 Novelty", f"{novelty:.1f}/10", help="Higher is more novel")
        if novelty >= 7:
            st.success("✅ High")
        elif novelty >= 5:
            st.info("ℹ️ Moderate")
        else:
            st.warning("⚠️ Low")

    with col4:
        ref_count = results.get('references', {}).get('count', 0)
        st.metric("📚 References", f"{ref_count}", help="Related papers found")

    with col5:
        trust = results.get('validation', {}).get('trust_score', 0)
        st.metric("✅ Trust Score", f"{trust:.0%}", help="Metadata validation score")
        if trust >= 0.8:
            st.success("✅ Verified")
        elif trust >= 0.5:
            st.warning("⚠️ Partial")
        else:
            st.error("❌ Low")

    st.markdown("---")

    # Tabs for detailed results
    tab_names = ["📝 Summary", "📋 Structure", "🏷️ Metadata", "🔍 Plagiarism",
                 "🤖 AI Content", "💡 Novelty", "📚 References", "🧬 Entities",
                 "✅ Validation", "💡 Recommendations"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        show_summary_details(results.get('summarization', {}))

    with tabs[1]:
        show_structure_details(results.get('structure', {}))

    with tabs[2]:
        show_metadata_details(results.get('metadata', {}))

    with tabs[3]:
        show_plagiarism_details(results.get('plagiarism', {}))

    with tabs[4]:
        show_ai_content_details(results.get('ai_content', {}))

    with tabs[5]:
        show_novelty_details(results.get('novelty', {}))

    with tabs[6]:
        show_references_details(results.get('references', {}))

    with tabs[7]:
        show_entity_details(results.get('entities', {}))

    with tabs[8]:
        show_validation_details(results.get('validation', {}))

    with tabs[9]:
        show_recommendation_details(results.get('recommendations', {}))


# ────────────────────────────────────────────
# Detail Views
# ────────────────────────────────────────────
def show_summary_details(data: Dict[str, Any]):
    st.subheader("📝 Paper Summaries")
    if data.get('status') == 'failed':
        st.error(f"Summarization failed: {data.get('error', 'Unknown error')}")
        return

    # Method & guard badges
    method = data.get('method', 'unknown')
    guard = "🛡️ Hallucination guard: ON" if data.get('hallucination_guard') else ""
    st.caption(f"Method: `{method}` · {guard}")

    # Summary Heading (faithful to paper)
    heading = data.get('summary_heading', 'Research Paper Summary')
    st.markdown(f"## {heading}")

    # Section availability matrix
    available = data.get('available_sections', {})
    if available:
        with st.expander("📋 Source Section Availability", expanded=False):
            avail_cols = st.columns(4)
            section_names = list(available.keys())
            for idx, sec_name in enumerate(section_names):
                col = avail_cols[idx % 4]
                with col:
                    icon = "✅" if available[sec_name] else "❌"
                    col.markdown(f"{icon} {sec_name.replace('_', ' ').title()}")

    # TL;DR
    st.markdown("### TL;DR")
    st.info(data.get('tldr', 'Not available'))

    # Executive Summary
    st.markdown("### Executive Summary")
    exec_summary = data.get('executive_summary', 'Not available')
    if exec_summary and exec_summary != 'Not available':
        st.write(exec_summary)
    else:
        st.warning("Executive summary could not be generated. Verify the paper has sufficient content.")

    # Detailed Summary
    with st.expander("📖 Detailed Summary", expanded=False):
        detailed = data.get('detailed_summary', 'Not available')
        if detailed and detailed != 'Not available':
            st.write(detailed)
        else:
            st.warning("Detailed summary unavailable. The paper may lack structured sections.")

    # Key Findings (sourced from Results/Conclusion)
    findings = data.get('key_findings', [])
    if findings:
        st.markdown("### 🔑 Key Findings")
        st.caption("_Sourced from the paper's Results/Conclusion sections_")
        for i, finding in enumerate(findings, 1):
            st.markdown(f"**{i}.** {finding}")
    else:
        st.warning("No key findings could be extracted. The paper may lack a clear Results or Conclusion section.")

    # Source sections used
    source_sections = data.get('source_sections_used', '')
    if source_sections:
        st.markdown("---")
        st.caption(f"📄 **Source sections used for this summary:** {source_sections}")


def show_structure_details(data: Dict[str, Any]):
    st.subheader("📋 Document Structure")
    if data.get('status') == 'failed':
        st.error(f"Structure analysis failed: {data.get('error', 'Unknown error')}")
        return

    sections = data.get('sections', [])
    st.write(f"**{len(sections)} sections detected** | Avg confidence: {data.get('avg_confidence', 0):.0%}")

    # Section checklist
    col1, col2, col3 = st.columns(3)
    with col1:
        if data.get('has_abstract'):
            st.success("✅ Abstract found")
        else:
            st.warning("⚠️ No abstract detected")
    with col2:
        if data.get('has_methodology'):
            st.success("✅ Methodology found")
        else:
            st.warning("⚠️ No methodology detected")
    with col3:
        if data.get('has_references'):
            st.success("✅ References found")
        else:
            st.warning("⚠️ No references detected")

    # Section list
    for section in sections:
        conf = section.get('confidence', 0)
        conf_icon = "🟢" if conf >= 0.75 else ("🟡" if conf >= 0.5 else "🔴")
        with st.expander(f"{conf_icon} **{section['type'].title()}** — {section.get('heading', '')} ({section.get('word_count', 0)} words)"):
            st.caption(f"Confidence: {conf:.0%} | Line: {section.get('line_number', '?')}")
            content = section.get('content', '')
            if content:
                st.text(content[:500] + ("..." if len(content) > 500 else ""))


def show_metadata_details(data: Dict[str, Any]):
    st.subheader("🏷️ Extracted Metadata")
    if data.get('status') == 'failed':
        st.error(f"Metadata extraction failed: {data.get('error', 'Unknown error')}")
        return

    # Completeness bar
    completeness = data.get('completeness_score', 0)
    st.progress(completeness, text=f"Metadata completeness: {completeness:.0%}")

    # ── Core metadata in structured format ──
    st.markdown("### 📄 Core Fields")

    # Title
    title = data.get('title', 'Unknown')
    st.markdown(f"**Title:** {title}")

    # Authors (as listed)
    authors = data.get('authors', [])
    if authors:
        author_names = [a['name'] if isinstance(a, dict) else str(a) for a in authors]
        st.markdown(f"**Authors:** {', '.join(author_names)}")
    else:
        st.markdown("**Authors:** ⚠️ Not detected")

    # Emails (validated)
    emails = data.get('email_addresses', [])
    if emails:
        st.markdown(f"**Emails:** {', '.join(emails)}")
    else:
        st.markdown("**Emails:** ⚠️ None found")

    # Show email validation issues
    email_issues = data.get('email_validation', [])
    if email_issues:
        with st.expander("⚠️ Email Validation Issues"):
            for issue in email_issues:
                st.markdown(f"- {issue}")

    col1, col2 = st.columns(2)
    with col1:
        if data.get('doi'):
            st.markdown(f"**DOI:** [{data['doi']}](https://doi.org/{data['doi']})")
        else:
            st.markdown("**DOI:** ⚠️ Not found")

        if data.get('publication_date'):
            st.markdown(f"**Date:** {data['publication_date']}")
        else:
            st.markdown("**Date:** Not found")

        if data.get('venue'):
            st.markdown(f"**Venue:** {data['venue']}")

    with col2:
        st.markdown(f"**Language:** {data.get('language', 'Unknown')}")
        st.markdown(f"**Word Count:** {data.get('word_count', 0):,}")
        st.markdown(f"**Est. Pages:** {data.get('page_estimate', 0)}")

    # Keywords
    keywords = data.get('keywords', [])
    if keywords:
        st.markdown("**Keywords:**")
        st.markdown(" · ".join([f"`{kw}`" for kw in keywords]))

    # Abstract
    abstract = data.get('abstract', '')
    if abstract:
        with st.expander("📄 Abstract", expanded=False):
            st.write(abstract)

    # ── Author ↔ Email Mapping ──
    mapping = data.get('author_email_mapping', [])
    if mapping:
        st.markdown("### 👤 Author–Email Mapping")
        for entry in mapping:
            conf = entry.get('confidence', 'none')
            icon = {'high': '✅', 'medium': '🟡', 'none': '❌'}.get(conf, '❔')
            email_str = entry.get('email', 'Not found')
            affil = entry.get('affiliation', '')
            line = f"{icon} **{entry.get('author', '?')}** → `{email_str}`"
            if affil:
                line += f" · _{affil}_"
            st.markdown(line)

    # ── Integrity Notes (discrepancy detection) ──
    integrity_notes = data.get('integrity_notes', [])
    if integrity_notes:
        st.markdown("### 🔍 Metadata Integrity Notes")

        severity_icons = {'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}

        for note in integrity_notes:
            severity = note.get('severity', 'info')
            icon = severity_icons.get(severity, 'ℹ️')
            field = note.get('field', '').title()
            message = note.get('message', '')
            suggestion = note.get('suggestion', '')

            st.markdown(f"{icon} **[{field}]** {message}")
            if suggestion:
                st.caption(f"   💡 {suggestion}")

    # ── Unavailable Sections ──
    unavailable = data.get('unavailable_sections', [])
    if unavailable:
        st.markdown("### ❓ Unavailable Metadata")
        for entry in unavailable:
            st.markdown(f"- **{entry['field'].title()}**: {entry['message']}")
            st.caption(f"  🔎 How to verify: {entry['how_to_verify']}")


def show_plagiarism_details(data: Dict[str, Any]):
    st.subheader("🔍 Plagiarism Analysis")
    if data.get('status') == 'failed':
        st.error(f"Analysis failed: {data.get('error', 'Unknown error')}")
        return

    percentage = data.get('percentage', 0)

    if percentage <= 5:
        st.success(f"✅ Plagiarism level ({percentage:.1f}%) is within acceptable range")
    else:
        st.error(f"❌ High plagiarism detected ({percentage:.1f}%)")

    # Sources checked
    sources = data.get('sources_checked', [])
    if sources:
        st.caption(f"Sources checked: {', '.join(sources)}")

    details = data.get('details', [])
    if details:
        st.write("**Potential matches found:**")
        for detail in details[:10]:
            st.write(f"- {detail}")


def show_ai_content_details(data: Dict[str, Any]):
    st.subheader("🤖 AI Content Detection")
    if data.get('status') == 'failed':
        st.error(f"Analysis failed: {data.get('error', 'Unknown error')}")
        return

    percentage = data.get('percentage', 0)
    confidence = data.get('confidence', 0)

    if percentage == 0:
        st.success("✅ No AI-generated content detected")
    elif percentage < 20:
        st.warning(f"⚠️ {percentage:.1f}% of content may be AI-generated")
    else:
        st.error(f"❌ {percentage:.1f}% of content appears AI-generated")

    st.write(f"**Confidence Level:** {confidence:.1f}/10")
    st.write(f"**Method:** {data.get('method', 'unknown')}")

    indicators = data.get('indicators', [])
    if indicators:
        st.write("**Indicators found:**")
        for ind in indicators:
            st.write(f"- {ind}")


def show_novelty_details(data: Dict[str, Any]):
    st.subheader("💡 Novelty Assessment")
    if data.get('status') == 'failed':
        st.error(f"Analysis failed: {data.get('error', 'Unknown error')}")
        return

    score = data.get('score', 0)
    st.write(f"**Overall Novelty Score:** {score:.1f}/10")

    # Breakdown
    breakdown = data.get('breakdown', {})
    if breakdown:
        st.markdown("**Score Breakdown:**")
        bd_cols = st.columns(4)
        with bd_cols[0]:
            st.metric("Terminology", f"{breakdown.get('terminology', 0):.1f}")
        with bd_cols[1]:
            st.metric("Methodology", f"{breakdown.get('methodology', 0):.1f}")
        with bd_cols[2]:
            st.metric("Concepts", f"{breakdown.get('concepts', 0):.1f}")
        with bd_cols[3]:
            st.metric("Citations", f"{breakdown.get('citations', 0):.1f}")

    factors = data.get('factors', [])
    if factors:
        st.write("**Contributing Factors:**")
        for factor in factors:
            st.write(f"- {factor}")


def show_references_details(data: Dict[str, Any]):
    st.subheader("📚 References & Related Work")
    if data.get('status') == 'failed':
        st.error(f"Analysis failed: {data.get('error', 'Unknown error')}")
        return

    links = data.get('links', [])
    if links:
        st.write(f"**Found {len(links)} related sources:**")
        for i, link in enumerate(links[:15], 1):
            title = link.get('title', 'Unknown Title')
            url = link.get('url', '#')
            description = link.get('description', 'No description available')
            source = link.get('source', '')

            st.markdown(f"**{i}. [{title}]({url})** `{source}`")
            st.caption(description[:200])
    else:
        st.info("No related references found. Try enabling more search services.")


def show_entity_details(data: Dict[str, Any]):
    st.subheader("🧬 Entity Classification")
    if data.get('status') == 'failed':
        st.error(f"Entity classification failed: {data.get('error', 'Unknown error')}")
        return

    st.write(f"**{data.get('entity_count', 0)} entities detected**")
    st.caption(data.get('summary', ''))

    grouped = data.get('grouped', {})

    entity_colors = {
        'method': '🔵', 'dataset': '🟢', 'metric': '🟡',
        'tool': '🟣', 'acronym': '🔷', 'organization': '🔴', 'person': '🔘'
    }

    for entity_type, entities in grouped.items():
        icon = entity_colors.get(entity_type, '⚪')
        with st.expander(f"{icon} **{entity_type.title()}** ({len(entities)} found)"):
            for entity in entities:
                conf = entity.get('confidence', 0)
                method = entity.get('method', 'unknown')
                st.markdown(f"- `{entity['text']}` — confidence: {conf:.0%} ({method})")


def show_validation_details(data: Dict[str, Any]):
    st.subheader("✅ Validation & Verification")
    if data.get('status') == 'failed':
        st.error(f"Validation failed: {data.get('error', 'Unknown error')}")
        return

    trust = data.get('trust_score', 0)
    st.progress(trust, text=f"Trust Score: {trust:.0%}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ Passed", data.get('passed', 0))
    with col2:
        st.metric("⚠️ Warnings", data.get('warnings', 0))
    with col3:
        st.metric("❌ Failed", data.get('failures', 0))

    checks = data.get('checks', [])
    for check in checks:
        status = check.get('status', 'unknown')
        icon = {'pass': '✅', 'warn': '⚠️', 'fail': '❌'}.get(status, '❔')
        msg = check.get('message', '')
        source = check.get('source', '')

        st.markdown(f"{icon} **{check.get('check', 'Unknown Check')}** — {msg} `{source}`")

    # External data
    ext = data.get('external_data', {})
    if ext:
        with st.expander("🌐 External Data Retrieved"):
            for key, value in ext.items():
                if isinstance(value, list):
                    st.markdown(f"**{key.title()}:** {', '.join(str(v) for v in value)}")
                else:
                    st.markdown(f"**{key.title()}:** {value}")


def show_recommendation_details(data: Dict[str, Any]):
    st.subheader("💡 Recommended Papers")
    if data.get('status') == 'failed':
        st.error(f"Recommendations failed: {data.get('error', 'Unknown error')}")
        return

    recs = data.get('recommendations', [])
    methods = data.get('methods_used', [])

    if methods:
        st.caption(f"Sources: {', '.join(methods)}")

    if recs:
        st.write(f"**{len(recs)} recommendations found:**")
        for i, rec in enumerate(recs, 1):
            title = rec.get('title', 'Unknown')
            url = rec.get('url', '')
            abstract = rec.get('abstract', '')
            source = rec.get('source', '')
            reason = rec.get('reason', '')
            relevance = rec.get('relevance', 0)
            year = rec.get('year', '')
            citations = rec.get('citations', '')

            if url:
                st.markdown(f"**{i}. [{title}]({url})** `{source}`")
            else:
                st.markdown(f"**{i}. {title}** `{source}`")

            info_parts = []
            if year:
                info_parts.append(f"📅 {year}")
            if citations:
                info_parts.append(f"📊 {citations} citations")
            info_parts.append(f"🎯 Relevance: {relevance:.0%}")

            if info_parts:
                st.caption(" · ".join(info_parts))

            if reason:
                st.caption(f"💡 {reason}")

            if abstract:
                st.caption(f"_{abstract[:150]}..._" if len(abstract) > 150 else f"_{abstract}_")

            st.markdown("")
    else:
        st.info("No recommendations available. Try providing a paper title for better results.")


def show_placeholder_dashboard():
    st.info("👆 Upload a research paper and start analysis to see results here")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("🔍 Plagiarism", "-%", help="Target: ≤ 5%")
    with col2:
        st.metric("🤖 AI Content", "-%", help="Target: ≤ 0%")
    with col3:
        st.metric("💡 Novelty", "-/10", help="Higher is more novel")
    with col4:
        st.metric("📚 References", "-", help="Related papers and sources")
    with col5:
        st.metric("✅ Trust", "-%", help="Metadata validation")


if __name__ == "__main__":
    main()