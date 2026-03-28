import re
import logging
from typing import Dict, Any, List

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SummarizationAgent(BaseAgent):
    """Agent for generating multi-level, header-accurate document summaries.
    
    Design principles:
    - Summary headings are faithful to the paper's actual content/sections
    - Key findings reflect the paper's own conclusions, not hallucinated content
    - Source sections are cited where possible
    - Missing sections are flagged explicitly
    """

    def __init__(self):
        super().__init__("SummarizationAgent")

    def analyze(self, text: str, api_keys: Dict[str, str], **kwargs) -> Dict[str, Any]:
        """Generate summaries at multiple granularity levels."""

        # Detect what sections actually exist in the paper
        available_sections = self._detect_available_sections(text)

        # Try LLM-based summarization first
        if api_keys.get('gemini_api_key'):
            try:
                result = self._gemini_summarize(text, api_keys['gemini_api_key'], available_sections)
                result['available_sections'] = available_sections
                return result
            except Exception as e:
                logger.warning(f"Gemini summarization failed: {e}")

        # Fallback: extractive summarization
        result = self._extractive_summarize(text, available_sections)
        result['available_sections'] = available_sections
        return result

    def _detect_available_sections(self, text: str) -> Dict[str, bool]:
        """Detect which major sections exist in the paper for faithful summarization."""
        text_lower = text.lower()
        sections = {
            'abstract': bool(re.search(r'\babstract\b', text_lower)),
            'introduction': bool(re.search(r'\bintroduction\b', text_lower)),
            'related_work': bool(re.search(r'\b(related\s+work|literature\s+review|prior\s+work)\b', text_lower)),
            'methodology': bool(re.search(r'\b(methodology|methods?|approach|proposed\s+(method|system|framework))\b', text_lower)),
            'results': bool(re.search(r'\b(results?|experiments?|evaluation)\b', text_lower)),
            'discussion': bool(re.search(r'\bdiscussion\b', text_lower)),
            'conclusion': bool(re.search(r'\b(conclusions?|concluding)\b', text_lower)),
            'references': bool(re.search(r'\breferences?\b', text_lower)),
        }
        return sections

    def _gemini_summarize(self, text: str, api_key: str, available_sections: Dict[str, bool]) -> Dict[str, Any]:
        """Use Gemini API for abstractive summarization."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"

        # Build section-aware constraints
        section_list = [s for s, present in available_sections.items() if present]
        section_note = f"The paper contains these sections: {', '.join(section_list)}." if section_list else ""

        prompt = f"""You are summarizing a research paper. Follow these strict rules:

RULES:
1. Do NOT hallucinate content. Only summarize what is actually in the paper.
2. Your summary headings must reflect the paper's actual content.
3. Key Findings MUST come from the paper's Results/Conclusion sections.
4. If a section is missing, say "Not available in source document."
5. Cite the section name when referencing specific content (e.g., "As stated in the Methodology section...").

{section_note}

Format your response EXACTLY as:

SUMMARY HEADING: [A concise heading that captures the paper's main contribution — do not invent, use the paper's own terminology]

TL;DR: [One sentence, max 30 words, faithful to the paper]

EXECUTIVE SUMMARY:
[100-200 word faithful summary covering: objective, methodology, key results. Cite sections.]

DETAILED SUMMARY:
[400-500 word summary covering all major sections. Cite each section by name.]

KEY FINDINGS:
- [Finding from Results/Conclusion section]
- [Finding from Results/Conclusion section]
- [Finding from Results/Conclusion section]
- [Finding if available]
- [Finding if available]

SOURCE SECTIONS USED: [List which paper sections you drew from]

Paper text:
{text[:6000]}
"""

        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        response = self._make_api_request(url, json_data=payload)
        content = response['candidates'][0]['content']['parts'][0]['text']

        # Parse structured response
        summary_heading = self._extract_field(content, r'SUMMARY HEADING:?\s*(.+?)(?:\n|$)')
        tldr = self._extract_field(content, r'TL;?DR:?\s*(.+?)(?:\n\n|\nEXECUTIVE SUMMARY|$)')
        executive = self._extract_field(
            content,
            r'EXECUTIVE SUMMARY:?\s*(.+?)(?:\n\n\s*DETAILED|\nDETAILED)',
        )
        detailed = self._extract_field(
            content,
            r'DETAILED SUMMARY:?\s*(.+?)(?:\n\n\s*KEY|\nKEY)',
        )
        key_findings = self._extract_findings(content)
        source_sections = self._extract_field(content, r'SOURCE SECTIONS USED:?\s*(.+?)(?:\n\n|\Z)')

        return {
            'summary_heading': summary_heading or 'Research Paper Summary',
            'tldr': tldr or 'Summary not available.',
            'executive_summary': executive or 'Executive summary not available.',
            'detailed_summary': detailed or 'Detailed summary not available.',
            'key_findings': key_findings,
            'source_sections_used': source_sections or 'Not specified',
            'method': 'gemini',
            'hallucination_guard': True,
        }

    def _extractive_summarize(self, text: str, available_sections: Dict[str, bool]) -> Dict[str, Any]:
        """Extractive fallback with section-aware sentence scoring."""
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 30]

        if not sentences:
            return {
                'summary_heading': 'Document Summary',
                'tldr': 'Document too short to summarize.',
                'executive_summary': 'Document too short to generate an executive summary.',
                'detailed_summary': text[:500] if text else 'No content available.',
                'key_findings': [],
                'source_sections_used': 'None — document too short',
                'method': 'extractive_fallback',
                'hallucination_guard': True,
            }

        # ── Section-aware sentence scoring ──
        # Identify conclusion/results zone
        text_lower = text.lower()
        conclusion_start = max(
            text_lower.rfind('conclusion'),
            text_lower.rfind('results'),
            text_lower.rfind('findings')
        )
        abstract_end = text_lower.find('introduction')
        if abstract_end == -1:
            abstract_end = min(len(text) // 10, 2000)

        scored = []
        for i, sentence in enumerate(sentences):
            score = 0.0
            words = sentence.split()
            sent_lower = sentence.lower()

            # Position in document
            char_pos = text.find(sentence[:50])

            # Abstract zone bonus
            if char_pos >= 0 and char_pos < abstract_end:
                score += 4.0

            # Conclusion/results zone bonus (key findings source)
            if conclusion_start > 0 and char_pos >= 0 and char_pos > conclusion_start:
                score += 3.5

            # Length preference: medium sentences
            if 12 <= len(words) <= 45:
                score += 1.5

            # Importance keywords
            importance_words = {
                'propose': 2, 'novel': 2, 'result': 1.5, 'show': 1, 'demonstrate': 1.5,
                'improve': 1.5, 'achieve': 1.5, 'outperform': 2, 'contribute': 1.5,
                'finding': 1.5, 'conclude': 2, 'significant': 1, 'approach': 1,
                'method': 1, 'framework': 1, 'accuracy': 1, 'performance': 1
            }
            for word, weight in importance_words.items():
                if word in sent_lower:
                    score += weight

            # Penalize citations, urls, figure references
            if re.search(r'\[\d+\]|http|fig\.|table\s+\d', sentence, re.IGNORECASE):
                score -= 1.5

            # Penalize very short or excessively long sentences
            if len(words) < 8 or len(words) > 60:
                score -= 2

            scored.append({
                'sentence': sentence,
                'score': score,
                'in_conclusion': (conclusion_start > 0 and char_pos >= 0 and char_pos > conclusion_start),
                'in_abstract': (char_pos >= 0 and char_pos < abstract_end),
            })

        # Sort by score
        scored.sort(key=lambda x: x['score'], reverse=True)

        # Build summaries
        top = [s['sentence'] for s in scored]
        conclusion_sentences = [s['sentence'] for s in scored if s['in_conclusion']]

        tldr = top[0] if top else 'No summary available.'

        # Executive: top 3 sentences, re-ordered by position in document
        exec_sents = sorted(top[:4], key=lambda s: text.find(s[:50]))
        executive = '. '.join(exec_sents) + '.'

        # Detailed: top 8 sentences, re-ordered
        detail_sents = sorted(top[:10], key=lambda s: text.find(s[:50]))
        detailed = '. '.join(detail_sents) + '.'

        # Key findings: prefer conclusion/results sentences
        findings = conclusion_sentences[:5] if conclusion_sentences else top[:5]

        # Determine which sections were used
        used = []
        if any(s.get('in_abstract') for s in scored[:10]):
            used.append('Abstract')
        if any(s.get('in_conclusion') for s in scored[:10]):
            used.append('Conclusion/Results')
        used.append('General body text')

        # Generate a faithful heading from the top sentence
        heading_words = tldr.split()[:8]
        summary_heading = ' '.join(heading_words) + ('...' if len(heading_words) == 8 else '')

        return {
            'summary_heading': summary_heading,
            'tldr': tldr,
            'executive_summary': executive,
            'detailed_summary': detailed,
            'key_findings': findings,
            'source_sections_used': ', '.join(used),
            'method': 'extractive_fallback',
            'hallucination_guard': True,
        }

    # ──────────────────────────────────────────
    # Parsing helpers
    # ──────────────────────────────────────────
    def _extract_field(self, content: str, pattern: str) -> str:
        """Extract a field from formatted text using regex."""
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ''

    def _extract_findings(self, content: str) -> List[str]:
        """Extract key findings bullet points."""
        # First try to isolate the KEY FINDINGS section
        key_section = re.search(r'KEY FINDINGS:?\s*(.+?)(?:\n\n\s*SOURCE|\nSOURCE|\Z)',
                                content, re.DOTALL | re.IGNORECASE)
        search_text = key_section.group(1) if key_section else content

        bullets = re.findall(r'[-•*]\s*(.+?)(?:\n|$)', search_text)
        # Also try numbered findings
        if not bullets:
            bullets = re.findall(r'\d+[.)]\s*(.+?)(?:\n|$)', search_text)

        return [b.strip() for b in bullets[:6] if len(b.strip()) > 10]
