import re
import logging
from typing import Dict, Any, List

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class StructureAnalyzerAgent(BaseAgent):
    """Agent for detecting document structure and section boundaries."""

    def __init__(self):
        super().__init__("StructureAnalyzerAgent")
        self.section_patterns = self._load_section_patterns()

    def analyze(self, text: str, api_keys: Dict[str, str], **kwargs) -> Dict[str, Any]:
        """Detect document sections and return a structured outline."""

        sections = self._detect_sections(text)

        # If Gemini API is available, refine section detection
        if api_keys.get('gemini_api_key') and len(sections) < 3:
            try:
                sections = self._gemini_section_detection(text, api_keys['gemini_api_key'])
            except Exception as e:
                logger.warning(f"Gemini section detection failed: {e}")

        # Calculate confidence
        avg_confidence = sum(s['confidence'] for s in sections) / max(len(sections), 1)

        return {
            'sections': sections,
            'section_count': len(sections),
            'avg_confidence': round(avg_confidence, 2),
            'has_abstract': any(s['type'] == 'abstract' for s in sections),
            'has_references': any(s['type'] == 'references' for s in sections),
            'has_methodology': any(s['type'] in ('methodology', 'methods') for s in sections),
        }

    def _detect_sections(self, text: str) -> List[Dict[str, Any]]:
        """Detect sections using regex-based heuristics."""
        sections = []
        lines = text.split('\n')

        current_pos = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                current_pos += len(line) + 1
                continue

            section_type = self._classify_heading(stripped)
            if section_type:
                # Find the content until next section
                content_lines = []
                for j in range(i + 1, len(lines)):
                    next_stripped = lines[j].strip()
                    if next_stripped and self._classify_heading(next_stripped):
                        break
                    content_lines.append(lines[j])

                content = '\n'.join(content_lines).strip()
                confidence = self._calculate_confidence(stripped, content, section_type)

                sections.append({
                    'type': section_type,
                    'heading': stripped,
                    'content': content[:2000],  # Limit content stored
                    'line_number': i + 1,
                    'word_count': len(content.split()),
                    'confidence': confidence
                })

            current_pos += len(line) + 1

        # If no sections detected, create a single "body" section
        if not sections:
            sections.append({
                'type': 'body',
                'heading': 'Document Body',
                'content': text[:2000],
                'line_number': 1,
                'word_count': len(text.split()),
                'confidence': 0.5
            })

        return sections

    def _classify_heading(self, line: str) -> str:
        """Classify a line as a section heading."""
        line_lower = line.lower().strip()

        # Remove numbering prefixes
        cleaned = re.sub(r'^[\d]+[.\)]\s*', '', line_lower)
        cleaned = re.sub(r'^[ivxlc]+[.\)]\s*', '', cleaned)

        for section_type, patterns in self.section_patterns.items():
            for pattern in patterns:
                if re.match(pattern, cleaned, re.IGNORECASE):
                    return section_type

        # Check if it looks like a heading (short, possibly uppercase)
        if len(line.split()) <= 6 and line.isupper() and len(line) > 3:
            return 'section'

        return None

    def _calculate_confidence(self, heading: str, content: str, section_type: str) -> float:
        """Calculate confidence score for section detection."""
        confidence = 0.5

        # Boost for standard section names
        standard_types = ['abstract', 'introduction', 'methodology', 'methods',
                          'results', 'discussion', 'conclusion', 'references']
        if section_type in standard_types:
            confidence += 0.3

        # Boost for content length matching expectations
        word_count = len(content.split())
        if section_type == 'abstract' and 50 <= word_count <= 400:
            confidence += 0.15
        elif section_type == 'references' and word_count > 100:
            confidence += 0.15
        elif word_count > 50:
            confidence += 0.05

        return min(1.0, confidence)

    def _gemini_section_detection(self, text: str, api_key: str) -> List[Dict[str, Any]]:
        """Use Gemini API for enhanced section detection."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"

        prompt = f"""Analyze this academic paper and identify all major sections.
For each section, provide:
- section_type (one of: title, abstract, introduction, literature_review, methodology, results, discussion, conclusion, references, acknowledgments, appendix)
- heading (the actual heading text)
- approximate_line (estimated line number)

Paper text (first 3000 chars):
{text[:3000]}

Respond as a JSON array of objects."""

        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            response = self._make_api_request(url, json_data=payload)
            content = response['candidates'][0]['content']['parts'][0]['text']

            import json
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return [{
                    'type': s.get('section_type', 'section'),
                    'heading': s.get('heading', 'Unknown'),
                    'content': '',
                    'line_number': s.get('approximate_line', 0),
                    'word_count': 0,
                    'confidence': 0.85
                } for s in parsed]
        except Exception:
            pass

        return self._detect_sections(text)

    def _load_section_patterns(self) -> Dict[str, List[str]]:
        """Load regex patterns for section heading detection."""
        return {
            'abstract': [r'^abstract\b', r'^summary\b'],
            'introduction': [r'^introduction\b', r'^background\b', r'^overview\b'],
            'literature_review': [r'^(literature\s+review|related\s+work|prior\s+work)\b'],
            'methodology': [r'^(methodology|methods?|approach|proposed\s+(method|approach|system))\b',
                            r'^(experimental\s+setup|research\s+design)\b'],
            'results': [r'^(results?|findings|experimental\s+results?)\b'],
            'discussion': [r'^discussion\b', r'^analysis\b'],
            'conclusion': [r'^(conclusions?|concluding\s+remarks?|future\s+work)\b'],
            'references': [r'^(references?|bibliography|works\s+cited)\b'],
            'acknowledgments': [r'^(acknowledgm?ents?|funding)\b'],
            'appendix': [r'^(appendix|appendices|supplementary)\b'],
        }
