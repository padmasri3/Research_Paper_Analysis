import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class MetadataExtractorAgent(BaseAgent):
    """Agent for extracting structured metadata from research papers with
    validation, discrepancy detection, and integrity reporting."""

    def __init__(self):
        super().__init__("MetadataExtractorAgent")

    def analyze(self, text: str, api_keys: Dict[str, str], **kwargs) -> Dict[str, Any]:
        """Extract metadata: title, authors, abstract, keywords, DOI, etc.
        Produces integrity notes flagging missing or ambiguous fields."""

        # ── Phase 1: Regex-based extraction ──
        metadata = {
            'title': self._extract_title(text),
            'authors': self._extract_authors(text),
            'abstract': self._extract_abstract(text),
            'keywords': self._extract_keywords(text),
            'doi': self._extract_doi(text),
            'email_addresses': self._extract_emails(text),
            'publication_date': self._extract_date(text),
            'venue': self._extract_venue(text),
            'word_count': len(text.split()),
            'page_estimate': max(1, len(text) // 3000),
            'language': self._detect_language(text),
        }

        # ── Phase 2: Gemini‑enhanced extraction ──
        gemini_used = False
        if api_keys.get('gemini_api_key'):
            try:
                gemini_metadata = self._gemini_extract(text, api_keys['gemini_api_key'])
                gemini_used = True
                # Merge: prefer Gemini for fields it found when regex came up empty
                for key, value in gemini_metadata.items():
                    if value and (not metadata.get(key) or metadata.get(key) in ('Unknown', '', [], None)):
                        metadata[key] = value
            except Exception as e:
                logger.warning(f"Gemini metadata extraction failed: {e}")

        # ── Phase 3: Validate emails ──
        raw_emails = metadata.get('email_addresses', [])
        validated_emails, email_issues = self._validate_emails(raw_emails)
        metadata['email_addresses'] = validated_emails
        metadata['email_validation'] = email_issues

        # ── Phase 4: Pair authors ↔ emails ──
        metadata['author_email_mapping'] = self._pair_authors_emails(
            metadata.get('authors', []),
            validated_emails,
            text
        )

        # ── Phase 5: Completeness score ──
        required_fields = ['title', 'authors', 'abstract', 'keywords', 'doi']
        filled = sum(1 for f in required_fields
                     if metadata.get(f) and metadata[f] not in ('Unknown', '', [], None))
        metadata['completeness_score'] = round(filled / len(required_fields), 2)

        # ── Phase 6: Integrity notes (discrepancy detection) ──
        metadata['integrity_notes'] = self._generate_integrity_notes(metadata, text, gemini_used)

        # ── Phase 7: Unavailable sections ──
        metadata['unavailable_sections'] = self._flag_unavailable(metadata)

        return metadata

    # ──────────────────────────────────────────
    # Extraction Methods
    # ──────────────────────────────────────────
    def _extract_title(self, text: str) -> str:
        """Extract paper title from text."""
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if not lines:
            return 'Unknown'

        skip_patterns = ['journal', 'volume', 'issn', 'doi:', 'http', 'arxiv',
                         'proceedings', 'copyright', '©', 'page ', 'pp.']

        for line in lines[:8]:
            if any(skip in line.lower() for skip in skip_patterns):
                continue
            # Title heuristic: reasonably long, not a date/number line
            if 10 < len(line) < 300 and not re.match(r'^[\d\s/\-:.]+$', line):
                return line.strip()

        return lines[0][:200] if lines else 'Unknown'

    def _extract_authors(self, text: str) -> List[Dict[str, str]]:
        """Extract author names and affiliations."""
        authors = []
        header = text[:2500]

        # Locate the region between title and abstract
        abstract_pos = re.search(r'\babstract\b', header, re.IGNORECASE)
        search_area = header[:abstract_pos.start()] if abstract_pos else header[:1000]

        # Pattern: "Firstname [M.] Lastname" with optional superscript markers
        author_patterns = [
            # Standard: John A. Smith
            r'([A-Z][a-z]+\s+(?:[A-Z]\.?\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?:\s*[\d,\*†‡§]+)?',
        ]

        skip_words = {'abstract', 'introduction', 'university', 'department', 'journal',
                      'volume', 'research', 'paper', 'received', 'accepted', 'published',
                      'science', 'engineering', 'technology', 'international', 'conference',
                      'proceedings', 'copyright', 'keywords', 'available', 'submitted'}

        for pattern in author_patterns:
            matches = re.findall(pattern, search_area)
            for match in matches:
                name = match.strip()
                words = name.split()
                if len(words) < 2 or len(name) > 60:
                    continue
                if any(w.lower() in skip_words for w in words):
                    continue
                # Must start with capital
                if not all(w[0].isupper() for w in words if len(w) > 1):
                    continue

                authors.append({'name': name, 'affiliation': '', 'email': ''})

        # Extract affiliations
        affil_patterns = [
            r'(?:Department|University|Institute|School|Laboratory|College|Faculty|Center|Centre)\s+(?:of\s+)?[^\n]{5,120}',
        ]
        for pattern in affil_patterns:
            matches = re.findall(pattern, header, re.IGNORECASE)
            for i, match in enumerate(matches):
                if i < len(authors):
                    authors[i]['affiliation'] = match.strip()

        # Deduplicate
        seen = set()
        unique_authors = []
        for author in authors:
            normalized = author['name'].lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique_authors.append(author)

        return unique_authors[:12]

    def _extract_abstract(self, text: str) -> str:
        """Extract abstract from paper."""
        patterns = [
            r'(?:^|\n)\s*abstract\s*[:\-—–.]*\s*\n?\s*(.*?)(?=\n\s*(?:keywords?|index\s+terms?|introduction|1[\.\)]\s|I\.\s))',
            r'(?:^|\n)\s*abstract\s*[:\-—–.]*\s*\n?\s*(.*?)(?:\n\s*\n)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                abstract = re.sub(r'\s+', ' ', abstract)  # Collapse whitespace
                if 20 < len(abstract.split()) < 600:
                    return abstract
        return ''

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords/index terms from paper."""
        patterns = [
            r'(?:keywords?|index\s+terms?)\s*[:\-—–.]\s*(.*?)(?:\n\s*\n|\.\s*\n|(?=\b(?:introduction|1[\.\)]\s|I\.\s)))',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                kw_text = match.group(1).strip()
                keywords = re.split(r'[;,•·\n]', kw_text)
                keywords = [k.strip().strip('.').strip() for k in keywords
                            if k.strip() and len(k.strip()) > 2 and len(k.strip()) < 80]
                return keywords[:15]
        return []

    def _extract_doi(self, text: str) -> Optional[str]:
        """Extract DOI from text."""
        doi_pattern = r'(?:doi[:\s]*)?(?:https?://(?:dx\.)?doi\.org/)?(\b10\.\d{4,}/[^\s,;)\]]+)'
        match = re.search(doi_pattern, text, re.IGNORECASE)
        if match:
            doi = match.group(1).rstrip('.').rstrip(')')
            return doi
        return None

    def _extract_emails(self, text: str) -> List[str]:
        """Extract email addresses from the header region of the paper."""
        email_pattern = r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
        # Search in the header region (first 4000 chars for broader coverage)
        emails = re.findall(email_pattern, text[:4000])
        # Unique, preserve order
        seen = set()
        unique = []
        for e in emails:
            e_lower = e.lower()
            if e_lower not in seen:
                seen.add(e_lower)
                unique.append(e)
        return unique[:10]

    def _extract_date(self, text: str) -> Optional[str]:
        """Extract publication / received / accepted date."""
        patterns = [
            r'(?:received|accepted|published|revised)\s*[:\s]*(\d{1,2}\s+\w+\s+\d{4})',
            r'(?:received|accepted|published|revised)\s*[:\s]*(\w+\s+\d{1,2},?\s+\d{4})',
            r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b',
            r'\b(\d{4}[-/]\d{2}[-/]\d{2})\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, text[:4000], re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _extract_venue(self, text: str) -> Optional[str]:
        """Extract journal or conference name."""
        patterns = [
            r'(?:published\s+in|appears?\s+in|proceedings\s+of)\s*[:\s]*([^\n]{10,100})',
            r'(?:journal|conference)\s+(?:of\s+)?([^\n]{10,80})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text[:3000], re.IGNORECASE)
            if match:
                return match.group(1).strip().rstrip('.')
        return None

    def _detect_language(self, text: str) -> str:
        """Simple language detection."""
        sample = text[:1500].lower()
        english_words = ['the', 'and', 'of', 'to', 'in', 'is', 'for', 'that', 'with', 'this',
                         'are', 'was', 'an', 'by', 'on', 'be', 'as', 'at']
        english_count = sum(1 for w in english_words if f' {w} ' in sample)
        return 'English' if english_count >= 4 else 'Unknown'

    # ──────────────────────────────────────────
    # Email Validation
    # ──────────────────────────────────────────
    def _validate_emails(self, emails: List[str]) -> tuple:
        """Validate email format and return (valid_emails, issues)."""
        valid = []
        issues = []
        email_re = re.compile(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$')

        for email in emails:
            email = email.strip()
            if email_re.match(email):
                # Additional checks
                if '..' in email:
                    issues.append(f"⚠️ `{email}` — double dots in address (potentially malformed)")
                elif email.count('@') != 1:
                    issues.append(f"❌ `{email}` — invalid format (multiple @ signs)")
                else:
                    valid.append(email)
            else:
                issues.append(f"❌ `{email}` — failed format validation")

        return valid, issues

    def _pair_authors_emails(self, authors: List[Dict], emails: List[str], text: str) -> List[Dict[str, str]]:
        """Attempt to pair authors with their email addresses using proximity and name matching."""
        mapping = []

        for author in authors:
            name = author.get('name', '')
            name_parts = name.lower().split()
            matched_email = None

            for email in emails:
                email_local = email.split('@')[0].lower()
                # Check if author name parts appear in the email local part
                if any(part in email_local for part in name_parts if len(part) > 2):
                    matched_email = email
                    break

            mapping.append({
                'author': name,
                'email': matched_email or 'Not found',
                'confidence': 'high' if matched_email else 'none',
                'affiliation': author.get('affiliation', '')
            })

        return mapping

    # ──────────────────────────────────────────
    # Integrity Notes & Discrepancy Detection
    # ──────────────────────────────────────────
    def _generate_integrity_notes(self, metadata: Dict, text: str, gemini_used: bool) -> List[Dict[str, str]]:
        """Generate integrity observations and flag discrepancies."""
        notes = []

        # 1. Title checks
        title = metadata.get('title', '')
        if title == 'Unknown' or len(title) < 5:
            notes.append({
                'field': 'title',
                'severity': 'error',
                'message': 'Title could not be extracted. Verify manually from the PDF first page.',
                'suggestion': 'Provide the title manually in the "Paper Title" field above.'
            })
        elif len(title) > 200:
            notes.append({
                'field': 'title',
                'severity': 'warning',
                'message': f'Extracted title is unusually long ({len(title)} chars). May include subtitle or extra text.',
                'suggestion': 'Review and truncate to the actual paper title.'
            })

        # 2. Authors checks
        authors = metadata.get('authors', [])
        emails = metadata.get('email_addresses', [])
        if not authors:
            notes.append({
                'field': 'authors',
                'severity': 'error',
                'message': 'No authors detected.',
                'suggestion': 'Check the area between the title and abstract in the source document.'
            })
        elif len(authors) > 8:
            notes.append({
                'field': 'authors',
                'severity': 'info',
                'message': f'{len(authors)} authors detected — large collaboration.',
                'suggestion': 'Verify author list against the source for false positives.'
            })

        # 3. Author-email count mismatch
        if authors and emails:
            if len(emails) < len(authors):
                notes.append({
                    'field': 'emails',
                    'severity': 'warning',
                    'message': f'{len(authors)} authors but only {len(emails)} email(s) found.',
                    'suggestion': 'Missing emails may be in image/figure format or only the corresponding author is listed.'
                })
            elif len(emails) > len(authors):
                notes.append({
                    'field': 'emails',
                    'severity': 'warning',
                    'message': f'More emails ({len(emails)}) than authors ({len(authors)}) found.',
                    'suggestion': 'Extra emails may belong to editors, reviewers, or footnotes.'
                })
        elif authors and not emails:
            notes.append({
                'field': 'emails',
                'severity': 'warning',
                'message': 'No email addresses found for any author.',
                'suggestion': 'Emails may be in image format or absent. Check the PDF header manually.'
            })

        # 4. Abstract checks
        abstract = metadata.get('abstract', '')
        if not abstract:
            notes.append({
                'field': 'abstract',
                'severity': 'warning',
                'message': 'Abstract could not be extracted.',
                'suggestion': 'Verify the document has an "Abstract" section header. Copy-paste PDFs may break this.'
            })
        elif len(abstract.split()) < 30:
            notes.append({
                'field': 'abstract',
                'severity': 'warning',
                'message': f'Abstract is very short ({len(abstract.split())} words). May be truncated.',
                'suggestion': 'Verify against the source document.'
            })

        # 5. DOI checks
        if not metadata.get('doi'):
            notes.append({
                'field': 'doi',
                'severity': 'info',
                'message': 'No DOI found in the document.',
                'suggestion': 'Search CrossRef (https://search.crossref.org) with the title to locate the DOI.'
            })

        # 6. Keywords
        if not metadata.get('keywords'):
            notes.append({
                'field': 'keywords',
                'severity': 'info',
                'message': 'No keywords/index terms section detected.',
                'suggestion': 'If the paper uses "Index Terms" instead of "Keywords", the parser may need adjustment.'
            })

        # 7. Method note
        notes.append({
            'field': 'method',
            'severity': 'info',
            'message': f'Extraction method: regex-based' + (' + Gemini LLM enhancement' if gemini_used else ' only (no LLM).'),
            'suggestion': 'Enable Gemini API key for improved extraction accuracy.' if not gemini_used else ''
        })

        return notes

    def _flag_unavailable(self, metadata: Dict) -> List[Dict[str, str]]:
        """Flag sections/fields that are unavailable and suggest verification."""
        unavailable = []
        checks = {
            'abstract': 'Look for the section header "Abstract" in your source PDF.',
            'keywords': 'Check for "Keywords:" or "Index Terms:" after the abstract.',
            'doi': 'Search CrossRef or Google Scholar with the exact title.',
            'publication_date': 'Check the first page footer or header of the PDF.',
            'venue': 'Look for journal/conference name on the first page or in PDF metadata.',
        }
        for field, how_to_verify in checks.items():
            val = metadata.get(field)
            if not val or val in ('Unknown', '', [], None):
                unavailable.append({
                    'field': field,
                    'message': f'"{field}" is unavailable in extracted metadata.',
                    'how_to_verify': how_to_verify
                })
        return unavailable

    # ──────────────────────────────────────────
    # Gemini LLM Enhancement
    # ──────────────────────────────────────────
    def _gemini_extract(self, text: str, api_key: str) -> Dict[str, Any]:
        """Use Gemini API for enhanced metadata extraction."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"

        prompt = f"""You are a precise academic metadata extractor. Extract the following from this research paper.
Do NOT hallucinate any information. If a field is not present, set it to null.

Return a JSON object with these exact keys:
- "title": exact paper title as written
- "authors": array of exact author names as listed (strings only)
- "abstract": the full abstract text
- "keywords": array of keywords as listed  
- "venue": journal or conference name if stated
- "publication_date": publication date if found
- "emails": array of email addresses as listed

Paper text (first 4000 chars):
{text[:4000]}

Return ONLY valid JSON. Do not wrap in markdown code blocks."""

        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            response = self._make_api_request(url, json_data=payload)
            content = response['candidates'][0]['content']['parts'][0]['text']

            import json
            # Strip markdown code fences if present
            content = re.sub(r'^```(?:json)?\s*', '', content.strip())
            content = re.sub(r'\s*```$', '', content.strip())

            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                result = {}
                if data.get('title'):
                    result['title'] = data['title']
                if data.get('authors'):
                    authors_raw = data['authors']
                    if isinstance(authors_raw, list) and authors_raw:
                        if isinstance(authors_raw[0], str):
                            result['authors'] = [
                                {'name': a.strip(), 'affiliation': '', 'email': ''}
                                for a in authors_raw if a.strip()
                            ]
                        elif isinstance(authors_raw[0], dict):
                            result['authors'] = authors_raw
                if data.get('abstract'):
                    result['abstract'] = data['abstract']
                if data.get('keywords'):
                    result['keywords'] = [k for k in data['keywords'] if k]
                if data.get('venue'):
                    result['venue'] = data['venue']
                if data.get('publication_date'):
                    result['publication_date'] = data['publication_date']
                if data.get('emails'):
                    result['email_addresses'] = [e for e in data['emails'] if e]
                return result
        except Exception as e:
            logger.warning(f"Gemini extraction parse error: {e}")

        return {}
