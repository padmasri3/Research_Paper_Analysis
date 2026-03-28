import re
import logging
from typing import Dict, Any, List, Optional
from urllib.parse import quote
import time

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ValidationAgent(BaseAgent):
    """Agent for cross-referencing and validating extracted metadata."""

    def __init__(self):
        super().__init__("ValidationAgent")

    def analyze(self, text: str, api_keys: Dict[str, str], **kwargs) -> Dict[str, Any]:
        """Validate extracted metadata by cross-referencing external sources."""

        metadata = kwargs.get('metadata', {})
        title = metadata.get('title', '') or kwargs.get('paper_title', '')
        doi = metadata.get('doi', '')

        checks = []
        external_data = {}

        # Check 1: Validate DOI if present
        if doi:
            doi_check = self._validate_doi(doi)
            checks.append(doi_check)
            if doi_check['status'] == 'pass':
                external_data = doi_check.get('external_data', {})

        # Check 2: Cross-reference title with Semantic Scholar
        if title and len(title) > 5:
            title_check = self._validate_title_semantic_scholar(title)
            checks.append(title_check)
            if not external_data and title_check.get('external_data'):
                external_data = title_check['external_data']

        # Check 3: Document structure quality
        structure_check = self._validate_structure(text)
        checks.append(structure_check)

        # Check 4: Reference quality
        ref_check = self._validate_references(text)
        checks.append(ref_check)

        # Check 5: Metadata completeness
        completeness_check = self._validate_completeness(metadata)
        checks.append(completeness_check)

        # Calculate overall trust score
        passed = sum(1 for c in checks if c['status'] == 'pass')
        warned = sum(1 for c in checks if c['status'] == 'warn')
        total = len(checks)
        trust_score = (passed + warned * 0.5) / max(total, 1)

        return {
            'checks': checks,
            'trust_score': round(trust_score, 2),
            'total_checks': total,
            'passed': passed,
            'warnings': warned,
            'failures': total - passed - warned,
            'external_data': external_data,
        }

    def _validate_doi(self, doi: str) -> Dict[str, Any]:
        """Validate DOI by resolving it via CrossRef API."""
        try:
            url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
            response = self._make_api_request(url, timeout=10)

            if response and response.get('status') == 'ok':
                work = response.get('message', {})
                return {
                    'field': 'doi',
                    'check': 'DOI Resolution',
                    'status': 'pass',
                    'message': f"DOI resolved successfully: {doi}",
                    'source': 'crossref',
                    'external_data': {
                        'title': work.get('title', [''])[0],
                        'authors': [f"{a.get('given', '')} {a.get('family', '')}" for a in work.get('author', [])],
                        'publisher': work.get('publisher', ''),
                        'published_date': str(work.get('published-print', {}).get('date-parts', [['']])[0]),
                    }
                }
        except Exception as e:
            logger.warning(f"DOI validation failed: {e}")

        return {
            'field': 'doi',
            'check': 'DOI Resolution',
            'status': 'warn',
            'message': f"Could not resolve DOI: {doi}",
            'source': 'crossref'
        }

    def _validate_title_semantic_scholar(self, title: str) -> Dict[str, Any]:
        """Cross-reference paper title with Semantic Scholar."""
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": title.strip(),
                "limit": 3,
                "fields": "title,authors,year,citationCount,url"
            }
            response = self._make_api_request(url, params=params, timeout=10)

            if 'data' in response and response['data']:
                best_match = response['data'][0]
                match_title = best_match.get('title', '')

                # Check if titles are similar
                if self._title_similarity(title, match_title) > 0.7:
                    return {
                        'field': 'title',
                        'check': 'Title Verification',
                        'status': 'pass',
                        'message': f"Paper found in Semantic Scholar: \"{match_title}\"",
                        'source': 'semantic_scholar',
                        'external_data': {
                            'title': match_title,
                            'authors': [a.get('name', '') for a in best_match.get('authors', [])],
                            'year': best_match.get('year'),
                            'citations': best_match.get('citationCount', 0),
                            'url': best_match.get('url', ''),
                        }
                    }
                else:
                    return {
                        'field': 'title',
                        'check': 'Title Verification',
                        'status': 'warn',
                        'message': f"Title partially matched. Best match: \"{match_title}\"",
                        'source': 'semantic_scholar'
                    }

            time.sleep(1)  # Rate limiting
        except Exception as e:
            logger.warning(f"Semantic Scholar validation failed: {e}")

        return {
            'field': 'title',
            'check': 'Title Verification',
            'status': 'warn',
            'message': "Could not verify title against external databases",
            'source': 'semantic_scholar'
        }

    def _validate_structure(self, text: str) -> Dict[str, Any]:
        """Validate document structure quality."""
        issues = []

        # Check for essential sections
        text_lower = text.lower()
        essential_sections = {
            'abstract': bool(re.search(r'\babstract\b', text_lower)),
            'introduction': bool(re.search(r'\bintroduction\b', text_lower)),
            'references': bool(re.search(r'\breferences?\b', text_lower)),
        }

        missing = [s for s, found in essential_sections.items() if not found]
        if missing:
            issues.append(f"Missing sections: {', '.join(missing)}")

        # Check word count
        word_count = len(text.split())
        if word_count < 500:
            issues.append(f"Very short document ({word_count} words)")
        elif word_count > 50000:
            issues.append(f"Unusually long document ({word_count} words)")

        status = 'pass' if not issues else ('warn' if len(issues) <= 1 else 'fail')

        return {
            'field': 'structure',
            'check': 'Document Structure',
            'status': status,
            'message': '; '.join(issues) if issues else 'Document structure looks good',
            'source': 'local'
        }

    def _validate_references(self, text: str) -> Dict[str, Any]:
        """Validate reference section quality."""
        # Count citations in text
        citation_count = len(re.findall(r'\[\d+\]', text))
        author_citations = len(re.findall(r'\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\d{4}\)', text))
        total_citations = citation_count + author_citations

        if total_citations > 10:
            return {
                'field': 'references',
                'check': 'Reference Quality',
                'status': 'pass',
                'message': f"Found {total_citations} citations — good reference coverage",
                'source': 'local'
            }
        elif total_citations > 3:
            return {
                'field': 'references',
                'check': 'Reference Quality',
                'status': 'warn',
                'message': f"Only {total_citations} citations found — consider adding more references",
                'source': 'local'
            }
        else:
            return {
                'field': 'references',
                'check': 'Reference Quality',
                'status': 'fail',
                'message': f"Very few citations ({total_citations}) — paper may lack scholarly support",
                'source': 'local'
            }

    def _validate_completeness(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Check metadata completeness."""
        required_fields = ['title', 'authors', 'abstract']
        optional_fields = ['keywords', 'doi', 'publication_date']

        missing_required = []
        missing_optional = []

        for field in required_fields:
            val = metadata.get(field)
            if not val or val == 'Unknown' or val == []:
                missing_required.append(field)

        for field in optional_fields:
            val = metadata.get(field)
            if not val or val == 'Unknown' or val == [] or val is None:
                missing_optional.append(field)

        if not missing_required:
            status = 'pass'
            message = 'All required metadata fields present'
            if missing_optional:
                status = 'warn'
                message += f'. Missing optional: {", ".join(missing_optional)}'
        else:
            status = 'fail'
            message = f'Missing required fields: {", ".join(missing_required)}'

        return {
            'field': 'completeness',
            'check': 'Metadata Completeness',
            'status': status,
            'message': message,
            'source': 'local'
        }

    def _title_similarity(self, title1: str, title2: str) -> float:
        """Simple title similarity based on word overlap."""
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)
