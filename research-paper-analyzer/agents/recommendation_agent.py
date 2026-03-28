import logging
from typing import Dict, Any, List

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class RecommendationAgent(BaseAgent):
    """Agent for generating personalized paper recommendations."""

    def __init__(self):
        super().__init__("RecommendationAgent")

    def analyze(self, text: str, api_keys: Dict[str, str], **kwargs) -> Dict[str, Any]:
        """Generate recommendations based on current paper content."""

        paper_title = kwargs.get('paper_title', '')
        indexed_docs = kwargs.get('indexed_documents', [])

        recommendations = []

        # Method 1: Semantic Scholar related papers
        if paper_title and len(paper_title) > 5:
            try:
                scholar_recs = self._semantic_scholar_recommendations(paper_title)
                recommendations.extend(scholar_recs)
            except Exception as e:
                logger.warning(f"Semantic Scholar recommendations failed: {e}")

        # Method 2: Content-based from local index
        if indexed_docs:
            local_recs = self._content_based_recommendations(text, indexed_docs)
            recommendations.extend(local_recs)

        # Method 3: Topic-based recommendations from DuckDuckGo
        if api_keys.get('duckduckgo_enabled'):
            try:
                topic_recs = self._topic_recommendations(text)
                recommendations.extend(topic_recs)
            except Exception as e:
                logger.warning(f"Topic recommendations failed: {e}")

        # Deduplicate and rank
        unique_recs = self._deduplicate(recommendations)
        unique_recs.sort(key=lambda x: x.get('relevance', 0), reverse=True)

        return {
            'recommendations': unique_recs[:10],
            'count': len(unique_recs[:10]),
            'methods_used': list(set(r.get('source', 'unknown') for r in unique_recs)),
        }

    def _semantic_scholar_recommendations(self, title: str) -> List[Dict[str, Any]]:
        """Get recommendations from Semantic Scholar."""
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": title.strip(),
            "limit": 10,
            "fields": "title,url,abstract,year,citationCount,authors"
        }

        try:
            response = self._make_api_request(url, params=params, timeout=10)
            recs = []

            if 'data' in response:
                for paper in response['data']:
                    paper_title = paper.get('title', '')
                    if paper_title.lower().strip() == title.lower().strip():
                        continue  # Skip the paper itself

                    authors = [a.get('name', '') for a in paper.get('authors', [])[:3]]
                    recs.append({
                        'title': paper_title,
                        'url': paper.get('url', ''),
                        'abstract': str(paper.get('abstract', ''))[:200],
                        'year': paper.get('year'),
                        'citations': paper.get('citationCount', 0),
                        'authors': authors,
                        'relevance': 0.9,
                        'source': 'Semantic Scholar',
                        'reason': 'Related paper from academic database'
                    })

            return recs
        except Exception:
            return []

    def _content_based_recommendations(self, text: str, indexed_docs: List[Dict]) -> List[Dict[str, Any]]:
        """Generate recommendations based on content similarity with indexed documents."""
        import re

        # Extract key terms from current paper
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        stopwords = {'the', 'and', 'for', 'that', 'with', 'this', 'from', 'are',
                      'was', 'were', 'been', 'have', 'has', 'had', 'not', 'but'}
        freq = {}
        for w in words:
            if w not in stopwords:
                freq[w] = freq.get(w, 0) + 1

        current_top = set(w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:30])

        recs = []
        for doc in indexed_docs:
            doc_title = doc.get('title', 'Unknown')
            doc_terms = set(doc.get('search_terms', []) if isinstance(doc.get('search_terms'), list) else [])

            overlap = len(current_top & doc_terms) / max(len(current_top | doc_terms), 1)

            if overlap > 0.1:
                recs.append({
                    'title': doc_title,
                    'url': '',
                    'abstract': doc.get('preview', '')[:200] if doc.get('preview') else '',
                    'relevance': round(overlap, 2),
                    'source': 'Local Index',
                    'reason': f'Content similarity: {overlap:.0%} term overlap'
                })

        return recs

    def _topic_recommendations(self, text: str) -> List[Dict[str, Any]]:
        """Get topic-based recommendations via DuckDuckGo."""
        try:
            from duckduckgo_search import DDGS
            import re

            # Extract key phrases
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 50]
            if not sentences:
                return []

            # Use the first meaningful sentence as query
            query = sentences[0][:100] + " research paper"

            recs = []
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
                for result in results:
                    url = result.get('href', '')
                    # Prefer academic sources
                    academic_domains = ['arxiv.org', 'ieee.org', 'acm.org', 'springer.com',
                                        'researchgate.net', 'scholar.google']
                    is_academic = any(d in url for d in academic_domains)

                    recs.append({
                        'title': result.get('title', ''),
                        'url': url,
                        'abstract': result.get('body', '')[:200],
                        'relevance': 0.7 if is_academic else 0.4,
                        'source': 'DuckDuckGo',
                        'reason': 'Topic-related search result' + (' (academic source)' if is_academic else '')
                    })

            return recs
        except Exception:
            return []

    def _deduplicate(self, recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate recommendations."""
        seen = set()
        unique = []
        for rec in recs:
            key = rec.get('title', '').lower().strip()[:50]
            if key and key not in seen:
                seen.add(key)
                unique.append(rec)
        return unique
