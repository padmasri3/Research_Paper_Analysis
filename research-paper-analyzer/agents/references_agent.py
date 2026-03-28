import re
import logging
import requests
from typing import Dict, Any, List
from urllib.parse import quote
import time

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class ReferencesAgent(BaseAgent):
    """Agent for finding references and related work."""
    
    def __init__(self):
        super().__init__("ReferencesAgent")
    
    def analyze(self, text: str, api_keys: Dict[str, str], paper_title: str = "") -> Dict[str, Any]:
        """Find references and related work for the paper."""
        
        references = []
        
        # Clean title for better matching
        clean_title = ""
        if paper_title:
            clean_title = re.sub(r'\.(pdf|txt|docx?)$', '', paper_title, flags=re.IGNORECASE)
            clean_title = clean_title.replace('_', ' ').replace('-', ' ').strip()
            
        # Extract key topics and terms
        topics = self._extract_key_topics(text, clean_title)
        
        # Method 0: Semantic Scholar (Best for exact title matches)
        if clean_title:
            try:
                semantic_refs = self._search_semantic_scholar(clean_title)
                references.extend(semantic_refs)
            except Exception as e:
                logger.warning(f"Semantic Scholar search failed: {e}")
        
        # Method 1: Wikipedia search
        if api_keys.get('wikipedia_enabled', True):
            try:
                wiki_refs = self._search_wikipedia(topics)
                references.extend(wiki_refs)
            except Exception as e:
                logger.warning(f"Wikipedia search failed: {e}")
        
        # Method 2: DuckDuckGo search
        if api_keys.get('duckduckgo_enabled', True):
            try:
                ddg_refs = self._search_duckduckgo(topics)
                references.extend(ddg_refs)
            except Exception as e:
                logger.warning(f"DuckDuckGo search failed: {e}")
        
        # Method 3: Extract existing citations from text
        existing_citations = self._extract_existing_citations(text)
        references.extend(existing_citations)
        
        # Deduplicate and rank
        unique_references = self._deduplicate_references(references)
        
        # STRICT ACCURACY FILTER: Only keep references that match the paper title exactly
        if clean_title:
            title_lower = clean_title.lower()
            exact_refs = []
            for ref in unique_references:
                ref_title = ref['title'].lower()
                # Check if the reference title contains the paper title or vice-versa
                if title_lower in ref_title or ref_title in title_lower:
                    exact_refs.append(ref)
            
            # If we found exact matches, ONLY return those to guarantee accuracy
            if exact_refs:
                unique_references = exact_refs
                
        ranked_references = self._rank_references(unique_references, topics)
        
        return {
            'links': ranked_references[:15],  # Top 15 references
            'count': len(ranked_references),
            'topics_searched': topics,
            'sources_used': self._get_sources_used(api_keys)
        }
    
    def _extract_key_topics(self, text: str, title: str = "") -> List[str]:
        """Extract highly specific topics and terms from the text."""
        
        topics = []
        
        # STRICT TARGETING: If a title is provided, ignore all other generic topics to ensure 100% accurate searches
        if title and len(title.strip()) > 3:
            return [title.strip()]
        
        # Extract from abstract
        abstract_match = re.search(r'abstract[:\s]+(.*?)(?:introduction|keywords|\n\n)', 
                                 text.lower(), re.IGNORECASE | re.DOTALL)
        
        if abstract_match:
            abstract_text = abstract_match.group(1)
            # Find the longest meaningful phrase or sentence in the abstract to use as a search query
            sentences = [s.strip() for s in abstract_text.split('.') if len(s.strip()) > 30]
            if sentences:
                topics.append(sentences[0])
                
        # We also extract explicit "Keywords:" if they exist in the paper
        keywords_match = re.search(r'keywords?[:\s]+(.*?)(?:\n\n|\.|introduction)', text.lower(), re.IGNORECASE)
        if keywords_match:
            kw_text = keywords_match.group(1).strip()
            kws = [k.strip() for k in kw_text.split(',') if len(k.strip()) > 4]
            # Add up to 2 multi-word keywords
            topics.extend([k for k in kws if ' ' in k][:2])
            
        return topics[:4]  # Limit to top 4 highly specific topics
    
    def _search_semantic_scholar(self, title: str) -> List[Dict[str, str]]:
        """Search Semantic Scholar for exact or highly relevant academic papers."""
        if not title or len(title.strip()) < 5:
            return []
            
        try:
            url = f"https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": title.strip(),
                "limit": 5,
                "fields": "title,url,abstract"
            }
            response = self._make_api_request(url, params=params)
            
            references = []
            if 'data' in response:
                for paper in response['data']:
                    references.append({
                        'title': paper.get('title', 'Unknown Title'),
                        'url': paper.get('url', '') or f"https://api.semanticscholar.org/CorpusID:{paper.get('corpusId', '')}",
                        'description': str(paper.get('abstract', 'No abstract available'))[:200] + "...",
                        'source': 'Semantic Scholar',
                        'relevance_score': 1.0
                    })
            return references
        except Exception:
            return []

    def _search_wikipedia(self, topics: List[str]) -> List[Dict[str, str]]:
        """Search Wikipedia for related articles."""
        
        try:
            import wikipedia
            references = []
            
            for topic in topics[:5]:  # Search top 5 topics
                try:
                    # Search for pages
                    search_results = wikipedia.search(topic, results=3)
                    
                    for result in search_results:
                        try:
                            page = wikipedia.page(result)
                            references.append({
                                'title': page.title,
                                'url': page.url,
                                'description': page.summary[:200] + "...",
                                'source': 'Wikipedia',
                                'relevance_score': 0.7
                            })
                            time.sleep(0.5)  # Rate limiting
                        except:
                            continue
                            
                except Exception:
                    continue
            
            return references
            
        except ImportError:
            return []
    
    def _search_duckduckgo(self, topics: List[str]) -> List[Dict[str, str]]:
        """Search DuckDuckGo for related papers and articles."""
        
        try:
            from duckduckgo_search import DDGS
            references = []
            
            with DDGS() as ddgs:
                for topic in topics[:3]:  # Search top 3 topics
                    query = f'"{topic}" research paper academic'
                    
                    try:
                        results = list(ddgs.text(query, max_results=5))
                        
                        for result in results:
                            # Filter for academic sources
                            if any(domain in result['href'] for domain in 
                                  ['arxiv.org', 'ieee.org', 'acm.org', 'springer.com', 'researchgate.net']):
                                references.append({
                                    'title': result['title'],
                                    'url': result['href'],
                                    'description': result['body'],
                                    'source': 'DuckDuckGo (Academic)',
                                    'relevance_score': 0.8
                                })
                        
                        time.sleep(2)  # Rate limiting
                    except Exception:
                        continue
            
            return references
            
        except ImportError:
            return []
    
    def _extract_existing_citations(self, text: str) -> List[Dict[str, str]]:
        """Extract existing citations from the paper text."""
        
        references = []
        
        # Look for reference sections
        ref_section = re.search(r'references?\s*\n(.*?)(?:\n\n|\Z)', text, re.IGNORECASE | re.DOTALL)
        
        if ref_section:
            ref_text = ref_section.group(1)
            
            # Extract individual references (basic pattern)
            ref_lines = [line.strip() for line in ref_text.split('\n') if line.strip()]
            
            for line in ref_lines[:10]:  # Process first 10 references
                # Try to extract title and create a search link
                title_match = re.search(r'"([^"]+)"', line)
                if title_match:
                    title = title_match.group(1)
                    search_url = f"https://scholar.google.com/scholar?q={quote(title)}"
                    
                    references.append({
                        'title': title,
                        'url': search_url,
                        'description': f"Citation from paper: {line[:100]}...",
                        'source': 'Paper Citations',
                        'relevance_score': 1.0
                    })
        
        return references
    
    def _extract_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        
        # Simple phrase extraction
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
        phrases = []
        
        for sentence in sentences[:3]:  # Process first 3 sentences
            # Extract noun phrases (basic pattern)
            words = sentence.split()
            for i in range(len(words) - 1):
                if len(words[i]) > 4 and words[i].isalpha():
                    phrases.append(words[i])
        
        return phrases[:10]
    
    def _deduplicate_references(self, references: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove duplicate references based on title similarity."""
        
        unique_refs = []
        seen_titles = set()
        
        for ref in references:
            title_lower = ref['title'].lower()
            
            # Simple deduplication based on title
            if not any(abs(len(title_lower) - len(seen)) < 5 and 
                      title_lower[:20] == seen[:20] for seen in seen_titles):
                unique_refs.append(ref)
                seen_titles.add(title_lower)
        
        return unique_refs
    
    def _rank_references(self, references: List[Dict[str, str]], topics: List[str]) -> List[Dict[str, str]]:
        """Rank references by relevance."""
        
        # Simple ranking based on relevance score and title matching
        for ref in references:
            score = ref.get('relevance_score', 0.5)
            
            # Boost score if title matches topics
            for topic in topics:
                if topic.lower() in ref['title'].lower():
                    score += 0.2
            
            ref['final_score'] = score
        
        # Sort by final score
        return sorted(references, key=lambda x: x.get('final_score', 0), reverse=True)
    
    def _get_sources_used(self, api_keys: Dict[str, str]) -> List[str]:
        """Get list of sources that were used for reference search."""
        
        sources = ['Semantic Scholar']  # Always checked when title provided
        if api_keys.get('wikipedia_enabled'):
            sources.append('Wikipedia')
        if api_keys.get('duckduckgo_enabled'):
            sources.append('DuckDuckGo')
        sources.append('Paper Citations')
        
        return sources
