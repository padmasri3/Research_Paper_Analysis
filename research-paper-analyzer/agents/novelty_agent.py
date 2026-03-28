import re
import math
import logging
from collections import Counter
from typing import Dict, Any, List
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class NoveltyAgent(BaseAgent):
    """Agent for assessing research novelty and uniqueness."""
    
    def __init__(self):
        super().__init__("NoveltyAgent")
        self.domain_keywords = self._load_domain_keywords()
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                nltk.download('punkt_tab', quiet=True)
            except Exception:
                nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
    
    def analyze(self, text: str, api_keys: Dict[str, str]) -> Dict[str, Any]:
        """Analyze text for novelty and uniqueness."""
        
        factors = []
        score = 0.0
        
        # 1. Terminology novelty
        term_score, term_factors = self._analyze_terminology(text)
        score += term_score * 0.3
        factors.extend(term_factors)
        
        # 2. Methodological novelty
        method_score, method_factors = self._analyze_methodology(text)
        score += method_score * 0.25
        factors.extend(method_factors)
        
        # 3. Conceptual novelty
        concept_score, concept_factors = self._analyze_concepts(text)
        score += concept_score * 0.25
        factors.extend(concept_factors)
        
        # 4. Citation patterns (novelty indicator)
        citation_score, citation_factors = self._analyze_citations(text)
        score += citation_score * 0.2
        factors.extend(citation_factors)
        
        # Normalize to 0-10 scale
        final_score = min(10, max(0, score))
        
        final_score = round(final_score, 1)
        
        return {
            'score': round(final_score, 1),
            'factors': factors,
            'breakdown': {
                'terminology': round(term_score, 1),
                'methodology': round(method_score, 1),
                'concepts': round(concept_score, 1),
                'citations': round(citation_score, 1)
            }
        }
    
    def _analyze_terminology(self, text: str) -> tuple[float, List[str]]:
        """Analyze terminology for novelty indicators."""
        
        score = 0.0
        factors = []
        
        # Extract potential technical terms
        tech_terms = self._extract_technical_terms(text)
        
        # Check for novel terminology patterns
        if len(tech_terms) > 10:
            score += 2.0
            factors.append(f"Rich technical vocabulary ({len(tech_terms)} terms)")
        
        # Check for new compound terms
        compound_terms = [term for term in tech_terms if '-' in term or len(term.split()) > 1]
        if len(compound_terms) > 5:
            score += 1.5
            factors.append(f"Novel compound terminology ({len(compound_terms)} terms)")
        
        # Check for domain-specific innovation
        domain_innovation = self._check_domain_innovation(tech_terms)
        if domain_innovation:
            score += 2.0
            factors.append(f"Domain-specific innovation detected")
        
        return score, factors
    
    def _analyze_methodology(self, text: str) -> tuple[float, List[str]]:
        """Analyze methodological novelty."""
        
        score = 0.0
        factors = []
        
        # Look for methodological keywords
        method_keywords = [
            'novel approach', 'new method', 'innovative technique', 'proposed algorithm',
            'framework', 'model', 'system', 'architecture', 'methodology'
        ]
        
        method_mentions = sum(text.lower().count(keyword) for keyword in method_keywords)
        
        if method_mentions > 5:
            score += 2.5
            factors.append(f"Strong methodological focus ({method_mentions} mentions)")
        
        # Check for experimental design novelty
        experiment_keywords = ['experiment', 'evaluation', 'benchmark', 'comparison', 'analysis']
        experiment_mentions = sum(text.lower().count(keyword) for keyword in experiment_keywords)
        
        if experiment_mentions > 8:
            score += 1.5
            factors.append("Comprehensive experimental design")
        
        # Check for interdisciplinary approaches
        disciplines = ['machine learning', 'deep learning', 'neural network', 'artificial intelligence',
                      'computer vision', 'natural language', 'data mining', 'statistics']
        
        discipline_count = sum(1 for disc in disciplines if disc in text.lower())
        
        if discipline_count > 2:
            score += 2.0
            factors.append(f"Interdisciplinary approach ({discipline_count} fields)")
        
        return score, factors
    
    def _analyze_concepts(self, text: str) -> tuple[float, List[str]]:
        """Analyze conceptual novelty."""
        
        score = 0.0
        factors = []
        
        # Check for theoretical contributions
        theory_keywords = ['theory', 'theorem', 'principle', 'concept', 'hypothesis', 'paradigm']
        theory_mentions = sum(text.lower().count(keyword) for keyword in theory_keywords)
        
        if theory_mentions > 3:
            score += 2.0
            factors.append("Theoretical contributions identified")
        
        # Check for problem formulation novelty
        problem_keywords = ['problem', 'challenge', 'limitation', 'gap', 'issue']
        problem_mentions = sum(text.lower().count(keyword) for keyword in problem_keywords)
        
        if problem_mentions > 5:
            score += 1.5
            factors.append("Novel problem formulation")
        
        # Check for solution uniqueness
        solution_keywords = ['solution', 'approach', 'strategy', 'technique', 'method']
        solution_mentions = sum(text.lower().count(keyword) for keyword in solution_keywords)
        
        if solution_mentions > 8:
            score += 1.8
            factors.append("Comprehensive solution approach")
        
        return score, factors
    
    def _analyze_citations(self, text: str) -> tuple[float, List[str]]:
        """Analyze citation patterns for novelty indicators."""
        
        score = 0.0
        factors = []
        
        # Count references/citations
        citations = len(re.findall(r'\[[0-9]+\]|\([A-Za-z]+ et al\.\, [0-9]{4}\)', text))
        
        if citations > 20:
            score += 1.5
            factors.append(f"Comprehensive literature review ({citations} citations)")
        elif citations < 5:
            # Very few citations might indicate very novel work or poor scholarship
            score += 0.5
            factors.append("Limited citations (potentially very novel or niche area)")
        
        # Check for recent citations (assuming format includes years)
        recent_citations = len(re.findall(r'202[0-9]', text))
        if recent_citations > citations * 0.3:  # More than 30% recent
            score += 1.0
            factors.append("High proportion of recent citations")
        
        return score, factors
    
    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms from text."""
        
        # Simple technical term extraction
        words = re.findall(r'\b[A-Z][a-z]*[A-Z][a-z]*\b', text)  # CamelCase
        words.extend(re.findall(r'\b[a-z]+-[a-z]+\b', text))      # hyphenated
        words.extend(re.findall(r'\b[A-Z]{2,}\b', text))          # Acronyms
        
        # Filter common words
        stopwords = {'The', 'This', 'That', 'These', 'Those', 'And', 'But', 'Or'}
        technical_terms = [word for word in words if word not in stopwords and len(word) > 2]
        
        return list(set(technical_terms))
    
    def _check_domain_innovation(self, terms: List[str]) -> bool:
        """Check if terms suggest domain innovation."""
        
        innovation_indicators = ['AI', 'ML', 'Deep', 'Neural', 'Quantum', 'Blockchain', 
                               'IoT', 'Edge', 'Federated', 'Adversarial']
        
        return any(indicator in ' '.join(terms) for indicator in innovation_indicators)
    
    def _load_domain_keywords(self) -> Dict[str, List[str]]:
        """Load domain-specific keywords for novelty assessment."""
        
        return {
            'ai_ml': ['artificial intelligence', 'machine learning', 'deep learning', 'neural networks'],
            'computer_vision': ['computer vision', 'image processing', 'object detection', 'segmentation'],
            'nlp': ['natural language processing', 'text mining', 'sentiment analysis', 'language models'],
            'data_science': ['data mining', 'big data', 'analytics', 'visualization'],
            'systems': ['distributed systems', 'cloud computing', 'edge computing', 'microservices']
        }
