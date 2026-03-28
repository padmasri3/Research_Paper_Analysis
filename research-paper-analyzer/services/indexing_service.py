import hashlib
import json
import logging
import os
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class IndexingService:
    """Service for storing and indexing processed documents with embeddings.
    
    Uses a file-based approach (JSON) for simplicity. Can be swapped for
    ChromaDB, Pinecone, or PostgreSQL in production.
    """

    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            storage_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'index')
        self.storage_dir = storage_dir
        self.index_file = os.path.join(storage_dir, 'document_index.json')
        self._ensure_storage()

    def _ensure_storage(self):
        """Create storage directory if needed."""
        os.makedirs(self.storage_dir, exist_ok=True)
        if not os.path.exists(self.index_file):
            self._save_index({'documents': {}, 'metadata': {'created': datetime.now().isoformat()}})

    def _load_index(self) -> Dict:
        """Load the document index from disk."""
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {'documents': {}, 'metadata': {'created': datetime.now().isoformat()}}

    def _save_index(self, index: Dict):
        """Save the document index to disk."""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, default=str)

    def add_document(self, doc_id: str, title: str, text: str,
                     metadata: Dict = None, analysis_results: Dict = None) -> Dict[str, Any]:
        """Add a document to the index."""
        index = self._load_index()

        # Generate text embedding (simple TF-IDF-based word vector)
        embedding = self._generate_embedding(text)

        # Extract searchable terms
        terms = self._extract_search_terms(text, metadata)

        doc_entry = {
            'id': doc_id,
            'title': title,
            'text_preview': text[:1000],
            'text_hash': hashlib.sha256(text.encode()).hexdigest(),
            'word_count': len(text.split()),
            'metadata': metadata or {},
            'analysis': analysis_results or {},
            'embedding': embedding,
            'search_terms': terms,
            'indexed_at': datetime.now().isoformat(),
        }

        index['documents'][doc_id] = doc_entry
        index['metadata']['last_updated'] = datetime.now().isoformat()
        index['metadata']['total_documents'] = len(index['documents'])

        self._save_index(index)

        return {
            'id': doc_id,
            'status': 'indexed',
            'total_documents': len(index['documents'])
        }

    def search(self, query: str, filters: Dict = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search the document index using keyword + simple semantic matching."""
        index = self._load_index()
        documents = index.get('documents', {})

        if not documents:
            return []

        query_terms = set(query.lower().split())
        query_embedding = self._generate_embedding(query)
        results = []

        for doc_id, doc in documents.items():
            score = 0.0

            # Keyword matching (BM25-like)
            doc_terms = set(doc.get('search_terms', []))
            keyword_overlap = len(query_terms & doc_terms)
            if keyword_overlap > 0:
                score += keyword_overlap * 2.0

            # Title matching (high boost)
            title_lower = doc.get('title', '').lower()
            for term in query_terms:
                if term in title_lower:
                    score += 5.0

            # Text preview matching
            preview_lower = doc.get('text_preview', '').lower()
            for term in query_terms:
                if term in preview_lower:
                    score += 1.0

            # Simple embedding similarity
            doc_embedding = doc.get('embedding', {})
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            score += similarity * 3.0

            # Apply filters
            if filters:
                if not self._apply_filters(doc, filters):
                    continue

            if score > 0:
                results.append({
                    'id': doc_id,
                    'title': doc.get('title', 'Unknown'),
                    'preview': doc.get('text_preview', '')[:300],
                    'score': round(score, 2),
                    'metadata': doc.get('metadata', {}),
                    'analysis': doc.get('analysis', {}),
                    'indexed_at': doc.get('indexed_at', ''),
                    'word_count': doc.get('word_count', 0),
                })

        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all indexed documents."""
        index = self._load_index()
        documents = []
        for doc_id, doc in index.get('documents', {}).items():
            documents.append({
                'id': doc_id,
                'title': doc.get('title', 'Unknown'),
                'word_count': doc.get('word_count', 0),
                'metadata': doc.get('metadata', {}),
                'analysis': doc.get('analysis', {}),
                'indexed_at': doc.get('indexed_at', ''),
            })
        return documents

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID."""
        index = self._load_index()
        return index.get('documents', {}).get(doc_id)

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the index."""
        index = self._load_index()
        if doc_id in index.get('documents', {}):
            del index['documents'][doc_id]
            index['metadata']['total_documents'] = len(index['documents'])
            self._save_index(index)
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        index = self._load_index()
        docs = index.get('documents', {})
        return {
            'total_documents': len(docs),
            'total_words': sum(d.get('word_count', 0) for d in docs.values()),
            'last_updated': index.get('metadata', {}).get('last_updated', 'Never'),
            'created': index.get('metadata', {}).get('created', 'Unknown'),
        }

    def get_similar_documents(self, doc_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find documents similar to a given document."""
        index = self._load_index()
        target_doc = index.get('documents', {}).get(doc_id)

        if not target_doc:
            return []

        target_embedding = target_doc.get('embedding', {})
        target_terms = set(target_doc.get('search_terms', []))

        results = []
        for did, doc in index.get('documents', {}).items():
            if did == doc_id:
                continue

            # Compute similarity
            doc_embedding = doc.get('embedding', {})
            sim = self._cosine_similarity(target_embedding, doc_embedding)

            # Term overlap
            doc_terms = set(doc.get('search_terms', []))
            term_overlap = len(target_terms & doc_terms) / max(len(target_terms | doc_terms), 1)

            score = sim * 0.6 + term_overlap * 0.4

            results.append({
                'id': did,
                'title': doc.get('title', 'Unknown'),
                'similarity': round(score, 3),
                'indexed_at': doc.get('indexed_at', ''),
            })

        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

    def _generate_embedding(self, text: str) -> Dict[str, float]:
        """Generate a simple word-frequency embedding. 
        Replace with sentence-transformers for production."""
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        # Filter stopwords
        stopwords = {'the', 'and', 'for', 'that', 'with', 'this', 'from', 'are',
                      'was', 'were', 'been', 'have', 'has', 'had', 'not', 'but',
                      'can', 'will', 'just', 'also', 'than', 'more', 'some',
                      'other', 'into', 'its', 'our', 'their', 'which', 'when'}
        words = [w for w in words if w not in stopwords]

        # Create frequency dict (acts as a sparse embedding)
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1

        # Normalize
        total = max(sum(freq.values()), 1)
        return {k: round(v / total, 6) for k, v in sorted(freq.items(), key=lambda x: -x[1])[:200]}

    def _cosine_similarity(self, emb1: Dict[str, float], emb2: Dict[str, float]) -> float:
        """Compute cosine similarity between two sparse embeddings."""
        if not emb1 or not emb2:
            return 0.0

        common_keys = set(emb1.keys()) & set(emb2.keys())
        if not common_keys:
            return 0.0

        dot = sum(emb1[k] * emb2[k] for k in common_keys)
        norm1 = sum(v ** 2 for v in emb1.values()) ** 0.5
        norm2 = sum(v ** 2 for v in emb2.values()) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    def _extract_search_terms(self, text: str, metadata: Dict = None) -> List[str]:
        """Extract searchable terms from document."""
        terms = set()

        # From text
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        # Get top frequent words (excluding stopwords)
        stopwords = {'the', 'and', 'for', 'that', 'with', 'this', 'from', 'are',
                      'was', 'were', 'been', 'have', 'has', 'had', 'not', 'but'}
        freq = {}
        for w in words:
            if w not in stopwords:
                freq[w] = freq.get(w, 0) + 1
        top_words = sorted(freq.items(), key=lambda x: -x[1])[:50]
        terms.update(w for w, _ in top_words)

        # From metadata
        if metadata:
            if metadata.get('keywords'):
                terms.update(k.lower() for k in metadata['keywords'])
            if metadata.get('title'):
                terms.update(metadata['title'].lower().split())

        return list(terms)

    def _apply_filters(self, doc: Dict, filters: Dict) -> bool:
        """Apply search filters to a document."""
        analysis = doc.get('analysis', {})

        if 'min_novelty' in filters:
            novelty = analysis.get('novelty', {}).get('score', 0)
            if novelty < filters['min_novelty']:
                return False

        if 'max_plagiarism' in filters:
            plagiarism = analysis.get('plagiarism', {}).get('percentage', 0)
            if plagiarism > filters['max_plagiarism']:
                return False

        if 'min_words' in filters:
            if doc.get('word_count', 0) < filters['min_words']:
                return False

        return True
