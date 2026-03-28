import re
import logging
from typing import Dict, Any, List

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class EntityClassifierAgent(BaseAgent):
    """Agent for classifying named entities in research papers."""

    def __init__(self):
        super().__init__("EntityClassifierAgent")
        self.entity_patterns = self._load_entity_patterns()

    def analyze(self, text: str, api_keys: Dict[str, str], **kwargs) -> Dict[str, Any]:
        """Extract and classify entities: methods, datasets, metrics, tools, etc."""

        entities = []

        # Rule-based entity extraction
        rule_entities = self._rule_based_extraction(text)
        entities.extend(rule_entities)

        # Gemini-enhanced extraction if available
        if api_keys.get('gemini_api_key'):
            try:
                gemini_entities = self._gemini_extraction(text, api_keys['gemini_api_key'])
                entities.extend(gemini_entities)
            except Exception as e:
                logger.warning(f"Gemini entity extraction failed: {e}")

        # Deduplicate
        unique_entities = self._deduplicate(entities)

        # Group by type
        grouped = {}
        for entity in unique_entities:
            etype = entity['type']
            if etype not in grouped:
                grouped[etype] = []
            grouped[etype].append(entity)

        return {
            'entities': unique_entities,
            'entity_count': len(unique_entities),
            'grouped': grouped,
            'types_found': list(grouped.keys()),
            'summary': self._generate_entity_summary(grouped)
        }

    def _rule_based_extraction(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using rule-based patterns."""
        entities = []

        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_text = match.group(0).strip()
                    if len(entity_text) > 2 and len(entity_text) < 100:
                        entities.append({
                            'text': entity_text,
                            'type': entity_type,
                            'confidence': 0.7,
                            'method': 'rule_based'
                        })

        # Extract acronyms
        acronyms = re.findall(r'\b([A-Z]{2,6})\b(?:\s*\(([^)]+)\))?', text)
        for acronym, expansion in acronyms:
            if acronym not in ('THE', 'AND', 'FOR', 'NOT', 'BUT', 'ARE', 'WAS', 'HAS',
                                'HIS', 'HER', 'PDF', 'TXT', 'URL'):
                entities.append({
                    'text': f"{acronym}" + (f" ({expansion})" if expansion else ""),
                    'type': 'acronym',
                    'confidence': 0.6,
                    'method': 'rule_based'
                })

        return entities

    def _gemini_extraction(self, text: str, api_key: str) -> List[Dict[str, Any]]:
        """Use Gemini API for entity extraction."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"

        prompt = f"""Extract all named entities from this research paper text and classify them.

Entity types to look for:
- method: algorithms, techniques, approaches (e.g., "Random Forest", "Transformer", "BERT")
- dataset: datasets mentioned (e.g., "ImageNet", "MNIST", "COCO")
- metric: evaluation metrics (e.g., "accuracy", "F1 score", "BLEU score")
- tool: software tools or libraries (e.g., "TensorFlow", "PyTorch", "scikit-learn")
- organization: universities, labs, companies
- person: researcher names mentioned in context

Paper text (first 3000 chars):
{text[:3000]}

Return a JSON array of objects with keys: text, type, confidence (0-1).
Return ONLY valid JSON."""

        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            response = self._make_api_request(url, json_data=payload)
            content = response['candidates'][0]['content']['parts'][0]['text']

            import json
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return [{
                    'text': e.get('text', ''),
                    'type': e.get('type', 'unknown'),
                    'confidence': float(e.get('confidence', 0.8)),
                    'method': 'gemini'
                } for e in parsed if e.get('text')]
        except Exception:
            pass

        return []

    def _deduplicate(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entities, keeping highest confidence."""
        seen = {}
        for entity in entities:
            key = entity['text'].lower().strip()
            if key not in seen or entity['confidence'] > seen[key]['confidence']:
                seen[key] = entity
        return list(seen.values())

    def _generate_entity_summary(self, grouped: Dict[str, List]) -> str:
        """Generate a brief summary of found entities."""
        parts = []
        for etype, elist in grouped.items():
            parts.append(f"{len(elist)} {etype}(s)")
        return "Found: " + ", ".join(parts) if parts else "No entities detected."

    def _load_entity_patterns(self) -> Dict[str, List[str]]:
        """Load regex patterns for entity detection."""
        return {
            'method': [
                r'\b(?:deep learning|machine learning|neural network|random forest|'
                r'support vector machine|SVM|decision tree|k-nearest neighbor|KNN|'
                r'gradient boosting|XGBoost|LSTM|GRU|CNN|RNN|GAN|VAE|'
                r'transformer|attention mechanism|BERT|GPT|ResNet|VGG|'
                r'reinforcement learning|transfer learning|federated learning|'
                r'backpropagation|dropout|batch normalization|'
                r'principal component analysis|PCA|t-SNE|UMAP)\b',
            ],
            'dataset': [
                r'\b(?:ImageNet|CIFAR-?\d+|MNIST|COCO|Pascal VOC|'
                r'SQuAD|GLUE|SuperGLUE|WikiText|Penn Treebank|'
                r'UCI\s+\w+|Kaggle\s+\w+|OpenML|IMDB|Yelp|Amazon\s+Reviews|'
                r'MS\s*MARCO|Natural\s+Questions)\b',
            ],
            'metric': [
                r'\b(?:accuracy|precision|recall|F1[\s-]?score|'
                r'AUC[-\s]?ROC|mean\s+squared\s+error|MSE|RMSE|MAE|'
                r'BLEU\s*(?:score)?|ROUGE|perplexity|IoU|mAP|'
                r'cross[\s-]entropy|log[\s-]?loss|R[\s-]?squared)\b',
            ],
            'tool': [
                r'\b(?:TensorFlow|PyTorch|Keras|scikit[\s-]learn|'
                r'Pandas|NumPy|SciPy|Matplotlib|Seaborn|'
                r'Jupyter|Colab|Hugging\s*Face|spaCy|NLTK|'
                r'OpenCV|Detectron|CUDA|cuDNN)\b',
            ],
        }
