import logging
from typing import List, Tuple
import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    """
    Engine for generating text embeddings and calculating semantic similarity
    using the all-MiniLM-L6-v2 model.
    """
    
    MODEL_NAME = 'all-MiniLM-L6-v2'
    DEFAULT_THRESHOLD = 0.85

    def __init__(self, model_name: str = MODEL_NAME):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed. Please install it via 'pip install sentence-transformers'")
        
        try:
            logger.info(f"Loading embedding model: {model_name}...")
            self.model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {e}")
            raise

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encodes a list of texts into embeddings.
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return np.array(embeddings)

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculates cosine similarity between two embeddings using numpy dot product
        of normalized vectors for efficiency.
        """
        e1 = np.asarray(embedding1)
        e2 = np.asarray(embedding2)
        
        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(np.dot(e1, e2) / (norm1 * norm2))

    def filter_redundant_texts(self, texts: List[str], threshold: float = DEFAULT_THRESHOLD) -> List[str]:
        """
        Filters out texts that are semantically too similar to previously kept texts.
        Uses matrix operations to compare current text against all kept texts.
        
        Args:
            texts: List of strings to filter.
            threshold: Similarity threshold above which a text is considered redundant.
            
        Returns:
            A list of texts with redundant entries removed.
        """
        if not texts:
            return []

        embeddings = self.encode(texts)
        kept_indices = []
        
        for i in range(len(embeddings)):
            if not kept_indices:
                kept_indices.append(i)
                continue
            
            # Vectorized similarity check: current embedding vs all kept embeddings
            current_emb = embeddings[i].reshape(1, -1)
            kept_embs = embeddings[kept_indices]
            
            # Use util.cos_sim for efficient matrix multiplication
            # returns a matrix of shape (1, len(kept_indices))
            similarities = util.cos_sim(current_emb, kept_embs)
            
            if np.max(similarities) <= threshold:
                kept_indices.append(i)
        
        return [texts[i] for i in kept_indices]

    def get_most_similar(self, query: str, documents: List[str], top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Finds the top_k most similar documents to a given query using matrix operations.
        """
        if not documents:
            return []

        query_emb = self.encode([query]) # shape (1, dim)
        doc_embs = self.encode(documents) # shape (num_docs, dim)
        
        # Vectorized similarity: (1, dim) x (num_docs, dim) -> (1, num_docs)
        similarities = util.cos_sim(query_emb, doc_embs)[0]
        
        # Pair documents with their similarity scores
        doc_sim_pairs = list(zip(documents, similarities.tolist()))
        
        # Sort by similarity descending
        results = sorted(doc_sim_pairs, key=lambda x: x[1], reverse=True)
        return results[:top_k]