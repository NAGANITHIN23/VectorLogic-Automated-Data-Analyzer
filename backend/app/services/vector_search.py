import faiss
import numpy as np
from typing import Optional, List, Dict
from sqlalchemy.ext.asyncio import AsyncSession

class VectorSearchService:
    def __init__(self, db: Optional[AsyncSession] = None, dimension: int = 768):
        self.db = db
        # Initialize a simple L2 distance index
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = [] # To store IDs or text snippets mapped to vectors

    def add_to_index(self, vectors: np.ndarray, doc_metadata: List[Dict]):
        """Adds new embeddings to the FAISS index."""
        if vectors.dtype != 'float32':
            vectors = vectors.astype('float32')
        self.index.add(vectors)
        self.metadata.extend(doc_metadata)

    def search(self, query_vector: np.ndarray, k: int = 5):
        """Searches for the top K most similar vectors."""
        if query_vector.dtype != 'float32':
            query_vector = query_vector.astype('float32')
        
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1:
                results.append({
                    "metadata": self.metadata[idx],
                    "distance": float(distances[0][i])
                })
        return results
