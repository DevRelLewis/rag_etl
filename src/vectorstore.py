import os
import pickle
from typing import List, Dict, Any, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from transform_rules import DocumentMetadata


class DocumentChunk(BaseModel):
    content: str
    metadata: DocumentMetadata
    embedding: Optional[List[float]] = None


class VectorStore:

    def __init__(self, embedding_model: str, index_path: str, dimension: int = 384):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index_path = index_path
        self.dimension = dimension
        self.index = None
        self.documents = []

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        self._load_or_create_index()

    def _load_or_create_index(self):
        if os.path.exists(f"{self.index_path}.index"):
            self.index = faiss.read_index(f"{self.index_path}.index")
            with open(f"{self.index_path}.docs", 'rb') as f:
                self.documents = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.documents = []

    def add_documents(self, chunks: List[DocumentChunk]):
        embeddings = []

        for chunk in chunks:
            if not chunk.embedding:
                embedding = self.embedding_model.encode(chunk.content)
                chunk.embedding = embedding.tolist()
            embeddings.append(np.array(chunk.embedding))

        if embeddings:
            embeddings_matrix = np.vstack(embeddings).astype('float32')
            faiss.normalize_L2(embeddings_matrix)

            self.index.add(embeddings_matrix)
            self.documents.extend(chunks)
            self._save_index()

    def similarity_search(self, query: str, k: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        query_embedding = self.embedding_model.encode([query])
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold and idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    'content': doc.content,
                    'metadata': doc.metadata.model_dump(),
                    'score': float(score)
                })

        return results

    def weighted_search(self, query: str, k: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
        raw_results = self.similarity_search(query, k * 2, threshold)

        for result in raw_results:
            weight = result['metadata']['weight']
            result['weighted_score'] = result['score'] * weight

        weighted_results = sorted(raw_results, key=lambda x: x['weighted_score'], reverse=True)
        return weighted_results[:k]

    def _save_index(self):
        faiss.write_index(self.index, f"{self.index_path}.index")
        with open(f"{self.index_path}.docs", 'wb') as f:
            pickle.dump(self.documents, f)

    def get_stats(self) -> Dict[str, Any]:
        source_counts = {}
        for doc in self.documents:
            source = doc.metadata.source_system
            source_counts[source] = source_counts.get(source, 0) + 1

        return {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal,
            'source_breakdown': source_counts
        }