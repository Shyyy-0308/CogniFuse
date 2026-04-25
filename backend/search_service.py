"""
CogniFuse — Semantic Search Service (FAISS)
Provides meaning-based concept search using GNN-generated embeddings.
Uses FAISS for fast approximate nearest-neighbor retrieval.
"""

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[Search] FAISS not available — falling back to brute-force cosine search")


class SemanticSearchService:
    """
    Builds and queries a FAISS index over GNN node embeddings.
    Enables:
      - Concept search by meaning (not keyword)
      - Finding related concept clusters
      - Detecting weak areas via embedding neighbourhood analysis
    """

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.index = None
        self.concepts = []         # ordered list matching FAISS index
        self.embeddings = {}       # concept -> np.array
        self._built = False

    def build_index(self, embeddings: dict):
        """
        Build the FAISS index from GNN embeddings.
        embeddings: {concept_name: np.array or list[float]}
        """
        if not embeddings:
            print("[Search] No embeddings to index")
            return

        self.concepts = list(embeddings.keys())
        vectors = []
        for concept in self.concepts:
            emb = embeddings[concept]
            if isinstance(emb, list):
                emb = np.array(emb, dtype=np.float32)
            vectors.append(emb.astype(np.float32))

        matrix = np.stack(vectors)

        # Normalize for cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        matrix = matrix / norms

        self.embeddings = {c: matrix[i] for i, c in enumerate(self.concepts)}

        if FAISS_AVAILABLE:
            # Use Inner Product index (cosine similarity after normalization)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(matrix)
            print(f"[Search] FAISS index built with {len(self.concepts)} concepts")
        else:
            print(f"[Search] Brute-force index built with {len(self.concepts)} concepts")

        self._built = True

    def search(self, query_concept: str, top_k: int = 5) -> list[dict]:
        """
        Find the most semantically similar concepts to a query concept.
        Returns list of {concept, similarity} sorted by similarity descending.
        """
        if not self._built or query_concept not in self.embeddings:
            return []

        query_vec = self.embeddings[query_concept].reshape(1, -1).astype(np.float32)

        if FAISS_AVAILABLE and self.index is not None:
            scores, indices = self.index.search(query_vec, min(top_k + 1, len(self.concepts)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                concept = self.concepts[idx]
                if concept == query_concept:
                    continue
                results.append({
                    "concept": concept,
                    "similarity": round(float(score), 4)
                })
            return results[:top_k]
        else:
            # Brute-force fallback
            return self._brute_force_search(query_vec, query_concept, top_k)

    def search_by_text(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Search using an arbitrary embedding vector (e.g. from an external model).
        """
        if not self._built:
            return []

        query_vec = query_embedding.reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        if FAISS_AVAILABLE and self.index is not None:
            scores, indices = self.index.search(query_vec, min(top_k, len(self.concepts)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                results.append({
                    "concept": self.concepts[idx],
                    "similarity": round(float(score), 4)
                })
            return results
        else:
            return self._brute_force_search(query_vec, None, top_k)

    def find_clusters(self, threshold: float = 0.6) -> list[list[str]]:
        """
        Group concepts into semantic clusters based on embedding similarity.
        Uses a simple greedy clustering approach.
        """
        if not self._built:
            return []

        visited = set()
        clusters = []

        for concept in self.concepts:
            if concept in visited:
                continue
            cluster = [concept]
            visited.add(concept)

            similar = self.search(concept, top_k=len(self.concepts))
            for item in similar:
                if item["concept"] not in visited and item["similarity"] >= threshold:
                    cluster.append(item["concept"])
                    visited.add(item["concept"])

            clusters.append(cluster)

        # Sort by cluster size (largest first)
        clusters.sort(key=len, reverse=True)
        return clusters

    def detect_weak_areas(self, mastery: dict, threshold: float = 0.5) -> list[dict]:
        """
        Detect "weak zones" by finding clusters of low-mastery concepts.
        Returns list of {cluster, avg_mastery, concepts} sorted by weakest first.
        """
        if not self._built:
            return []

        clusters = self.find_clusters(threshold=threshold)
        weak_areas = []

        for cluster in clusters:
            masteries = [mastery.get(c, 0) for c in cluster]
            avg_m = sum(masteries) / len(masteries) if masteries else 0

            weak_areas.append({
                "concepts": cluster,
                "avg_mastery": round(avg_m, 1),
                "size": len(cluster)
            })

        # Sort by avg mastery (weakest first)
        weak_areas.sort(key=lambda x: x["avg_mastery"])
        return weak_areas

    def _brute_force_search(self, query_vec, exclude_concept, top_k):
        """Fallback cosine similarity search without FAISS."""
        results = []
        for concept in self.concepts:
            if concept == exclude_concept:
                continue
            emb = self.embeddings[concept]
            sim = float(np.dot(query_vec.flatten(), emb.flatten()))
            results.append({"concept": concept, "similarity": round(sim, 4)})

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def is_built(self) -> bool:
        return self._built
