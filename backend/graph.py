"""
CogniFuse — Knowledge Graph Module
NetworkX-based directed graph with topological ordering, root cause analysis,
and GNN-powered embeddings via GraphSAGE.
"""

import networkx as nx
from collections import deque
from gnn_service import GNNService
from search_service import SemanticSearchService


class KnowledgeGraph:
    """
    Directed knowledge graph using NetworkX.
    Nodes = concepts, Edges = relations between concepts.
    Tracks mastery per concept for adaptive learning.

    Enhanced with:
    - GNN (GraphSAGE) for node embeddings and link prediction
    - FAISS semantic search for meaning-based concept retrieval
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.mastery = {}  # concept -> mastery score (0-100)
        self.node_types = {} # concept -> 'base' | 'current'
        self.gnn = GNNService()
        self.search = SemanticSearchService()

    def reset_graph(self):
        """Clears all persistent data."""
        self.graph.clear()
        self.mastery.clear()
        self.node_types.clear()

    def load_json(self, data: dict):
        """Loads graph state from a JSON dictionary (from Supabase)."""
        if not data:
            return
        self.reset_graph()
        # Restore nodes & node types
        for node in data.get("nodes", []):
            self.graph.add_node(node["id"])
            self.mastery[node["id"]] = node.get("mastery", 0)
            self.node_types[node["id"]] = node.get("type", "current")
        # Restore edges
        for edge in data.get("edges", []):
            self.graph.add_edge(
                edge["source"], 
                edge["target"], 
                relation=edge.get("relation", "related_to"),
                predicted=edge.get("predicted", False),
                confidence=edge.get("confidence", 0)
            )
        # Re-run pipeline to rebuild embeddings and search index
        self._run_gnn_pipeline()

    def build_from_triplets(self, triplets: list[dict], concept_type: str = "current"):
        """
        Build graph from extracted triplets.
        accumulates into the existing persistent graph.
        """
        for triplet in triplets:
            subj = triplet["subject"]
            obj = triplet["object"]
            rel = triplet["relation"]

            # Add nodes with initial mastery of 0
            if subj not in self.graph:
                self.graph.add_node(subj)
                self.mastery[subj] = 0
                self.node_types[subj] = concept_type
            if obj not in self.graph:
                self.graph.add_node(obj)
                self.mastery[obj] = 0
                self.node_types[obj] = concept_type

            # Add directed edge: subject → object
            self.graph.add_edge(subj, obj, relation=rel)

        # ── Run GNN pipeline after graph is built ──
        self._run_gnn_pipeline()

    def _run_gnn_pipeline(self):
        """
        Train GNN, perform link prediction, and build semantic search index.
        This is the core intelligence layer of CogniFuse.
        """
        if len(self.graph.nodes()) < 2:
            print("[Graph] Too few nodes for GNN pipeline")
            return

        # Step 1: Train GraphSAGE embeddings
        print("[Graph] Starting GNN training...")
        self.gnn.train_embeddings(self.graph, self.mastery, epochs=100)

        # Step 2: Link prediction — discover implicit relationships
        predicted_links = self.gnn.predict_links(self.graph, threshold=0.7)
        for link in predicted_links[:10]:  # Add top 10 predicted links
            self.graph.add_edge(
                link["source"],
                link["target"],
                relation=link["relation"],
                predicted=True,
                confidence=link["confidence"]
            )
        if predicted_links:
            print(f"[Graph] Added {min(len(predicted_links), 10)} predicted edges to graph")

        # Step 3: Build FAISS semantic search index
        embeddings = self.gnn.get_all_embeddings()
        self.search.build_index(embeddings)

    def topological_order(self) -> list[str]:
        """
        Return concepts in topological order using Kahn's algorithm (via NetworkX).
        Foundational concepts come first, dependent concepts come later.
        Falls back to a best-effort ordering if the graph has cycles.
        """
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            # Graph has cycles — fallback: break cycles and try again
            print("[Graph] Cycle detected, using fallback ordering")
            # Use a simple approach: sort by in-degree (fewer prerequisites first)
            nodes_by_indegree = sorted(
                self.graph.nodes(),
                key=lambda n: self.graph.in_degree(n)
            )
            return nodes_by_indegree

    def reverse_bfs_root_cause(self, failed_concept: str) -> str | None:
        """
        Traverse predecessors backwards via BFS to find the deepest
        unmastered prerequisite (mastery < 50).
        Returns the root cause concept, or None if all prerequisites are mastered.
        """
        if failed_concept not in self.graph:
            return None

        visited = set()
        queue = deque([failed_concept])
        root_cause = None

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            # Check all predecessors (concepts that lead TO current)
            predecessors = list(self.graph.predecessors(current))
            for pred in predecessors:
                if pred not in visited:
                    # If this predecessor is unmastered, it's a candidate root cause
                    if self.mastery.get(pred, 0) < 50:
                        root_cause = pred  # Keep going deeper
                    queue.append(pred)

        return root_cause

    def update_mastery(self, concept: str, score: float):
        """Update mastery score for a concept (0-100)."""
        if concept in self.mastery:
            self.mastery[concept] = max(0, min(100, score))
        else:
            # Concept might not be in graph yet, add it
            self.mastery[concept] = max(0, min(100, score))

    def get_mastery(self, concept: str) -> float:
        """Get mastery score for a concept."""
        return self.mastery.get(concept, 0)

    def get_neighbors(self, concept: str) -> list[str]:
        """Get all connected concepts (predecessors + successors)."""
        if concept not in self.graph:
            return []
        preds = list(self.graph.predecessors(concept))
        succs = list(self.graph.successors(concept))
        return list(set(preds + succs))

    def get_graph_data(self) -> dict:
        """
        Return graph data for frontend visualization.
        Returns {
            nodes: [{id, mastery, embedding, cluster}],
            edges: [{source, target, relation, predicted}],
            gnn_active: bool
        }
        """
        # Get clusters from semantic search
        clusters = self.search.find_clusters(threshold=0.5) if self.search.is_built() else []
        concept_cluster = {}
        for i, cluster in enumerate(clusters):
            for concept in cluster:
                concept_cluster[concept] = i

        nodes = []
        for node in self.graph.nodes():
            node_data = {
                "id": node,
                "mastery": self.mastery.get(node, 0),
                "cluster": concept_cluster.get(node, 0),
                "type": self.node_types.get(node, "current")
            }
            # Include embedding info if available
            emb = self.gnn.get_embedding(node)
            if emb is not None:
                node_data["has_embedding"] = True
            nodes.append(node_data)

        edges = []
        for u, v, data in self.graph.edges(data=True):
            edge_data = {
                "source": u,
                "target": v,
                "relation": data.get("relation", "related_to"),
            }
            if data.get("predicted", False):
                edge_data["predicted"] = True
                edge_data["confidence"] = data.get("confidence", 0)
            edges.append(edge_data)

        return {
            "nodes": nodes,
            "edges": edges,
            "gnn_active": self.gnn.is_trained()
        }

    # ── GNN-powered methods ──────────────────────────────────

    def get_readiness_scores(self) -> dict:
        """Get GNN-computed readiness scores for all concepts."""
        return self.gnn.compute_readiness_scores(self.graph, self.mastery)

    def get_recommended_concepts(self, top_k: int = 5) -> list[dict]:
        """
        Return the top-K concepts the student should study next,
        ranked by GNN readiness score.
        """
        readiness = self.get_readiness_scores()
        sorted_concepts = sorted(readiness.items(), key=lambda x: x[1], reverse=True)

        recommendations = []
        for concept, score in sorted_concepts[:top_k]:
            recommendations.append({
                "concept": concept,
                "readiness_score": score,
                "current_mastery": self.mastery.get(concept, 0),
                "prerequisites_met": score > 50
            })
        return recommendations

    def semantic_search(self, concept: str, top_k: int = 5) -> list[dict]:
        """Search for semantically similar concepts."""
        return self.search.search(concept, top_k=top_k)

    def get_weak_areas(self) -> list[dict]:
        """Detect weak concept clusters using GNN embeddings."""
        return self.search.detect_weak_areas(self.mastery, threshold=0.5)

    def get_concept_clusters(self) -> list[list[str]]:
        """Get semantic clusters of related concepts."""
        return self.search.find_clusters(threshold=0.5)
