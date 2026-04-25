"""
CogniFuse — GNN Service (GraphSAGE)
Implements Graph Neural Network for:
  1. Node embedding generation (128-dim vectors)
  2. Link prediction (discovering implicit relationships)
  3. Readiness scoring (mastery propagation)
Uses PyTorch Geometric with GraphSAGE architecture.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import numpy as np
import networkx as nx


# ── GraphSAGE Model ─────────────────────────────────────────
class GraphSAGEModel(torch.nn.Module):
    """
    2-layer GraphSAGE encoder.
    Produces 128-dimensional node embeddings via neighbourhood aggregation.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 128):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# ── Link Predictor ──────────────────────────────────────────
class LinkPredictor(torch.nn.Module):
    """Simple MLP link predictor using dot-product of node embeddings."""

    def __init__(self, in_channels: int = 128):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels * 2, 64)
        self.lin2 = torch.nn.Linear(64, 1)

    def forward(self, z_src, z_dst):
        z = torch.cat([z_src, z_dst], dim=-1)
        z = F.relu(self.lin1(z))
        return torch.sigmoid(self.lin2(z))


# ── GNN Service (main interface) ────────────────────────────
class GNNService:
    """
    Manages the GraphSAGE model lifecycle:
    - Converts NetworkX graphs to PyTorch Geometric Data objects
    - Trains embeddings unsupervised
    - Performs link prediction
    - Computes readiness scores for adaptive learning
    """

    def __init__(self):
        self.model = None
        self.link_predictor = None
        self.embeddings = {}          # concept_name -> np.array (128-d)
        self.node_to_idx = {}         # concept_name -> int index
        self.idx_to_node = {}         # int index -> concept_name
        self._trained = False

    def _nx_to_pyg(self, G: nx.DiGraph, mastery: dict) -> Data:
        """
        Convert a NetworkX DiGraph into a PyTorch Geometric Data object.
        Node features = [in_degree, out_degree, mastery, pagerank, clustering_coeff]
        """
        nodes = list(G.nodes())
        if not nodes:
            return None

        self.node_to_idx = {n: i for i, n in enumerate(nodes)}
        self.idx_to_node = {i: n for n, i in self.node_to_idx.items()}

        # Build edge index
        edges_src, edges_dst = [], []
        for u, v in G.edges():
            edges_src.append(self.node_to_idx[u])
            edges_dst.append(self.node_to_idx[v])

        # Make undirected for message passing (add reverse edges)
        all_src = edges_src + edges_dst
        all_dst = edges_dst + edges_src

        if not all_src:
            # No edges — create self-loops
            all_src = list(range(len(nodes)))
            all_dst = list(range(len(nodes)))

        edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)

        # Compute PageRank safely
        try:
            pagerank = nx.pagerank(G, alpha=0.85)
        except Exception:
            pagerank = {n: 1.0 / len(nodes) for n in nodes}

        # Build node features: [in_deg, out_deg, mastery, pagerank, clustering]
        features = []
        for node in nodes:
            in_deg = G.in_degree(node)
            out_deg = G.out_degree(node)
            m = mastery.get(node, 0) / 100.0  # normalize to [0, 1]
            pr = pagerank.get(node, 0)
            # Clustering coefficient (treat as undirected)
            try:
                cc = nx.clustering(G.to_undirected(), node)
            except Exception:
                cc = 0.0
            features.append([float(in_deg), float(out_deg), m, pr, cc])

        x = torch.tensor(features, dtype=torch.float)
        return Data(x=x, edge_index=edge_index)

    def train_embeddings(self, G: nx.DiGraph, mastery: dict, epochs: int = 100):
        """
        Train GraphSAGE on the knowledge graph to produce node embeddings.
        Uses an unsupervised contrastive-style loss:
        - Positive pairs: connected nodes should have similar embeddings
        - Negative pairs: random nodes should have dissimilar embeddings
        """
        data = self._nx_to_pyg(G, mastery)
        if data is None or data.x.size(0) < 2:
            print("[GNN] Graph too small for training, using random embeddings")
            self._assign_random_embeddings(G)
            return

        in_channels = data.x.size(1)  # 5 features
        self.model = GraphSAGEModel(in_channels=in_channels)
        self.link_predictor = LinkPredictor(in_channels=128)

        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.link_predictor.parameters()),
            lr=0.01
        )

        self.model.train()
        self.link_predictor.train()

        num_nodes = data.x.size(0)
        num_edges = data.edge_index.size(1)

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass — get embeddings
            z = self.model(data.x, data.edge_index)

            # Positive samples: actual edges
            pos_src = data.edge_index[0]
            pos_dst = data.edge_index[1]
            pos_pred = self.link_predictor(z[pos_src], z[pos_dst])

            # Negative samples: random node pairs
            neg_src = torch.randint(0, num_nodes, (num_edges,))
            neg_dst = torch.randint(0, num_nodes, (num_edges,))
            neg_pred = self.link_predictor(z[neg_src], z[neg_dst])

            # Binary cross-entropy loss
            pos_loss = F.binary_cross_entropy(pos_pred.squeeze(), torch.ones(pos_pred.size(0)))
            neg_loss = F.binary_cross_entropy(neg_pred.squeeze(), torch.zeros(neg_pred.size(0)))
            loss = pos_loss + neg_loss

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 25 == 0:
                print(f"[GNN] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        # Extract final embeddings
        self.model.eval()
        with torch.no_grad():
            z = self.model(data.x, data.edge_index)
            for i, node in self.idx_to_node.items():
                self.embeddings[node] = z[i].numpy()

        self._trained = True
        print(f"[GNN] Training complete. {len(self.embeddings)} embeddings generated.")

    def _assign_random_embeddings(self, G: nx.DiGraph):
        """Fallback: assign random embeddings for very small graphs."""
        for node in G.nodes():
            self.embeddings[node] = np.random.randn(128).astype(np.float32)
        self._trained = True

    def predict_links(self, G: nx.DiGraph, threshold: float = 0.7) -> list[dict]:
        """
        Use the trained link predictor to find missing relationships.
        Returns list of {source, target, confidence} for predicted edges.
        """
        if not self._trained or self.model is None:
            return []

        nodes = list(G.nodes())
        predictions = []

        self.model.eval()
        self.link_predictor.eval()

        with torch.no_grad():
            for i, src in enumerate(nodes):
                for j, dst in enumerate(nodes):
                    if i == j:
                        continue
                    # Skip existing edges
                    if G.has_edge(src, dst):
                        continue

                    src_emb = torch.tensor(self.embeddings[src]).unsqueeze(0)
                    dst_emb = torch.tensor(self.embeddings[dst]).unsqueeze(0)
                    score = self.link_predictor(src_emb, dst_emb).item()

                    if score >= threshold:
                        predictions.append({
                            "source": src,
                            "target": dst,
                            "confidence": round(score, 3),
                            "relation": "implicitly_related_to"
                        })

        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        print(f"[GNN] Link prediction found {len(predictions)} implicit relationships")
        return predictions

    def compute_readiness_scores(self, G: nx.DiGraph, mastery: dict) -> dict:
        """
        Compute a 'readiness score' for each concept using GNN embeddings +
        mastery propagation. This helps determine which concepts the student
        is ready to learn next.

        Readiness = weighted combination of:
          - Own mastery
          - Average mastery of prerequisites (weighted by embedding similarity)
          - Graph centrality (important hubs are harder to be ready for)
        """
        if not self._trained:
            # Fallback: use simple mastery-based ordering
            return {n: mastery.get(n, 0) for n in G.nodes()}

        readiness = {}
        for node in G.nodes():
            own_mastery = mastery.get(node, 0) / 100.0

            # Get predecessor mastery weighted by embedding similarity
            predecessors = list(G.predecessors(node))
            if predecessors and node in self.embeddings:
                node_emb = self.embeddings[node]
                weighted_prereq_mastery = 0
                total_weight = 0

                for pred in predecessors:
                    if pred in self.embeddings:
                        # Cosine similarity as weight
                        pred_emb = self.embeddings[pred]
                        sim = np.dot(node_emb, pred_emb) / (
                            np.linalg.norm(node_emb) * np.linalg.norm(pred_emb) + 1e-8
                        )
                        sim = max(0, sim)  # clamp negatives
                        pred_mastery = mastery.get(pred, 0) / 100.0
                        weighted_prereq_mastery += sim * pred_mastery
                        total_weight += sim

                prereq_score = weighted_prereq_mastery / (total_weight + 1e-8)
            else:
                # No prerequisites — fully ready
                prereq_score = 1.0

            # Combine: student is "ready" if they haven't mastered it yet
            # but have mastered prerequisites
            if own_mastery >= 0.67:
                # Already mastered — low priority
                readiness[node] = 0.1
            else:
                # Readiness = how much prerequisites are done * (1 - own mastery)
                readiness[node] = round(prereq_score * (1.0 - own_mastery) * 100, 1)

        return readiness

    def get_embedding(self, concept: str) -> np.ndarray | None:
        """Get the 128-dim embedding vector for a concept."""
        return self.embeddings.get(concept)

    def get_all_embeddings(self) -> dict:
        """Get all embeddings as {concept: list[float]}."""
        return {k: v.tolist() for k, v in self.embeddings.items()}

    def is_trained(self) -> bool:
        return self._trained
