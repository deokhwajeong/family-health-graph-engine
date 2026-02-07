"""
Link Prediction module for the Household Health Graph Engine.

Predicts potential relationships (edges) between nodes that don't yet exist.
For example: food-health_outcome relationships.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import json


class HealthLinkPredictor:
    """
    Predicts missing links in the health graph, particularly:
    - Food -> Health metric relationships
    - Activity -> Sleep quality relationships
    - Nutrition pattern -> Energy level relationships
    """
    
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names: List[str] = []
    
    def _extract_node_pair_features(
        self, G: nx.DiGraph, node1: str, node2: str
    ) -> np.ndarray:
        """
        Extract similarity/relationship features between two nodes.
        """
        n1_data = G.nodes[node1]
        n2_data = G.nodes[node2]
        
        features = []
        
        # Type compatibility score
        type1 = n1_data.get("type", "")
        type2 = n2_data.get("type", "")
        
        type_pairs = {
            ("food", "metric"): 1.0,
            ("food", "activity"): 0.5,
            ("sleep", "metric"): 1.0,
            ("activity", "metric"): 0.8,
            ("meal", "metric"): 0.7,
        }
        
        type_compatibility = type_pairs.get((type1, type2), 0.0)
        features.append(type_compatibility)
        
        # Common neighbors (co-occurrence)
        neighbors1 = set(G.successors(node1)) | set(G.predecessors(node1))
        neighbors2 = set(G.successors(node2)) | set(G.predecessors(node2))
        common = len(neighbors1 & neighbors2)
        features.append(min(common / 10.0, 1.0))  # normalize
        
        # Nutritional compatibility (for food nodes)
        if type1 == "food" and type2 == "metric":
            protein1 = n1_data.get("protein_g", 0)
            fiber1 = n1_data.get("fiber_g", 0)
            sugar1 = n1_data.get("sugar_g", 0)
            
            # Metrics that correlate with good nutrition
            if "focus" in n2_data.get("metric_type", "").lower():
                # high protein -> high focus
                features.append(min(protein1 / 30.0, 1.0))
            elif "energy" in n2_data.get("metric_type", "").lower():
                # balanced nutrition -> energy
                nutrition_score = (protein1 + fiber1) / 50.0
                features.append(min(nutrition_score, 1.0))
            else:
                features.append(0.5)
        else:
            features.append(0.5)
        
        # Distance in graph (approximate via neighbor overlap instead of shortest path)
        # Using Jaccard-like measure — much faster than shortest_path_length
        neighbors1_set = set(G.successors(node1)) | set(G.predecessors(node1))
        neighbors2_set = set(G.successors(node2)) | set(G.predecessors(node2))
        union_size = len(neighbors1_set | neighbors2_set)
        if union_size > 0:
            jaccard = len(neighbors1_set & neighbors2_set) / union_size
            features.append(jaccard)
        else:
            features.append(0.0)
        
        # Temporal correlation (for time-based nodes)
        if "date" in n1_data and "date" in n2_data:
            date1 = n1_data.get("date", "")
            date2 = n2_data.get("date", "")
            # Same day = 1.0, 1 day apart = 0.8, etc.
            if date1 == date2:
                features.append(1.0)
            elif date1 and date2:
                try:
                    from datetime import date
                    d1 = date.fromisoformat(date1)
                    d2 = date.fromisoformat(date2)
                    days_apart = abs((d1 - d2).days)
                    features.append(max(0, 1.0 - days_apart / 7.0))
                except:
                    features.append(0.0)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        return np.array(features)
    
    def fit(self, G: nx.DiGraph) -> None:
        """
        Fit link prediction model using existing edges as positive examples
        and randomly sampled non-existing edges as negative examples.
        Uses sampling to avoid O(N²) complexity.
        """
        import random
        random.seed(42)
        
        X_list = []
        y_list = []
        
        nodes = list(G.nodes())
        
        # Positive examples: sample edges if too many
        all_edges = list(G.edges())
        max_pos = min(len(all_edges), 2000)
        sampled_edges = random.sample(all_edges, max_pos) if len(all_edges) > max_pos else all_edges
        
        for u, v in sampled_edges:
            features = self._extract_node_pair_features(G, u, v)
            X_list.append(features)
            y_list.append(1)
        
        # Negative examples: random sampling (not O(N²) iteration)
        existing_edges = set(G.edges()) | set((v, u) for u, v in G.edges())
        max_negatives = len(sampled_edges) * 2
        negative_count = 0
        attempts = 0
        max_attempts = max_negatives * 10
        
        while negative_count < max_negatives and attempts < max_attempts:
            u = random.choice(nodes)
            v = random.choice(nodes)
            if u != v and (u, v) not in existing_edges and (v, u) not in existing_edges:
                features = self._extract_node_pair_features(G, u, v)
                X_list.append(features)
                y_list.append(0)
                negative_count += 1
            attempts += 1
        
        if len(X_list) < 5:
            raise ValueError("Not enough examples to fit link prediction model")
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        self.feature_names = [
            "type_compatibility", "common_neighbors", "nutritional_fit",
            "graph_distance", "temporal_proximity"
        ]
        
        # Fit scaler and model
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        self.is_fitted = True
    
    def predict_links(self, G: nx.DiGraph, top_k: int = 10) -> List[Dict]:
        """
        Predict potential new links that should be added to the graph.
        Only checks semantically meaningful type pairs to avoid O(N²).
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        predictions = []
        existing_edges = set(G.edges()) | set((v, u) for u, v in G.edges())
        
        # Only check meaningful type pairs (not all N² pairs)
        interesting_pairs = [
            ("food", "metric"),
            ("food", "activity"),
            ("sleep", "metric"),
            ("activity", "metric"),
            ("meal", "metric"),
        ]
        
        # Group nodes by type
        nodes_by_type: Dict[str, List[str]] = {}
        for n in G.nodes():
            t = G.nodes[n].get("type", "")
            nodes_by_type.setdefault(t, []).append(n)
        
        import random
        random.seed(42)
        
        for type1, type2 in interesting_pairs:
            src_nodes = nodes_by_type.get(type1, [])
            tgt_nodes = nodes_by_type.get(type2, [])
            
            # Sample if too many candidates
            max_candidates = 200
            if len(src_nodes) * len(tgt_nodes) > max_candidates:
                sampled_src = random.sample(src_nodes, min(len(src_nodes), 30))
                sampled_tgt = random.sample(tgt_nodes, min(len(tgt_nodes), 30))
            else:
                sampled_src, sampled_tgt = src_nodes, tgt_nodes
            
            for u in sampled_src:
                for v in sampled_tgt:
                    if (u, v) in existing_edges:
                        continue
                    features = self._extract_node_pair_features(G, u, v)
                    X_scaled = self.scaler.transform(features.reshape(1, -1))
                    prob = self.model.predict_proba(X_scaled)[0, 1]
                    
                    u_data = G.nodes[u]
                    v_data = G.nodes[v]
                    predictions.append({
                        "source": u,
                        "source_type": u_data.get("type"),
                        "source_label": u_data.get("name", u),
                        "target": v,
                        "target_type": v_data.get("type"),
                        "target_label": v_data.get("name", v),
                        "probability": float(prob),
                        "reasoning": self._explain_prediction(G, u, v, prob)
                    })
        
        predictions.sort(key=lambda x: x["probability"], reverse=True)
        return predictions[:top_k]
    
    def _explain_prediction(self, G: nx.DiGraph, u: str, v: str, 
                           prob: float) -> str:
        """Generate an explanation for a predicted link."""
        reasons = []
        
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        
        u_type = u_data.get("type", "")
        v_type = v_data.get("type", "")
        
        if u_type == "food" and v_type == "metric":
            metric_type = v_data.get("metric_type", "")
            protein = u_data.get("protein_g", 0)
            
            if "focus" in metric_type and protein > 15:
                reasons.append("High protein content likely improves focus")
            elif "energy" in metric_type:
                fiber = u_data.get("fiber_g", 0)
                if fiber > 3:
                    reasons.append("Good fiber content supports energy")
        
        elif u_type == "sleep" and v_type == "metric":
            reasons.append("Sleep quality directly impacts daily metrics")
        
        elif u_type == "activity" and v_type == "metric":
            reasons.append("Physical activity correlates with health metrics")
        
        return " + ".join(reasons) if reasons else "Structural similarity in graph"


class NodePairSimilarity:
    """
    Computes similarity between node pairs for link recommendation.
    """
    
    @staticmethod
    def cosine_similarity(G: nx.DiGraph, node1: str, node2: str) -> float:
        """
        Compute cosine similarity based on common neighbors.
        """
        neighbors1 = set(G.successors(node1)) | set(G.predecessors(node1))
        neighbors2 = set(G.successors(node2)) | set(G.predecessors(node2))
        
        common = len(neighbors1 & neighbors2)
        
        if len(neighbors1) == 0 or len(neighbors2) == 0:
            return 0.0
        
        return common / np.sqrt(len(neighbors1) * len(neighbors2))
    
    @staticmethod
    def jaccard_similarity(G: nx.DiGraph, node1: str, node2: str) -> float:
        """
        Compute Jaccard similarity of neighborhoods.
        """
        neighbors1 = set(G.successors(node1)) | set(G.predecessors(node1))
        neighbors2 = set(G.successors(node2)) | set(G.predecessors(node2))
        
        if len(neighbors1 | neighbors2) == 0:
            return 0.0
        
        return len(neighbors1 & neighbors2) / len(neighbors1 | neighbors2)
