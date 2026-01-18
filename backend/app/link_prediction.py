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
        
        # Distance in graph (shorter = more likely to have relationship)
        try:
            distance = nx.shortest_path_length(G.to_undirected(), node1, node2)
            features.append(max(0, 1.0 - distance / 10.0))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
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
        and non-existing edges as negative examples.
        """
        X_list = []
        y_list = []
        
        # Get all node pairs
        nodes = list(G.nodes())
        
        # Positive examples (existing edges)
        for u, v in G.edges():
            features = self._extract_node_pair_features(G, u, v)
            X_list.append(features)
            y_list.append(1)  # positive
        
        # Negative examples (non-existing edges)
        # Sample random non-edges
        existing_edges = set(G.edges()) | set((v, u) for u, v in G.edges())
        
        negative_count = 0
        max_negatives = len(G.edges()) * 3  # 3:1 ratio
        
        for i, u in enumerate(nodes):
            for v in nodes[i+1:]:
                if (u, v) not in existing_edges and (v, u) not in existing_edges:
                    features = self._extract_node_pair_features(G, u, v)
                    X_list.append(features)
                    y_list.append(0)  # negative
                    
                    negative_count += 1
                    if negative_count >= max_negatives:
                        break
            if negative_count >= max_negatives:
                break
        
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
        
        Args:
            G: NetworkX graph
            top_k: number of top predictions to return
        
        Returns:
            List of predicted links with scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        predictions = []
        
        # Get existing edges
        existing_edges = set(G.edges()) | set((v, u) for u, v in G.edges())
        
        # Check all non-existing edges
        nodes = list(G.nodes())
        
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if i != j and (u, v) not in existing_edges:
                    features = self._extract_node_pair_features(G, u, v)
                    X_scaled = self.scaler.transform(features.reshape(1, -1))
                    
                    # Get probability of link
                    prob = self.model.predict_proba(X_scaled)[0, 1]
                    
                    if prob > 0.3:  # threshold
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
        
        # Sort by probability and return top K
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
