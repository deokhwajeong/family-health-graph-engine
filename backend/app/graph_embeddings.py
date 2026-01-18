"""
Graph Embedding module for the Household Health Graph Engine.

Uses GraphSAGE-inspired approach to generate embeddings for nodes
representing daily health patterns.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
import json


class SimpleGraphSAGE:
    """
    Simplified GraphSAGE-like embedding model for the household health graph.
    
    Features:
    - Aggregates neighbor node features
    - Learns daily embedding that captures health patterns
    - Interpretable aggregation of meal, sleep, activity data
    """
    
    def __init__(self, embedding_dim: int = 16, num_aggregation_layers: int = 2):
        self.embedding_dim = embedding_dim
        self.num_layers = num_aggregation_layers
        self.embeddings: Dict[str, np.ndarray] = {}
        
    def _extract_node_features(self, G: nx.DiGraph, node_id: str) -> np.ndarray:
        """
        Extract a feature vector for a node based on its attributes and connections.
        """
        node = G.nodes[node_id]
        node_type = node.get("type", "unknown")
        
        features = []
        
        # Node type one-hot encoding (8 types max)
        type_map = {
            "household": 0, "person": 1, "day": 2, "meal": 3,
            "food": 4, "sleep": 5, "activity": 6, "metric": 7
        }
        type_idx = type_map.get(node_type, 7)
        type_onehot = np.zeros(8)
        type_onehot[type_idx] = 1.0
        features.append(type_onehot)
        
        # Node-specific features
        if node_type == "day":
            # weekday, is_weekend, is_school_day
            weekday = node.get("weekday", 0) / 7.0  # normalize [0, 1]
            is_weekend = float(node.get("is_weekend", False))
            is_school = float(node.get("is_school_day", True))
            features.append(np.array([weekday, is_weekend, is_school]))
        
        elif node_type == "meal":
            # meal_type encoding
            meal_types = {"breakfast": 0, "lunch": 1, "dinner": 2, "snack": 3}
            meal_type = node.get("meal_type", "snack")
            meal_idx = meal_types.get(meal_type, 3) / 4.0
            features.append(np.array([meal_idx]))
        
        elif node_type == "sleep":
            # sleep quality & duration (normalized)
            duration = node.get("duration_hours", 8.0) / 12.0  # [0, 1]
            quality = node.get("quality_score", 80.0) / 100.0  # [0, 1]
            features.append(np.array([duration, quality]))
        
        elif node_type == "activity":
            # intensity & duration
            intensity_map = {"low": 0, "medium": 0.5, "high": 1.0}
            intensity = intensity_map.get(node.get("intensity", "low"), 0)
            duration = min(node.get("duration_min", 30) / 120.0, 1.0)
            features.append(np.array([intensity, duration]))
        
        elif node_type == "metric":
            # metric value (normalized to [0, 1])
            value = node.get("value", 50.0)
            metric_type = node.get("metric_type", "")
            
            # normalize based on typical ranges
            if "energy" in metric_type or "focus" in metric_type or "mood" in metric_type:
                normalized = min(value / 100.0, 1.0)
            else:
                normalized = min(value / 200.0, 1.0)
            
            features.append(np.array([normalized]))
        
        elif node_type == "food":
            # nutrition profile
            protein = min(node.get("protein_g", 5) / 30.0, 1.0)
            fiber = min(node.get("fiber_g", 2) / 10.0, 1.0)
            sugar = min(node.get("sugar_g", 5) / 30.0, 1.0)
            features.append(np.array([protein, fiber, sugar]))
        
        else:
            # default: zeros
            features.append(np.array([0.0]))
        
        # Concatenate all features
        feature_vector = np.concatenate(features)
        
        # Pad or truncate to consistent dimension
        if len(feature_vector) < self.embedding_dim:
            feature_vector = np.pad(feature_vector, 
                                   (0, self.embedding_dim - len(feature_vector)))
        else:
            feature_vector = feature_vector[:self.embedding_dim]
        
        return feature_vector
    
    def _aggregate_neighbors(self, G: nx.DiGraph, node_id: str, 
                           layer: int, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Aggregate embeddings from neighboring nodes.
        """
        neighbors = list(G.successors(node_id)) + list(G.predecessors(node_id))
        
        if not neighbors:
            return embeddings.get(node_id, np.zeros(self.embedding_dim))
        
        neighbor_embeddings = [
            embeddings.get(n, np.zeros(self.embedding_dim)) 
            for n in neighbors
        ]
        
        # Mean aggregation
        aggregated = np.mean(neighbor_embeddings, axis=0)
        
        # Combine with current embedding
        current = embeddings.get(node_id, np.zeros(self.embedding_dim))
        combined = (current + aggregated) / 2.0
        
        return combined
    
    def fit_embeddings(self, G: nx.DiGraph) -> Dict[str, np.ndarray]:
        """
        Learn embeddings for all nodes in the graph.
        
        Returns:
            Dictionary mapping node_id -> embedding vector
        """
        # Initialize with node features
        embeddings = {}
        for node_id in G.nodes():
            embeddings[node_id] = self._extract_node_features(G, node_id)
        
        # Aggregate over multiple layers
        for layer in range(self.num_layers):
            new_embeddings = {}
            for node_id in G.nodes():
                new_embeddings[node_id] = self._aggregate_neighbors(
                    G, node_id, layer, embeddings
                )
            embeddings = new_embeddings
        
        self.embeddings = embeddings
        return embeddings
    
    def get_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """Get the learned embedding for a specific node."""
        return self.embeddings.get(node_id)
    
    def get_daily_embedding(self, G: nx.DiGraph, person_id: str, 
                           date: str) -> Optional[np.ndarray]:
        """
        Get the embedding for a specific day of a person.
        Aggregates all activity-related nodes for that day.
        """
        # Normalize person_id (accepts 'yooni' or 'person:yooni')
        normalized = person_id.split(":")[-1] if ":" in person_id else person_id
        day_id = f"day:{normalized}:{date}"
        if day_id not in self.embeddings:
            return None
        
        day_embedding = self.embeddings[day_id].copy()
        
        # Aggregate related meal, sleep, activity nodes
        related_nodes = list(G.successors(day_id))
        related_embeddings = [
            self.embeddings.get(n, np.zeros(self.embedding_dim)) 
            for n in related_nodes
        ]
        
        if related_embeddings:
            aggregated = np.mean(related_embeddings, axis=0)
            day_embedding = (day_embedding + aggregated) / 2.0
        
        return day_embedding
    
    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(emb1, emb2) / (norm1 * norm2))


def extract_day_features(G: nx.DiGraph, person_id: str, date: str) -> Dict[str, float]:
    """
    Extract interpretable health features for a specific day.
    
    Returns:
        Dictionary with keys like 'total_calories', 'sleep_quality', 'activity_intensity', etc.
    """
    # Normalize person_id (accepts 'yooni' or 'person:yooni')
    normalized = person_id.split(":")[-1] if ":" in person_id else person_id
    day_id = f"day:{normalized}:{date}"
    features = {}
    
    if day_id not in G.nodes:
        return features
    
    # Meals on this day
    day_node = G.nodes[day_id]
    meal_nodes = list(G.successors(day_id))
    
    total_calories = 0.0
    total_protein = 0.0
    total_fiber = 0.0
    total_sugar = 0.0
    meal_count = 0
    
    for meal_id in meal_nodes:
        if G.nodes[meal_id].get("type") == "meal":
            meal_count += 1
            # Get foods in this meal
            foods = list(G.successors(meal_id))
            for food_id in foods:
                if G.nodes[food_id].get("type") == "food":
                    food_node = G.nodes[food_id]
                    total_calories += food_node.get("kcal", 0)
                    total_protein += food_node.get("protein_g", 0)
                    total_fiber += food_node.get("fiber_g", 0)
                    total_sugar += food_node.get("sugar_g", 0)
    
    features["total_calories"] = total_calories
    features["total_protein_g"] = total_protein
    features["total_fiber_g"] = total_fiber
    features["total_sugar_g"] = total_sugar
    features["meal_count"] = meal_count
    
    # Sleep on this day
    sleep_nodes = [n for n in meal_nodes if G.nodes[n].get("type") == "sleep"]
    if sleep_nodes:
        sleep_node = G.nodes[sleep_nodes[0]]
        features["sleep_duration_h"] = sleep_node.get("duration_hours", 8.0)
        features["sleep_quality"] = sleep_node.get("quality_score", 80.0)
    
    # Activity on this day
    activity_nodes = [n for n in meal_nodes if G.nodes[n].get("type") == "activity"]
    features["activity_count"] = len(activity_nodes)
    features["total_calories_burned"] = 0.0
    features["max_intensity"] = 0.0
    
    intensity_map = {"low": 1, "medium": 2, "high": 3}
    for act_id in activity_nodes:
        act_node = G.nodes[act_id]
        features["total_calories_burned"] += act_node.get("calories_burned", 0)
        intensity = intensity_map.get(act_node.get("intensity", "low"), 1)
        features["max_intensity"] = max(features["max_intensity"], intensity)
    
    # Metrics on this day
    metric_nodes = [n for n in meal_nodes if G.nodes[n].get("type") == "metric"]
    for met_id in metric_nodes:
        met_node = G.nodes[met_id]
        metric_type = met_node.get("metric_type", "unknown")
        features[f"metric_{metric_type}"] = met_node.get("value", 0.0)
    
    return features
