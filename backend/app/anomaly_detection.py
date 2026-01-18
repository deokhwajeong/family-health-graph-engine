"""
Anomaly Detection module for the Household Health Graph Engine.

Detects unusual health patterns using Isolation Forest and Local Outlier Factor.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import json


class HealthAnomalyDetector:
    """
    Detects anomalies in daily health patterns using multiple algorithms.
    
    Features:
    - Isolation Forest: detects global outliers
    - Local Outlier Factor: detects local density-based anomalies
    - Multi-feature analysis: combines nutrition, sleep, activity
    """
    
    def __init__(self, contamination: float = 0.1):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: expected proportion of anomalies (0.1 = 10%)
        """
        self.contamination = contamination
        self.iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        self.lof = LocalOutlierFactor(
            n_neighbors=min(20, max(3, int(10 / contamination))),
            contamination=contamination
        )
        self.scaler = StandardScaler()
        
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.anomaly_scores: Dict[str, float] = {}
    
    def _extract_features_from_graph(
        self, G: nx.DiGraph, person_id: str
    ) -> Tuple[List[Dict[str, float]], List[str]]:
        """
        Extract features for all days of a person from the graph.
        
        Returns:
            Tuple of (feature_dicts, sorted_dates)
        """
        from .graph_embeddings import extract_day_features
        
        # Find all day nodes for this person
        day_nodes = [
            node for node in G.nodes()
            if G.nodes[node].get("type") == "day" and 
            node.startswith(f"day:{person_id}:")
        ]
        
        features_list = []
        dates = []
        
        for day_id in sorted(day_nodes):
            date = day_id.split(":")[-1]
            features = extract_day_features(G, person_id, date)
            features_list.append(features)
            dates.append(date)
        
        return features_list, dates
    
    def fit(self, G: nx.DiGraph, person_id: str) -> None:
        """
        Fit anomaly detectors on historical data for a person.
        
        Args:
            G: NetworkX graph
            person_id: ID of the person to analyze
        """
        features_list, dates = self._extract_features_from_graph(G, person_id)
        
        if len(features_list) < 3:
            raise ValueError("Need at least 3 days of data to fit anomaly detector")
        
        # Convert to feature matrix
        X, feature_names = self._features_to_matrix(features_list)
        self.feature_names = feature_names
        
        # Fit models
        X_scaled = self.scaler.fit_transform(X)
        self.iso_forest.fit(X_scaled)
        self.lof.fit(X_scaled)
        
        self.is_fitted = True
        
        # Compute anomaly scores
        iso_scores = self.iso_forest.score_samples(X_scaled)
        lof_scores = self.lof.negative_outlier_factor_
        
        # Normalize to [0, 1]
        iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-8)
        lof_scores_norm = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min() + 1e-8)
        
        # Store anomaly scores
        for i, date in enumerate(dates):
            combined_score = (iso_scores_norm[i] + lof_scores_norm[i]) / 2.0
            self.anomaly_scores[date] = float(combined_score)
    
    def _features_to_matrix(
        self, features_list: List[Dict[str, float]]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Convert list of feature dicts to a feature matrix.
        
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        if not features_list:
            return np.array([]), []
        
        # Determine all feature names
        all_keys = set()
        for features in features_list:
            all_keys.update(features.keys())
        
        feature_names = sorted(list(all_keys))
        
        # Build matrix
        X = np.zeros((len(features_list), len(feature_names)))
        
        for i, features in enumerate(features_list):
            for j, key in enumerate(feature_names):
                X[i, j] = features.get(key, 0.0)
        
        return X, feature_names
    
    def predict(self, G: nx.DiGraph, person_id: str, date: str) -> Dict[str, any]:
        """
        Detect if a specific day is anomalous.
        
        Args:
            G: NetworkX graph
            person_id: ID of the person
            date: date in ISO format (YYYY-MM-DD)
        
        Returns:
            Dictionary with:
            - anomaly_score: float in [0, 1], where 1 = most anomalous
            - is_anomaly: bool (True if score > threshold)
            - reasons: List of specific anomalies detected
        """
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        
        from .graph_embeddings import extract_day_features
        
        day_features = extract_day_features(G, person_id, date)
        
        if not day_features:
            return {
                "anomaly_score": 0.0,
                "is_anomaly": False,
                "reasons": ["No data for this day"]
            }
        
        # Convert to vector
        X = np.zeros((1, len(self.feature_names)))
        for j, key in enumerate(self.feature_names):
            X[0, j] = day_features.get(key, 0.0)
        
        # Get anomaly score
        X_scaled = self.scaler.transform(X)
        iso_score = self.iso_forest.score_samples(X_scaled)[0]
        lof_score = self.lof._decision_function(X_scaled)[0]
        
        # Normalize
        iso_norm = (iso_score - self.iso_forest.offset_) / max(abs(self.iso_forest.offset_), 1.0)
        iso_norm = max(0, min(1, iso_norm))
        
        lof_norm = -lof_score
        lof_norm = max(0, min(1, lof_norm))
        
        combined_score = (iso_norm + lof_norm) / 2.0
        threshold = self.contamination * 1.5  # slightly higher than training contamination
        is_anomaly = combined_score > threshold
        
        # Identify specific reasons
        reasons = self._identify_anomaly_reasons(day_features)
        
        return {
            "anomaly_score": float(combined_score),
            "is_anomaly": bool(is_anomaly),
            "reasons": reasons,
            "feature_values": {k: float(v) for k, v in day_features.items()}
        }
    
    def _identify_anomaly_reasons(self, day_features: Dict[str, float]) -> List[str]:
        """
        Identify specific reasons why a day might be anomalous.
        """
        reasons = []
        
        # Low sleep
        if day_features.get("sleep_duration_h", 8) < 6:
            reasons.append("Very low sleep duration")
        
        # Poor sleep quality
        if day_features.get("sleep_quality", 80) < 60:
            reasons.append("Poor sleep quality")
        
        # Extreme calorie intake
        calories = day_features.get("total_calories", 0)
        if calories < 1000:
            reasons.append("Very low calorie intake")
        elif calories > 4000:
            reasons.append("Very high calorie intake")
        
        # Extreme sugar intake
        sugar = day_features.get("total_sugar_g", 0)
        if sugar > 100:
            reasons.append("High sugar consumption")
        
        # Low fiber
        fiber = day_features.get("total_fiber_g", 0)
        if fiber < 5:
            reasons.append("Low fiber intake")
        
        # High activity without food
        activity = day_features.get("activity_count", 0)
        calories_burned = day_features.get("total_calories_burned", 0)
        if activity > 0 and calories < 1500:
            reasons.append("High activity with low calorie intake")
        
        return reasons


class PatternCorrelationAnalyzer:
    """
    Analyzes correlations between health behaviors and outcomes.
    
    Example patterns:
    - High protein + high sleep quality -> high focus score
    - Low sleep quality -> low energy next day
    - Weekend high activity -> better sleep
    """
    
    def __init__(self, lag_days: int = 1):
        self.lag_days = lag_days
        self.correlations: Dict[Tuple[str, str], float] = {}
    
    def analyze(self, G: nx.DiGraph, person_id: str) -> Dict[str, float]:
        """
        Analyze correlations between features across days.
        
        Returns:
            Dictionary mapping "feature1->feature2" -> correlation
        """
        from .graph_embeddings import extract_day_features
        
        # Get all days
        day_nodes = sorted([
            node for node in G.nodes()
            if G.nodes[node].get("type") == "day" and 
            node.startswith(f"day:{person_id}:")
        ])
        
        if len(day_nodes) < self.lag_days + 2:
            return {}
        
        # Extract features for all days
        all_features = []
        for day_id in day_nodes:
            date = day_id.split(":")[-1]
            features = extract_day_features(G, person_id, date)
            all_features.append(features)
        
        # Compute correlations
        correlations = {}
        
        # Feature pairs to analyze
        feature_pairs = [
            ("sleep_quality", "metric_focus_score"),
            ("total_protein_g", "metric_energy_level"),
            ("total_sugar_g", "sleep_quality"),
            ("activity_count", "sleep_duration_h"),
            ("total_calories", "metric_energy_level"),
        ]
        
        for feat1, feat2 in feature_pairs:
            values1 = []
            values2_lagged = []
            
            for i in range(len(all_features) - self.lag_days):
                if feat1 in all_features[i] and feat2 in all_features[i + self.lag_days]:
                    v1 = all_features[i][feat1]
                    v2 = all_features[i + self.lag_days][feat2]
                    
                    # Only include valid values
                    if not np.isnan(v1) and not np.isnan(v2) and v1 != 0 and v2 != 0:
                        values1.append(v1)
                        values2_lagged.append(v2)
            
            if len(values1) > 2:
                corr = float(np.corrcoef(values1, values2_lagged)[0, 1])
                if not np.isnan(corr):
                    correlations[f"{feat1}->{feat2}(+{self.lag_days}d)"] = corr
        
        self.correlations = correlations
        return correlations
    
    def get_insights(self) -> List[str]:
        """
        Generate interpretable insights from correlations.
        """
        insights = []
        
        for key, corr in sorted(self.correlations.items(), 
                               key=lambda x: abs(x[1]), reverse=True):
            if abs(corr) > 0.5:  # strong correlation
                direction = "increases" if corr > 0 else "decreases"
                feat1, feat2 = key.split("->")
                feat2 = feat2.split("(")[0]
                
                insights.append(
                    f"High {feat1} {direction} {feat2}"
                )
        
        return insights
