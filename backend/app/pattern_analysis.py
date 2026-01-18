"""
Pattern Analysis and Prediction module for the Household Health Graph Engine.

Predicts future health metrics based on historical patterns.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import json


class HealthMetricPredictor:
    """
    Predicts future health metrics (energy, focus, mood) based on daily features.
    
    Uses multiple regression to learn relationships between:
    - Nutrition inputs (calories, protein, fiber, sugar)
    - Sleep inputs (duration, quality)
    - Activity inputs (count, intensity, calories_burned)
    - Outcomes (energy_level, focus_score, mood)
    """
    
    def __init__(self, prediction_window: int = 1):
        """
        Initialize predictor.
        
        Args:
            prediction_window: number of days ahead to predict
        """
        self.prediction_window = prediction_window
        
        # Separate models for each metric
        self.models: Dict[str, LinearRegression] = {}
        self.scaler = StandardScaler()
        
        self.is_fitted = False
        self.feature_names: List[str] = []
    
    def fit(self, G: nx.DiGraph, person_id: str) -> None:
        """
        Fit prediction models on historical data.
        
        Args:
            G: NetworkX graph
            person_id: ID of the person to analyze
        """
        from .graph_embeddings import extract_day_features
        
        # Get all days
        day_nodes = sorted([
            node for node in G.nodes()
            if G.nodes[node].get("type") == "day" and 
            node.startswith(f"day:{person_id}:")
        ])
        
        if len(day_nodes) < self.prediction_window + 5:
            raise ValueError(
                f"Need at least {self.prediction_window + 5} days of data"
            )
        
        # Extract all features
        all_features = []
        for day_id in day_nodes:
            date = day_id.split(":")[-1]
            features = extract_day_features(G, person_id, date)
            all_features.append(features)
        
        # Select input features (exclude metrics)
        input_features = [
            "total_calories", "total_protein_g", "total_fiber_g", "total_sugar_g",
            "meal_count", "sleep_duration_h", "sleep_quality",
            "activity_count", "total_calories_burned", "max_intensity"
        ]
        
        # Target features
        target_features = [
            "metric_energy_level", "metric_focus_score", "metric_mood"
        ]
        
        # Build training matrix
        X_list = []
        y_dict = {target: [] for target in target_features}
        
        for i in range(len(all_features) - self.prediction_window):
            # Current day features as input
            x_row = []
            for feat in input_features:
                x_row.append(all_features[i].get(feat, 0.0))
            
            # Check if we have valid data
            if all(v is not None for v in x_row):
                X_list.append(x_row)
                
                # Future day metrics as targets
                for target in target_features:
                    y_dict[target].append(
                        all_features[i + self.prediction_window].get(target, 0.0)
                    )
        
        if len(X_list) < 3:
            raise ValueError("Not enough valid training data")
        
        X = np.array(X_list)
        self.feature_names = input_features
        
        # Fit scaler and scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit separate model for each target
        for target in target_features:
            y = np.array(y_dict[target])
            
            # Only fit if we have non-zero variance
            if np.std(y) > 0:
                model = LinearRegression()
                model.fit(X_scaled, y)
                self.models[target] = model
        
        self.is_fitted = True
    
    def predict(self, G: nx.DiGraph, person_id: str, date: str) -> Dict[str, any]:
        """
        Predict future health metrics for a given date.
        
        Args:
            G: NetworkX graph
            person_id: ID of the person
            date: reference date (we predict prediction_window days ahead)
        
        Returns:
            Dictionary with predicted metrics and confidence
        """
        if not self.is_fitted:
            raise RuntimeError("Predictor not fitted. Call fit() first.")
        
        from .graph_embeddings import extract_day_features
        
        day_features = extract_day_features(G, person_id, date)
        
        # Build input vector
        X = np.zeros((1, len(self.feature_names)))
        for i, feat in enumerate(self.feature_names):
            X[0, i] = day_features.get(feat, 0.0)
        
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = {}
        for target, model in self.models.items():
            pred = model.predict(X_scaled)[0]
            # Clamp to realistic ranges
            if "energy" in target or "focus" in target or "mood" in target:
                pred = max(0, min(100, pred))
            predictions[target] = float(pred)
        
        # Compute feature importance
        feature_importance = {}
        for target, model in self.models.items():
            # Get coefficients
            coefs = np.abs(model.coef_)
            top_indices = np.argsort(coefs)[-3:][::-1]  # top 3 features
            
            for idx in top_indices:
                feat_name = self.feature_names[idx]
                importance = float(coefs[idx])
                if feat_name not in feature_importance:
                    feature_importance[feat_name] = 0
                feature_importance[feat_name] += importance
        
        return {
            "date": date,
            "predictions": predictions,
            "prediction_window_days": self.prediction_window,
            "feature_importance": feature_importance,
            "input_features": {k: float(v) for k, v in day_features.items()}
        }


class AssociationRuleAnalyzer:
    """
    Discovers association rules like:
    - high_protein AND high_sleep -> high_focus
    - high_sugar AND low_activity -> low_energy
    
    Uses simple frequency analysis.
    """
    
    def __init__(self, min_support: float = 0.2):
        """
        Args:
            min_support: minimum proportion of days for rule to be valid
        """
        self.min_support = min_support
        self.rules: List[Dict] = []
    
    def _discretize_feature(self, values: List[float], name: str) -> List[str]:
        """Convert continuous values to categorical (low, medium, high)."""
        arr = np.array(values)
        
        if len(arr) == 0 or np.std(arr) == 0:
            return ["medium"] * len(arr)
        
        q33 = np.percentile(arr, 33)
        q67 = np.percentile(arr, 67)
        
        categories = []
        for v in arr:
            if v < q33:
                categories.append(f"{name}_low")
            elif v > q67:
                categories.append(f"{name}_high")
            else:
                categories.append(f"{name}_medium")
        
        return categories
    
    def discover_rules(self, G: nx.DiGraph, person_id: str,
                      min_confidence: float = 0.6) -> List[Dict[str, any]]:
        """
        Discover association rules from historical data.
        
        Args:
            G: NetworkX graph
            person_id: ID of the person
            min_confidence: minimum confidence for rules (0.6 = 60%)
        
        Returns:
            List of rules with support and confidence
        """
        from .graph_embeddings import extract_day_features
        
        # Get all days
        day_nodes = sorted([
            node for node in G.nodes()
            if G.nodes[node].get("type") == "day" and 
            node.startswith(f"day:{person_id}:")
        ])
        
        if len(day_nodes) < 10:
            return []
        
        # Extract features for all days
        all_features = []
        dates = []
        for day_id in day_nodes:
            date = day_id.split(":")[-1]
            features = extract_day_features(G, person_id, date)
            all_features.append(features)
            dates.append(date)
        
        # Discretize key features
        feature_series = {
            "sleep_quality": [f.get("sleep_quality", 80) for f in all_features],
            "total_protein": [f.get("total_protein_g", 20) for f in all_features],
            "total_sugar": [f.get("total_sugar_g", 20) for f in all_features],
            "activity_intensity": [f.get("max_intensity", 0) for f in all_features],
            "energy": [f.get("metric_energy_level", 50) for f in all_features],
            "focus": [f.get("metric_focus_score", 50) for f in all_features],
        }
        
        categorical_data = {}
        for key, values in feature_series.items():
            categorical_data[key] = self._discretize_feature(values, key)
        
        # Find rules
        rules = []
        
        # Rule patterns: antecedent -> consequent
        antecedent_patterns = [
            ["sleep_quality_high"],
            ["total_protein_high"],
            ["activity_intensity_high"],
            ["sleep_quality_high", "total_protein_high"],
            ["total_sugar_low", "activity_intensity_high"],
        ]
        
        consequent_patterns = [
            ["energy_high"],
            ["focus_high"],
            ["energy_high", "focus_high"],
        ]
        
        for antecedent in antecedent_patterns:
            for consequent in consequent_patterns:
                # Count occurrences
                support_count = 0
                confidence_count = 0
                
                for i in range(len(dates)):
                    # Check if all antecedent items are present
                    antecedent_match = all(
                        cat in categorical_data[key.split("_")[0]][i]
                        for key in antecedent
                        for cat in [key]
                    )
                    
                    if antecedent_match:
                        confidence_count += 1
                        
                        # Check if all consequent items are present
                        consequent_match = all(
                            cat in categorical_data[key.split("_")[0]][i]
                            for key in consequent
                            for cat in [key]
                        )
                        
                        if consequent_match:
                            support_count += 1
                
                support = support_count / len(dates)
                confidence = support_count / max(1, confidence_count)
                
                if support >= self.min_support and confidence >= min_confidence:
                    rule = {
                        "if": " AND ".join(antecedent),
                        "then": " AND ".join(consequent),
                        "support": float(support),
                        "confidence": float(confidence),
                        "count": support_count
                    }
                    rules.append(rule)
        
        # Sort by confidence
        rules.sort(key=lambda x: x["confidence"], reverse=True)
        self.rules = rules
        
        return rules
