"""
Node Classification module for the Household Health Graph Engine.

Classifies nodes and predicts their attributes.
Particularly useful for:
- Food categorization (nutrition level: low/medium/high)
- Activity intensity classification
- Health status classification
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json


class FoodNutritionClassifier:
    """
    Classifies foods by their nutritional profile:
    - "healthy": high protein, high fiber, low sugar
    - "balanced": moderate across all nutrients
    - "indulgent": higher sugar/fat, lower nutritional density
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def _extract_food_features(self, node_data: Dict) -> np.ndarray:
        """Extract nutritional features from a food node."""
        features = [
            node_data.get("protein_g", 0),
            node_data.get("fiber_g", 0),
            node_data.get("sugar_g", 0),
            node_data.get("fat_g", 0),
            node_data.get("kcal", 100),  # calories
        ]
        return np.array(features)
    
    def _label_food(self, node_data: Dict) -> str:
        """
        Create a label for training.
        Simple heuristic-based labeling.
        """
        protein = node_data.get("protein_g", 0)
        fiber = node_data.get("fiber_g", 0)
        sugar = node_data.get("sugar_g", 0)
        
        # Scoring logic
        protein_score = min(protein / 20.0, 1.0)  # higher is better
        fiber_score = min(fiber / 5.0, 1.0)  # higher is better
        sugar_score = max(1.0 - sugar / 30.0, 0.0)  # lower is better
        
        health_score = (protein_score + fiber_score + sugar_score) / 3.0
        
        if health_score > 0.7:
            return "healthy"
        elif health_score > 0.4:
            return "balanced"
        else:
            return "indulgent"
    
    def fit(self, G: nx.DiGraph) -> None:
        """
        Fit classifier on food nodes in the graph.
        """
        X_list = []
        y_list = []
        
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            if node_data.get("type") == "food":
                features = self._extract_food_features(node_data)
                label = self._label_food(node_data)
                
                X_list.append(features)
                y_list.append(label)
        
        if len(X_list) < 3:
            raise ValueError("Need at least 3 food nodes to fit classifier")
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Fit scaler and model
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        self.is_fitted = True
    
    def predict(self, G: nx.DiGraph, food_id: str) -> Dict[str, any]:
        """
        Predict nutrition classification for a food.
        
        Returns:
            Dictionary with classification, confidence, and reasoning
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        
        if food_id not in G.nodes:
            return None
        
        node_data = G.nodes[food_id]
        features = self._extract_food_features(node_data)
        X_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        confidence = float(np.max(probabilities))
        
        # Get feature importance
        feature_names = ["protein (g)", "fiber (g)", "sugar (g)", "fat (g)", "calories"]
        importances = self.model.feature_importances_
        top_features = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )[:2]
        
        return {
            "food_id": food_id,
            "name": node_data.get("name", food_id),
            "classification": prediction,
            "confidence": confidence,
            "top_contributing_factors": [f[0] for f in top_features],
            "nutritional_profile": {
                "protein_g": float(node_data.get("protein_g", 0)),
                "fiber_g": float(node_data.get("fiber_g", 0)),
                "sugar_g": float(node_data.get("sugar_g", 0)),
                "fat_g": float(node_data.get("fat_g", 0)),
                "kcal": float(node_data.get("kcal", 0)),
            }
        }
    
    def classify_all_foods(self, G: nx.DiGraph) -> Dict[str, Dict]:
        """Classify all food nodes in the graph."""
        results = {}
        for node_id in G.nodes():
            if G.nodes[node_id].get("type") == "food":
                results[node_id] = self.predict(G, node_id)
        return results


class ActivityIntensityClassifier:
    """
    Classifies activities by intensity and expected health benefits.
    Categories:
    - "light": low intensity, good for recovery
    - "moderate": balanced intensity and benefit
    - "vigorous": high intensity, maximal benefit
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=30, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def _extract_activity_features(self, node_data: Dict) -> np.ndarray:
        """Extract features from an activity node."""
        intensity_map = {"light": 1, "moderate": 2, "medium": 2, "vigorous": 3, "high": 3}
        
        features = [
            node_data.get("duration_min", 30),
            intensity_map.get(node_data.get("intensity", "moderate"), 2),
            node_data.get("calories_burned", 0),
        ]
        return np.array(features)
    
    def _label_activity(self, node_data: Dict) -> str:
        """Create a label for training."""
        intensity = node_data.get("intensity", "moderate").lower()
        duration = node_data.get("duration_min", 30)
        calories = node_data.get("calories_burned", 100)
        
        if intensity in ("light", "low"):
            return "light"
        elif intensity in ("vigorous", "high") or (duration > 45 and calories > 250):
            return "vigorous"
        else:
            return "moderate"
    
    def fit(self, G: nx.DiGraph) -> None:
        """Fit classifier on activity nodes."""
        X_list = []
        y_list = []
        
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            if node_data.get("type") == "activity":
                features = self._extract_activity_features(node_data)
                label = self._label_activity(node_data)
                
                X_list.append(features)
                y_list.append(label)
        
        if len(X_list) < 2:
            return  # Not enough data
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        self.is_fitted = True
    
    def predict(self, G: nx.DiGraph, activity_id: str) -> Dict[str, any]:
        """Predict intensity classification for an activity."""
        if not self.is_fitted:
            return {
                "activity_id": activity_id,
                "name": G.nodes[activity_id].get("name", activity_id),
                "classification": "moderate",
                "confidence": 0.5,
                "note": "Classifier not fitted"
            }
        
        if activity_id not in G.nodes:
            return None
        
        node_data = G.nodes[activity_id]
        features = self._extract_activity_features(node_data)
        X_scaled = self.scaler.transform(features.reshape(1, -1))
        
        prediction = self.model.predict(X_scaled)[0]
        confidence = float(np.max(self.model.predict_proba(X_scaled)[0]))
        
        return {
            "activity_id": activity_id,
            "name": node_data.get("name", activity_id),
            "classification": prediction,
            "confidence": confidence,
            "duration_min": int(node_data.get("duration_min", 0)),
            "calories_burned": float(node_data.get("calories_burned", 0)),
        }


class HealthStatusClassifier:
    """
    Classifies daily health status based on metrics and behaviors.
    Categories:
    - "optimal": excellent sleep, nutrition, activity, metrics
    - "good": mostly positive indicators
    - "fair": mixed indicators
    - "concerning": one or more poor indicators
    """
    
    @staticmethod
    def classify_day(day_features: Dict[str, float]) -> Dict[str, any]:
        """
        Classify health status for a day.
        
        Args:
            day_features: features extracted from a day node
        
        Returns:
            Dictionary with classification and details
        """
        scores = {}
        
        # Sleep score
        sleep_duration = day_features.get("sleep_duration_h", 8)
        sleep_quality = day_features.get("sleep_quality", 80)
        
        if sleep_duration >= 7.5 and sleep_quality >= 80:
            scores["sleep"] = 3  # good
        elif sleep_duration >= 6.5 and sleep_quality >= 60:
            scores["sleep"] = 2  # fair
        else:
            scores["sleep"] = 1  # poor
        
        # Nutrition score
        calories = day_features.get("total_calories", 0)
        protein = day_features.get("total_protein_g", 0)
        fiber = day_features.get("total_fiber_g", 0)
        
        nutrition_score = 0
        if 1500 <= calories <= 3000:
            nutrition_score += 1
        if protein > 40:
            nutrition_score += 1
        if fiber > 15:
            nutrition_score += 1
        
        scores["nutrition"] = min(nutrition_score, 3)
        
        # Activity score
        activity_count = day_features.get("activity_count", 0)
        max_intensity = day_features.get("max_intensity", 0)
        
        if activity_count > 0 and max_intensity >= 2:
            scores["activity"] = 3
        elif activity_count > 0:
            scores["activity"] = 2
        else:
            scores["activity"] = 1
        
        # Metric scores
        energy = day_features.get("metric_energy_level", 50)
        focus = day_features.get("metric_focus_score", 50)
        mood = day_features.get("metric_mood", 50)
        
        avg_metric = (energy + focus + mood) / 3.0
        if avg_metric >= 75:
            scores["outcomes"] = 3
        elif avg_metric >= 50:
            scores["outcomes"] = 2
        else:
            scores["outcomes"] = 1
        
        # Overall classification
        total_score = sum(scores.values())
        
        if total_score >= 12:
            classification = "optimal"
        elif total_score >= 9:
            classification = "good"
        elif total_score >= 6:
            classification = "fair"
        else:
            classification = "concerning"
        
        # Identify weak areas
        weak_areas = [k for k, v in scores.items() if v == 1]
        
        return {
            "classification": classification,
            "overall_score": total_score,
            "max_score": 15,
            "breakdown": {k: v for k, v in scores.items()},
            "weak_areas": weak_areas,
            "recommendation": HealthStatusClassifier._get_recommendation(
                classification, weak_areas
            )
        }
    
    @staticmethod
    def _get_recommendation(classification: str, weak_areas: List[str]) -> str:
        """Generate a recommendation based on classification."""
        if classification == "optimal":
            return "Keep up the excellent habits!"
        elif classification == "good":
            return "You're doing well. Minor improvements possible."
        elif classification == "fair":
            return f"Focus on improving: {', '.join(weak_areas)}"
        else:  # concerning
            return f"Prioritize: {', '.join(weak_areas)}. Consider lifestyle adjustments."
