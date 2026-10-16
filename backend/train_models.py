"""
ML Training pipeline for the Household Health Graph Engine.

This script trains all ML models and saves them for later use.
Run this to initialize the system with pre-trained models.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.synthetic_data import build_demo_family_household, DEMO_MEMBERS
from backend.app.graph_embeddings import SimpleGraphSAGE
from backend.app.anomaly_detection import HealthAnomalyDetector
from backend.app.pattern_analysis import HealthMetricPredictor, AssociationRuleAnalyzer
from backend.app.link_prediction import HealthLinkPredictor
from backend.app.node_classifier import FoodNutritionClassifier, ActivityIntensityClassifier
import joblib


def train_all_models(num_days: int = 90):
    """Train all ML models."""
    print(f"Loading synthetic data ({num_days} days)...")
    G = build_demo_family_household(num_days=num_days)
    print(f"✓ Loaded graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    print(f"✓ Models directory: {models_dir}")
    
    people = [f"person:{m['key']}" for m in DEMO_MEMBERS]
    
    # 1. Train GraphSAGE
    print("\n[1/7] Training GraphSAGE embeddings...")
    sage = SimpleGraphSAGE(embedding_dim=16, num_aggregation_layers=2)
    sage.fit_embeddings(G)
    joblib.dump(sage, models_dir / "graph_sage.pkl")
    print("✓ GraphSAGE model saved")
    
    # 2-3. Per-person anomaly detector and next-day metric predictor
    for person_id in people:
        key = person_id.split(":")[-1]
        print(f"\n[2/7] Training anomaly detector for {key}...")
        detector = HealthAnomalyDetector(contamination=0.1)
        detector.fit(G, person_id)
        joblib.dump(detector, models_dir / f"anomaly_detector_{key}.pkl")
        anomaly_count = sum(1 for s in detector.anomaly_scores.values() if s > 0.15)
        print(f"  saved ({anomaly_count} anomalous days flagged)")

        print(f"[3/7] Training metric predictor for {key}...")
        predictor = HealthMetricPredictor(prediction_window=1)
        predictor.fit(G, person_id)
        joblib.dump(predictor, models_dir / f"metric_predictor_{key}.pkl")
        print(f"  saved (targets: {list(predictor.models.keys())})")

    # 4. Train Link Predictor
    print("\n[4/7] Training link predictor...")
    link_predictor = HealthLinkPredictor()
    link_predictor.fit(G)
    joblib.dump(link_predictor, models_dir / "link_predictor.pkl")
    print("✓ Link predictor trained")
    
    # 5. Train Food Nutrition Classifier
    print("\n[5/7] Training food nutrition classifier...")
    food_clf = FoodNutritionClassifier()
    food_clf.fit(G)
    joblib.dump(food_clf, models_dir / "food_classifier.pkl")
    print("✓ Food classifier trained")
    
    # 6. Train Activity Intensity Classifier
    print("\n[6/7] Training activity intensity classifier...")
    activity_clf = ActivityIntensityClassifier()
    activity_clf.fit(G)
    joblib.dump(activity_clf, models_dir / "activity_classifier.pkl")
    print("✓ Activity classifier trained")
    
    # 7. Analyze patterns (no model to save, but good for verification)
    print("\n[7/7] Analyzing patterns...")
    rule_analyzer = AssociationRuleAnalyzer(min_support=0.2)
    rules = rule_analyzer.discover_rules(G, people[0], min_confidence=0.6)
    print(f"✓ Found {len(rules)} association rules")
    
    # Summary
    print("\n" + "="*60)
    print("ML PIPELINE TRAINING COMPLETE")
    print("="*60)
    print(f"\nModels saved to: {models_dir}")
    print("\nTrained models:")
    print("  ✓ graph_sage.pkl - GraphSAGE embeddings")
    print("  ✓ anomaly_detector_<person>.pkl - Health anomaly detection")
    print("  ✓ metric_predictor_<person>.pkl - Daily metric prediction")
    print("  ✓ link_predictor.pkl - Graph link prediction")
    print("  ✓ food_classifier.pkl - Food nutrition classification")
    print("  ✓ activity_classifier.pkl - Activity intensity classification")
    print("\nYou can now use the API endpoints!")
    print("\nStartup: uvicorn backend.main:app --reload")


if __name__ == "__main__":
    train_all_models(num_days=90)
