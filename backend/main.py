# backend/main.py

from fastapi import FastAPI
from fastapi.responses import RedirectResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from networkx.readwrite import json_graph
import joblib
from pathlib import Path

from .app.graph_builder import build_example_graph
from .app.synthetic_data import build_single_child_household
from .app.graph_viz import graph_to_svg
from .app.graph_embeddings import SimpleGraphSAGE, extract_day_features
from .app.anomaly_detection import HealthAnomalyDetector, PatternCorrelationAnalyzer
from .app.pattern_analysis import HealthMetricPredictor, AssociationRuleAnalyzer
from .app.link_prediction import HealthLinkPredictor
from .app.node_classifier import FoodNutritionClassifier, ActivityIntensityClassifier, HealthStatusClassifier


app = FastAPI(
    title="Household Health Graph API with ML",
    version="0.2.0",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model storage directory
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Root / Health
# ---------------------------------------------------------

@app.get("/")
def root():
    return RedirectResponse(url="/docs", status_code=307)


@app.get("/health")
def health():
    return {"ok": True}


# ---------------------------------------------------------
# JSON Debug Endpoints
# ---------------------------------------------------------

@app.get("/debug/example-graph")
def debug_example_graph():
    G = build_example_graph()
    return json_graph.node_link_data(G)


@app.get("/debug/synthetic-graph")
def debug_synthetic_graph(days: int = 60):
    G = build_single_child_household(num_days=days)
    return json_graph.node_link_data(G)


# ---------------------------------------------------------
# SVG Debug Endpoints
# ---------------------------------------------------------

@app.get("/debug/example-graph/svg")
def debug_example_graph_svg():
    G = build_example_graph()
    svg_bytes = graph_to_svg(G)

    return Response(
        content=svg_bytes,
        media_type="image/svg+xml",
        headers={
            "Content-Disposition": "inline; filename=example-graph.svg",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/debug/synthetic-graph/svg")
def debug_synthetic_graph_svg(days: int = 60):
    G = build_single_child_household(num_days=days)
    svg_bytes = graph_to_svg(G)

    return Response(
        content=svg_bytes,
        media_type="image/svg+xml",
        headers={
            "Content-Disposition": "inline; filename=synthetic-graph.svg",
            "X-Content-Type-Options": "nosniff",
        },
    )


# ---------------------------------------------------------
# Optional HTML Wrapper
# ---------------------------------------------------------

@app.get("/debug/synthetic-graph/html")
def debug_synthetic_graph_html(days: int = 60):
    G = build_single_child_household(num_days=days)
    svg_bytes = graph_to_svg(G)
    svg = svg_bytes.decode("utf-8")

    html = f"""
    <html>
      <head><title>Synthetic Graph Debug</title></head>
      <body>
        <h2>Synthetic Graph (days={days})</h2>
        {svg}
      </body>
    </html>
    """

    return Response(content=html, media_type="text/html")


# ---------------------------------------------------------
# ML Endpoints: Graph Embeddings
# ---------------------------------------------------------

@app.post("/ml/embeddings/train")
def train_embeddings(days: int = 60):
    """Train graph embeddings using GraphSAGE."""
    G = build_single_child_household(num_days=days)
    
    sage = SimpleGraphSAGE(embedding_dim=16, num_aggregation_layers=2)
    embeddings = sage.fit_embeddings(G)
    
    # Save model
    joblib.dump(sage, MODEL_DIR / "graph_sage.pkl")
    
    return {
        "status": "success",
        "num_nodes": len(G.nodes()),
        "num_embeddings": len(embeddings),
        "embedding_dim": 16,
        "message": "GraphSAGE model trained and saved"
    }


@app.get("/ml/embeddings/daily/{person_id}/{date}")
def get_daily_embedding(person_id: str, date: str, days: int = 60):
    """Get daily embedding for a specific person and date."""
    G = build_single_child_household(num_days=days)
    
    # Load or train model
    model_path = MODEL_DIR / "graph_sage.pkl"
    if model_path.exists():
        sage = joblib.load(model_path)
    else:
        sage = SimpleGraphSAGE(embedding_dim=16, num_aggregation_layers=2)
        sage.fit_embeddings(G)
    
    embedding = sage.get_daily_embedding(G, person_id, date)
    
    if embedding is None:
        return {"error": f"No data for {person_id} on {date}"}
    
    return {
        "person_id": person_id,
        "date": date,
        "embedding": embedding.tolist(),
        "embedding_dim": len(embedding)
    }


# ---------------------------------------------------------
# ML Endpoints: Anomaly Detection
# ---------------------------------------------------------

@app.post("/ml/anomalies/train")
def train_anomaly_detector(days: int = 60):
    """Train anomaly detection models."""
    G = build_single_child_household(num_days=days)
    
    detector = HealthAnomalyDetector(contamination=0.1)
    detector.fit(G, "person:yooni")
    
    # Save model
    joblib.dump(detector, MODEL_DIR / "anomaly_detector.pkl")
    
    return {
        "status": "success",
        "contamination": 0.1,
        "anomalies_detected": sum(1 for s in detector.anomaly_scores.values() if s > 0.15),
        "message": "Anomaly detector trained and saved"
    }


@app.get("/ml/anomalies/detect/{person_id}/{date}")
def detect_anomaly(person_id: str, date: str, days: int = 60):
    """Detect if a specific day is anomalous."""
    G = build_single_child_household(num_days=days)
    
    # Load or train model
    model_path = MODEL_DIR / "anomaly_detector.pkl"
    if model_path.exists():
        detector = joblib.load(model_path)
    else:
        detector = HealthAnomalyDetector(contamination=0.1)
        detector.fit(G, person_id)
    
    result = detector.predict(G, person_id, date)
    return result


@app.get("/ml/anomalies/correlations/{person_id}")
def analyze_correlations(person_id: str, days: int = 60):
    """Analyze correlations between health behaviors and outcomes."""
    G = build_single_child_household(num_days=days)
    
    analyzer = PatternCorrelationAnalyzer(lag_days=1)
    correlations = analyzer.analyze(G, person_id)
    insights = analyzer.get_insights()
    
    return {
        "person_id": person_id,
        "correlations": correlations,
        "insights": insights,
        "lag_days": 1
    }


# ---------------------------------------------------------
# ML Endpoints: Pattern Analysis & Prediction
# ---------------------------------------------------------

@app.post("/ml/prediction/train")
def train_predictor(days: int = 60):
    """Train health metric predictors."""
    G = build_single_child_household(num_days=days)
    
    predictor = HealthMetricPredictor(prediction_window=1)
    predictor.fit(G, "person:yooni")
    
    # Save model
    joblib.dump(predictor, MODEL_DIR / "metric_predictor.pkl")
    
    return {
        "status": "success",
        "prediction_window": 1,
        "message": "Metric predictor trained and saved"
    }


@app.get("/ml/prediction/metrics/{person_id}/{date}")
def predict_metrics(person_id: str, date: str, days: int = 60):
    """Predict future health metrics."""
    G = build_single_child_household(num_days=days)
    
    # Load or train model
    model_path = MODEL_DIR / "metric_predictor.pkl"
    if model_path.exists():
        predictor = joblib.load(model_path)
    else:
        predictor = HealthMetricPredictor(prediction_window=1)
        predictor.fit(G, person_id)
    
    result = predictor.predict(G, person_id, date)
    return result


@app.get("/ml/patterns/association-rules/{person_id}")
def discover_rules(person_id: str, days: int = 60):
    """Discover association rules in health behaviors."""
    G = build_single_child_household(num_days=days)
    
    analyzer = AssociationRuleAnalyzer(min_support=0.2)
    rules = analyzer.discover_rules(G, person_id, min_confidence=0.6)
    
    return {
        "person_id": person_id,
        "rules": rules,
        "total_rules": len(rules)
    }


# ---------------------------------------------------------
# ML Endpoints: Link Prediction
# ---------------------------------------------------------

@app.post("/ml/link-prediction/train")
def train_link_predictor(days: int = 60):
    """Train link prediction model."""
    G = build_single_child_household(num_days=days)
    
    predictor = HealthLinkPredictor()
    predictor.fit(G)
    
    # Save model
    joblib.dump(predictor, MODEL_DIR / "link_predictor.pkl")
    
    return {
        "status": "success",
        "total_edges": len(G.edges()),
        "message": "Link predictor trained and saved"
    }


@app.get("/ml/link-prediction/suggest")
def suggest_links(days: int = 60, top_k: int = 10):
    """Suggest new links to add to the graph."""
    G = build_single_child_household(num_days=days)
    
    # Load or train model
    model_path = MODEL_DIR / "link_predictor.pkl"
    if model_path.exists():
        predictor = joblib.load(model_path)
    else:
        predictor = HealthLinkPredictor()
        predictor.fit(G)
    
    predictions = predictor.predict_links(G, top_k=top_k)
    
    return {
        "suggested_links": predictions,
        "total_suggestions": len(predictions)
    }


# ---------------------------------------------------------
# ML Endpoints: Node Classification
# ---------------------------------------------------------

@app.post("/ml/classification/train")
def train_classifiers(days: int = 60):
    """Train node classifiers."""
    G = build_single_child_household(num_days=days)
    
    food_clf = FoodNutritionClassifier()
    food_clf.fit(G)
    
    activity_clf = ActivityIntensityClassifier()
    activity_clf.fit(G)
    
    # Save models
    joblib.dump(food_clf, MODEL_DIR / "food_classifier.pkl")
    joblib.dump(activity_clf, MODEL_DIR / "activity_classifier.pkl")
    
    return {
        "status": "success",
        "models_trained": ["food_nutrition", "activity_intensity"],
        "message": "Classifiers trained and saved"
    }


@app.get("/ml/classification/food/{food_id}")
def classify_food(food_id: str, days: int = 60):
    """Classify a food's nutritional profile."""
    G = build_single_child_household(num_days=days)
    
    # Load or train model
    model_path = MODEL_DIR / "food_classifier.pkl"
    if model_path.exists():
        clf = joblib.load(model_path)
    else:
        clf = FoodNutritionClassifier()
        clf.fit(G)
    
    result = clf.predict(G, food_id)
    return result


@app.get("/ml/classification/activity/{activity_id}")
def classify_activity(activity_id: str, days: int = 60):
    """Classify an activity's intensity."""
    G = build_single_child_household(num_days=days)
    
    # Load or train model
    model_path = MODEL_DIR / "activity_classifier.pkl"
    if model_path.exists():
        clf = joblib.load(model_path)
    else:
        clf = ActivityIntensityClassifier()
        clf.fit(G)
    
    result = clf.predict(G, activity_id)
    return result


@app.get("/ml/classification/daily-status/{person_id}/{date}")
def classify_daily_status(person_id: str, date: str, days: int = 60):
    """Classify overall health status for a day."""
    G = build_single_child_household(num_days=days)
    
    day_features = extract_day_features(G, person_id, date)
    
    if not day_features:
        return {"error": f"No data for {person_id} on {date}"}
    
    result = HealthStatusClassifier.classify_day(day_features)
    result["date"] = date
    result["person_id"] = person_id
    
    return result


# ---------------------------------------------------------
# ML Endpoints: Summary & Dashboard
# ---------------------------------------------------------

@app.get("/ml/insights/daily/{person_id}/{date}")
def daily_health_insights(person_id: str, date: str, days: int = 60):
    """Get comprehensive daily health insights combining all ML models."""
    G = build_single_child_household(num_days=days)
    
    # Daily features
    day_features = extract_day_features(G, person_id, date)
    
    # Status
    status = HealthStatusClassifier.classify_day(day_features)
    
    # Load anomaly detector
    model_path = MODEL_DIR / "anomaly_detector.pkl"
    if model_path.exists():
        detector = joblib.load(model_path)
        anomaly = detector.predict(G, person_id, date)
    else:
        anomaly = {"anomaly_score": 0.0, "is_anomaly": False, "reasons": []}
    
    # Load predictor
    model_path = MODEL_DIR / "metric_predictor.pkl"
    if model_path.exists():
        predictor = joblib.load(model_path)
        predictions = predictor.predict(G, person_id, date)
    else:
        predictions = {}
    
    return {
        "date": date,
        "person_id": person_id,
        "daily_features": day_features,
        "health_status": status,
        "anomaly_detection": {
            "is_anomalous": anomaly["is_anomaly"],
            "anomaly_score": anomaly["anomaly_score"],
            "reasons": anomaly["reasons"]
        },
        "next_day_predictions": predictions.get("predictions", {}),
        "important_factors": predictions.get("feature_importance", {})
    }


@app.get("/ml/summary/{person_id}")
def person_summary(person_id: str, days: int = 60):
    """Get comprehensive summary for a person."""
    G = build_single_child_household(num_days=days)
    
    # Correlations
    analyzer = PatternCorrelationAnalyzer(lag_days=1)
    correlations = analyzer.analyze(G, person_id)
    insights = analyzer.get_insights()
    
    # Association rules
    rule_analyzer = AssociationRuleAnalyzer(min_support=0.2)
    rules = rule_analyzer.discover_rules(G, person_id, min_confidence=0.6)
    
    return {
        "person_id": person_id,
        "days_analyzed": days,
        "behavior_insights": insights[:5],  # top 5
        "health_rules": rules[:5],  # top 5
        "data_points": {
            "total_nodes": len(G.nodes()),
            "meals_recorded": len([n for n in G.nodes() if G.nodes[n].get("type") == "meal"]),
            "activities_recorded": len([n for n in G.nodes() if G.nodes[n].get("type") == "activity"]),
            "metrics_recorded": len([n for n in G.nodes() if G.nodes[n].get("type") == "metric"]),
        }
    }

