"""
Household Health Graph API with Machine Learning.

A FastAPI application that combines graph-based health modeling with ML inference,
providing insights into household health patterns through multiple analytical lenses:
- Graph embeddings (GraphSAGE-based node representations)
- Anomaly detection (Isolation Forest + LOF ensemble)
- Pattern analysis (association rules, metric prediction)
- Link prediction (hidden food-health relationships)
- Node classification (nutrition, activity, health status)

Author: deokhwajeong
Version: 0.2.0
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from networkx.readwrite import json_graph
import joblib
from pathlib import Path
from datetime import datetime

from .app.graph_builder import build_example_graph
from .app.synthetic_data import build_single_child_household
from .app.graph_viz import graph_to_svg
from .app.graph_embeddings import SimpleGraphSAGE, extract_day_features
from .app.anomaly_detection import HealthAnomalyDetector, PatternCorrelationAnalyzer
from .app.pattern_analysis import HealthMetricPredictor, AssociationRuleAnalyzer
from .app.link_prediction import HealthLinkPredictor
from .app.node_classifier import FoodNutritionClassifier, ActivityIntensityClassifier, HealthStatusClassifier
from .app.ethics import (
    EthicsEngine, DataPurpose, ConsentStatus,
    PrivacyGuard, ConsentManager, MinorProtectionGuard,
    FairnessAuditor, TransparencyReporter, DataGovernance,
)
from .app.schemas import (
    ConsentRequest, DeletionRequest, FairnessAuditRequest,
)
from .app.logger import api_logger, ml_logger


app = FastAPI(
    title="Household Health Graph API with ML",
    version="0.2.0",
    description="Graph-based health analysis with ML-powered insights for households",
    contact={"name": "deokhwajeong", "url": "https://github.com/deokhwajeong"},
    license_info={"name": "MIT"},
)

# CORS configuration for cross-origin requests
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

# Ethics engine (singleton)
ethics_engine = EthicsEngine(retention_days=365)

api_logger.info("FastAPI application initialized", version="0.2.0", model_dir=str(MODEL_DIR))

# ---------------------------------------------------------
# Root / Health
# ---------------------------------------------------------

@app.get("/", tags=["System"])
def root():
    """Redirect to API documentation."""
    return RedirectResponse(url="/docs", status_code=307)


@app.get("/health", tags=["System"])
def health():
    """Health check endpoint."""
    api_logger.info("Health check requested")
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.2.0"
    }


@app.get("/ml/debug/graph-stats", tags=["Debug"])
def get_graph_stats(days: int = 60):
    """Get comprehensive graph statistics."""
    try:
        G = build_single_child_household(num_days=days)
        
        node_types = {}
        for node in G.nodes():
            node_type = G.nodes[node].get("type", "unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        avg_degree = sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0
        
        stats = {
            "total_nodes": len(G.nodes()),
            "total_edges": len(G.edges()),
            "node_types": node_types,
            "avg_degree": avg_degree,
            "days": days,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        api_logger.info("Graph stats requested", days=days, nodes=len(G.nodes()), edges=len(G.edges()))
        return stats
    except Exception as e:
        api_logger.error("Error getting graph stats", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/example-graph", tags=["Debug"])
def debug_example_graph():
    """Get example graph as JSON (node-link format)."""
    try:
        G = build_example_graph()
        result = json_graph.node_link_data(G)
        api_logger.info("Example graph served", nodes=len(G.nodes()), edges=len(G.edges()))
        return result
    except Exception as e:
        api_logger.error("Error generating example graph", exc=e)
        raise HTTPException(status_code=500, detail="Failed to generate example graph")


@app.get("/debug/synthetic-graph", tags=["Debug"])
def debug_synthetic_graph(days: int = 60):
    """Get synthetic household graph as JSON (node-link format)."""
    try:
        if days < 1 or days > 365:
            raise ValueError("days must be between 1 and 365")
        
        G = build_single_child_household(num_days=days)
        result = json_graph.node_link_data(G)
        api_logger.info("Synthetic graph served", days=days, nodes=len(G.nodes()), edges=len(G.edges()))
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        api_logger.error("Error generating synthetic graph", exc=e, days=days)
        raise HTTPException(status_code=500, detail="Failed to generate synthetic graph")


# ---------------------------------------------------------
# SVG Debug Endpoints
# ---------------------------------------------------------

@app.get("/debug/example-graph/svg", tags=["Debug"])
def debug_example_graph_svg():
    """Get example graph as SVG visualization."""
    try:
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
    except Exception as e:
        api_logger.error("Error generating SVG", exc=e)
        raise HTTPException(status_code=500, detail="Failed to generate SVG")


@app.get("/debug/synthetic-graph/svg", tags=["Debug"])
def debug_synthetic_graph_svg(days: int = 60):
    """Get synthetic graph as SVG visualization."""
    try:
        if days < 1 or days > 365:
            raise ValueError("days must be between 1 and 365")
        
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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        api_logger.error("Error generating SVG", exc=e, days=days)
        raise HTTPException(status_code=500, detail="Failed to generate SVG")


# ---------------------------------------------------------
# Optional HTML Wrapper
# ---------------------------------------------------------

@app.get("/debug/synthetic-graph/html", tags=["Debug"])
def debug_synthetic_graph_html(days: int = 60):
    """Get synthetic graph as interactive HTML visualization."""
    try:
        if days < 1 or days > 365:
            raise ValueError("days must be between 1 and 365")
        
        G = build_single_child_household(num_days=days)
        svg_bytes = graph_to_svg(G)
        svg = svg_bytes.decode("utf-8")

        html = f"""
        <html>
          <head>
            <title>Household Health Graph - Synthetic Data (days={days})</title>
            <style>
              body {{ font-family: sans-serif; margin: 20px; }}
              h2 {{ color: #333; }}
              svg {{ border: 1px solid #ccc; margin-top: 20px; }}
              .stats {{ background: #f5f5f5; padding: 10px; border-radius: 5px; margin: 20px 0; }}
            </style>
          </head>
          <body>
            <h2>Household Health Graph Visualization</h2>
            <div class="stats">
              <strong>Statistics:</strong> {len(G.nodes())} nodes, {len(G.edges())} edges, {days} days
            </div>
            {svg}
          </body>
        </html>
        """

        return Response(content=html, media_type="text/html")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        api_logger.error("Error generating HTML", exc=e, days=days)
        raise HTTPException(status_code=500, detail="Failed to generate HTML")


# ---------------------------------------------------------
# ML Endpoints: Graph Embeddings
# ---------------------------------------------------------

@app.post("/ml/embeddings/train", tags=["Embeddings"])
def train_embeddings(days: int = 60):
    """Train graph embeddings using GraphSAGE."""
    try:
        if days < 1 or days > 365:
            raise ValueError("days must be between 1 and 365")
        
        ml_logger.info("Training GraphSAGE embeddings", days=days)
        G = build_single_child_household(num_days=days)
        
        sage = SimpleGraphSAGE(embedding_dim=16, num_aggregation_layers=2)
        embeddings = sage.fit_embeddings(G)
        
        # Save model
        joblib.dump(sage, MODEL_DIR / "graph_sage.pkl")
        
        ml_logger.info("GraphSAGE training complete", nodes=len(G.nodes()), embeddings=len(embeddings))
        
        return {
            "status": "success",
            "num_nodes": len(G.nodes()),
            "num_embeddings": len(embeddings),
            "embedding_dim": 16,
            "message": "GraphSAGE model trained and saved"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        ml_logger.error("Error training embeddings", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ml/embeddings/daily/{person_id}/{date}", tags=["Embeddings"])
def get_daily_embedding(person_id: str, date: str, days: int = 60):
    """Get daily embedding for a specific person and date."""
    try:
        if not person_id or not date:
            raise ValueError("person_id and date are required")
        
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
            ml_logger.warning("No embedding found", person_id=person_id, date=date)
            raise HTTPException(status_code=404, detail=f"No data for {person_id} on {date}")
        
        return {
            "person_id": person_id,
            "date": date,
            "embedding": embedding.tolist(),
            "embedding_dim": len(embedding)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        ml_logger.error("Error getting embedding", exc=e, person_id=person_id, date=date)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# ML Endpoints: Anomaly Detection
# ---------------------------------------------------------

@app.post("/ml/anomalies/train")
def train_anomaly_detector(days: int = 60):
    """Train anomaly detection models."""
    G = build_single_child_household(num_days=days)
    
    detector = HealthAnomalyDetector(contamination=0.1)
    detector.fit(G, "person:mia")
    
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
    predictor.fit(G, "person:mia")
    
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


# ---------------------------------------------------------
# Ethics Endpoints: Privacy
# ---------------------------------------------------------

@app.get("/ethics/privacy/scan", tags=["Ethics - Privacy"])
def scan_pii(days: int = 60):
    """Scan the health graph for PII (Personally Identifiable Information)."""
    try:
        G = build_single_child_household(num_days=days)
        report = ethics_engine.privacy.scan_graph(G)
        api_logger.info("PII scan completed", risk=report["risk_level"])
        return report
    except Exception as e:
        api_logger.error("PII scan failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ethics/privacy/anonymize", tags=["Ethics - Privacy"])
def anonymize_graph_data(days: int = 60):
    """Create an anonymized version of the health graph."""
    try:
        G = build_single_child_household(num_days=days)
        anon_G = ethics_engine.privacy.anonymize_graph(G)

        return {
            "status": "success",
            "original_nodes": len(G.nodes()),
            "anonymized_nodes": len(anon_G.nodes()),
            "original_edges": len(G.edges()),
            "anonymized_edges": len(anon_G.edges()),
            "message": "Graph anonymized — person identifiers hashed, sensitive fields masked.",
        }
    except Exception as e:
        api_logger.error("Anonymization failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# Ethics Endpoints: Consent Management
# ---------------------------------------------------------

@app.post("/ethics/consent/grant", tags=["Ethics - Consent"])
def grant_consent(request: ConsentRequest):
    """Grant data usage consent for a person."""
    try:
        purposes = [DataPurpose(p) for p in request.purposes]
        result = ethics_engine.consent.grant_consent(
            person_id=request.person_id,
            purposes=purposes,
            granted_by=request.granted_by,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        api_logger.error("Consent grant failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ethics/consent/withdraw", tags=["Ethics - Consent"])
def withdraw_consent(request: ConsentRequest):
    """Withdraw data usage consent."""
    try:
        purposes = [DataPurpose(p) for p in request.purposes]
        result = ethics_engine.consent.withdraw_consent(
            person_id=request.person_id,
            purposes=purposes,
            withdrawn_by=request.granted_by,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        api_logger.error("Consent withdrawal failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ethics/consent/{person_id}", tags=["Ethics - Consent"])
def get_consent_status(person_id: str):
    """Get consent summary for a person."""
    return ethics_engine.consent.get_consent_summary(person_id)


@app.get("/ethics/consent/audit-trail", tags=["Ethics - Consent"])
def get_consent_audit_trail(person_id: str = None, limit: int = 50):
    """Get consent audit trail."""
    return ethics_engine.consent.get_audit_trail(person_id=person_id, limit=limit)


# ---------------------------------------------------------
# Ethics Endpoints: Minor Protection
# ---------------------------------------------------------

@app.get("/ethics/minors/detect", tags=["Ethics - Minor Protection"])
def detect_minors(days: int = 60):
    """Detect minor-aged members in the household graph."""
    try:
        G = build_single_child_household(num_days=days)
        minors = ethics_engine.minor_protection.detect_minors(G)
        return {
            "minors_found": len(minors),
            "details": minors,
        }
    except Exception as e:
        api_logger.error("Minor detection failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ethics/minors/setup-guardians", tags=["Ethics - Minor Protection"])
def setup_guardians(days: int = 60):
    """Automatically set up guardian relationships from the graph."""
    try:
        G = build_single_child_household(num_days=days)
        result = ethics_engine.minor_protection.setup_guardian_relationships(G)
        return result
    except Exception as e:
        api_logger.error("Guardian setup failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ethics/minors/evaluate/{person_id}", tags=["Ethics - Minor Protection"])
def evaluate_minor_operation(
    person_id: str,
    operation: str = "read",
    purpose: str = "health_monitoring",
):
    """Evaluate whether an operation is allowed for a person (enhanced checks for minors)."""
    try:
        data_purpose = DataPurpose(purpose)
        result = ethics_engine.minor_protection.evaluate_operation(
            person_id=person_id,
            operation=operation,
            purpose=data_purpose,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        api_logger.error("Minor evaluation failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# Ethics Endpoints: Fairness Audit
# ---------------------------------------------------------

@app.post("/ethics/fairness/audit-anomalies", tags=["Ethics - Fairness"])
def audit_anomaly_fairness(request: FairnessAuditRequest):
    """Audit anomaly detection scores for demographic bias."""
    try:
        G = build_single_child_household(num_days=request.days)

        # Train anomaly detector and collect scores per person
        from .app.anomaly_detection import HealthAnomalyDetector

        persons = [
            n for n, d in G.nodes(data=True) if d.get("type") == "person"
        ]
        all_scores: dict = {}

        for pid in persons:
            try:
                detector = HealthAnomalyDetector(contamination=0.1)
                detector.fit(G, pid)
                for date_key, score in detector.anomaly_scores.items():
                    all_scores[pid] = score
            except Exception:
                pass

        if not all_scores:
            return {
                "status": "insufficient_data",
                "message": "Could not compute anomaly scores for fairness audit.",
            }

        report = ethics_engine.fairness.audit_anomaly_scores(
            G, all_scores, sensitive_attr=request.sensitive_attribute
        )
        return report
    except Exception as e:
        api_logger.error("Fairness audit failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ethics/fairness/history", tags=["Ethics - Fairness"])
def get_fairness_history(limit: int = 20):
    """Get historical fairness audit results."""
    return ethics_engine.fairness.get_audit_history(limit=limit)


# ---------------------------------------------------------
# Ethics Endpoints: Transparency
# ---------------------------------------------------------

@app.get("/ethics/transparency/model-card/{model_name}", tags=["Ethics - Transparency"])
def get_model_card(model_name: str):
    """Generate a model card for an ML model (Mitchell et al., 2019)."""
    try:
        card = ethics_engine.transparency.generate_model_card(model_name)
        return card
    except Exception as e:
        api_logger.error("Model card generation failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ethics/transparency/compliance-report", tags=["Ethics - Transparency"])
def get_compliance_report():
    """Generate an ethics compliance report."""
    return ethics_engine.transparency.generate_compliance_report()


@app.get("/ethics/transparency/decisions", tags=["Ethics - Transparency"])
def get_decision_log(
    person_id: str = None,
    model_name: str = None,
    limit: int = 50,
):
    """Get ML decision audit log."""
    return ethics_engine.transparency.get_decision_log(
        person_id=person_id, model_name=model_name, limit=limit
    )


# ---------------------------------------------------------
# Ethics Endpoints: Data Governance
# ---------------------------------------------------------

@app.get("/ethics/governance/retention", tags=["Ethics - Governance"])
def check_retention_policy(days: int = 60):
    """Check data retention policy compliance."""
    try:
        G = build_single_child_household(num_days=days)
        report = ethics_engine.governance.check_retention_policy(G)
        return report
    except Exception as e:
        api_logger.error("Retention check failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ethics/governance/deletion-request", tags=["Ethics - Governance"])
def request_data_deletion(request: DeletionRequest):
    """Submit a right-to-deletion request (GDPR Art. 17 inspired)."""
    try:
        result = ethics_engine.governance.request_deletion(
            person_id=request.person_id, reason=request.reason
        )
        return result
    except Exception as e:
        api_logger.error("Deletion request failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ethics/governance/deletion-requests", tags=["Ethics - Governance"])
def list_deletion_requests(status: str = None):
    """List data deletion requests."""
    return ethics_engine.governance.get_deletion_requests(status=status)


@app.get("/ethics/governance/data-inventory", tags=["Ethics - Governance"])
def get_data_inventory(days: int = 60):
    """Generate a data inventory (GDPR Art. 30 — Records of Processing)."""
    try:
        G = build_single_child_household(num_days=days)
        inventory = ethics_engine.governance.generate_data_inventory(G)
        return inventory
    except Exception as e:
        api_logger.error("Data inventory failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# Ethics Endpoints: Full Audit
# ---------------------------------------------------------

@app.get("/ethics/audit", tags=["Ethics"])
def full_ethics_audit(days: int = 60):
    """Run a comprehensive ethics audit on the health graph.

    Combines: PII scan, minor detection, data retention, and data inventory.
    """
    try:
        G = build_single_child_household(num_days=days)
        report = ethics_engine.full_ethics_audit(G)
        return report
    except Exception as e:
        api_logger.error("Full ethics audit failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))

