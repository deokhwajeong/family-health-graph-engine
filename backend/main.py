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

import io
import json

from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Query
from fastapi.responses import RedirectResponse, Response, HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from networkx.readwrite import json_graph
import joblib
from pathlib import Path
from datetime import datetime
from typing import Optional
from functools import lru_cache

from .app.graph_builder import build_example_graph, GraphBuilder
from .app.synthetic_data import build_demo_family_household, DEMO_END_DATE
from .app.health_recommendations import FAMILY_PROFILES, generate_recommendations, compare_current_vs_ideal
from .app.apple_health_parser import parse_apple_health_export, QUANTITY_METRICS, WORKOUT_TYPES
from .app.food_ocr import NUTRITION_DB
from .app.graph_viz import graph_to_svg
from .app.graph_embeddings import SimpleGraphSAGE, extract_day_features
from .app.anomaly_detection import HealthAnomalyDetector, PatternCorrelationAnalyzer
from .app.pattern_analysis import HealthMetricPredictor, AssociationRuleAnalyzer
from .app.link_prediction import HealthLinkPredictor
from .app.node_classifier import FoodNutritionClassifier, ActivityIntensityClassifier, HealthStatusClassifier
from .app.apple_health import AppleHealthParser, add_health_data_to_graph
from .app.ethics import (
    EthicsEngine, DataPurpose,
)
from .app.schemas import (
    ConsentRequest, DeletionRequest, FairnessAuditRequest,
)
from .app.logger import api_logger, ml_logger
from .app.llm_engine import get_llm_engine, configure_llm_engine
from .app.rag_engine import get_rag_engine, GraphDocumentConverter
from .app.food_ocr import analyze_food_image as ocr_analyze_food_image, analyze_food_text, add_meal_to_graph


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

# Frontend directory
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
BACKEND_STATIC_DIR = Path(__file__).parent / "static"

# Mount static files if frontend exists
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")
    api_logger.info("Frontend static files mounted", frontend_dir=str(FRONTEND_DIR))

# ---------------------------------------------------------
# Demo graph + model helpers
# ---------------------------------------------------------

@lru_cache(maxsize=16)
def _cached_demo_graph(days: int):
    return build_demo_family_household(num_days=days)


def build_demo_graph(days: int = 90):
    """Return a fresh copy of the synthetic Demo family graph for `days` days."""
    if days < 1 or days > 365:
        raise HTTPException(status_code=400, detail="days must be between 1 and 365")
    return _cached_demo_graph(days).copy()


_fitted_models: dict = {}


def _model_file(kind: str, person_id: str) -> str:
    """Per-person model filename, e.g. anomaly_detector_leo.pkl."""
    key = "".join(ch for ch in person_id.split(":")[-1].lower() if ch.isalnum() or ch in "-_")
    return f"{kind}_{key}.pkl"


def load_or_fit(filename: str, fit_fn, cache_key=None):
    """
    Use a model saved by a /train endpoint (or backend/train_models.py) if it
    exists and loads cleanly; otherwise fit one on demand and keep it in memory.
    The repo ships without .pkl files, so the on-demand path is the default.
    """
    path = MODEL_DIR / filename
    if path.exists():
        try:
            return joblib.load(path)
        except Exception as e:  # stale or incompatible pickle
            ml_logger.warning("Could not load saved model, refitting", model=filename, error=str(e))
    key = (filename, cache_key)
    if key not in _fitted_models:
        _fitted_models[key] = fit_fn()
    return _fitted_models[key]


# Ethics engine (singleton)
ethics_engine = EthicsEngine(retention_days=365)

api_logger.info("FastAPI application initialized", version="0.2.0", model_dir=str(MODEL_DIR))

# ---------------------------------------------------------
# Root / Health
# ---------------------------------------------------------

@app.get("/", tags=["System"])
def root():
    """Serve the frontend dashboard or redirect to API docs."""
    # Prefer the self-contained backend/static/index.html (has all features inline)
    backend_index = BACKEND_STATIC_DIR / "index.html"
    if backend_index.exists():
        return FileResponse(
            str(backend_index),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"},
        )
    # Fallback to frontend/index.html
    frontend_index = FRONTEND_DIR / "index.html"
    if frontend_index.exists():
        return FileResponse(
            str(frontend_index),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"},
        )
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


# ---------------------------------------------------------
# Graph Visualization API
# ---------------------------------------------------------

NODE_COLORS = {
    "household": "#1f77b4",
    "person": "#ff7f0e",
    "day": "#2ca02c",
    "activity": "#17becf",
    "meal": "#d62728",
    "sleep": "#9467bd",
    "food": "#e377c2",
    "metric": "#bcbd22",
}
NODE_SIZES = {
    "household": 14,
    "person": 12,
    "day": 5,
    "activity": 6,
    "meal": 5,
    "sleep": 6,
    "food": 7,
    "metric": 4,
}


@app.get("/api/graph/d3", tags=["Graph Visualization"])
def get_graph_d3(days: int = 14, max_nodes: int = 500):
    """Return graph data formatted for D3.js force-directed visualization."""
    try:
        G = build_demo_graph(days)

        # Downsample if too many nodes
        all_nodes = list(G.nodes())
        if len(all_nodes) > max_nodes:
            # Keep persons, household, days, and proportional sample of rest
            priority = [n for n in all_nodes if G.nodes[n].get("type") in ("person", "household", "day")]
            rest = [n for n in all_nodes if n not in priority]
            import random
            random.shuffle(rest)
            keep = set(priority + rest[: max_nodes - len(priority)])
            G = G.subgraph(keep).copy()

        nodes = []
        for n in G.nodes():
            data = G.nodes[n]
            ntype = data.get("type", "unknown")
            label = data.get("name") or data.get("meal_type") or data.get("metric_type") or n.split(":")[-1]
            node_obj = {
                "id": n,
                "label": label,
                "type": ntype,
                "color": NODE_COLORS.get(ntype, "#888"),
                "size": NODE_SIZES.get(ntype, 5),
            }
            # Extra attributes for tooltip
            if "date" in data:
                node_obj["date"] = data["date"]
            if "name" in data:
                node_obj["name"] = data["name"]
            if ntype == "sleep":
                node_obj["duration"] = round(data.get("duration_hours", 0), 1)
                node_obj["quality"] = round(data.get("quality_score", 0), 1)
            if ntype == "food":
                node_obj["kcal"] = data.get("kcal", 0)
                node_obj["category"] = data.get("category", "")
            if ntype == "metric":
                node_obj["value"] = round(data.get("value", 0), 1)
                node_obj["metric_type"] = data.get("metric_type", "")
            if ntype == "activity":
                node_obj["duration_min"] = data.get("duration_min", 0)
                node_obj["intensity"] = data.get("intensity", "")
                node_obj["calories_burned"] = data.get("calories_burned", 0)
            nodes.append(node_obj)

        links = []
        for u, v, d in G.edges(data=True):
            links.append({
                "source": u,
                "target": v,
                "relation": d.get("relation", "connected"),
            })

        # Summary stats
        node_types = {}
        for n in G.nodes():
            t = G.nodes[n].get("type", "unknown")
            node_types[t] = node_types.get(t, 0) + 1

        return {
            "nodes": nodes,
            "links": links,
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(links),
                "node_types": node_types,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Graph D3 export failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/stats", tags=["Graph Visualization"])
def get_graph_viz_stats(days: int = 14):
    """Get graph statistics for the visualization page."""
    try:
        G = build_demo_graph(days)

        node_types = {}
        persons = {}
        for n in G.nodes():
            data = G.nodes[n]
            t = data.get("type", "unknown")
            node_types[t] = node_types.get(t, 0) + 1
            if t == "person":
                persons[n] = data.get("name", n)

        degrees = dict(G.degree())
        avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
        max_degree = max(degrees.values()) if degrees else 0
        density = (2 * G.number_of_edges()) / (G.number_of_nodes() * (G.number_of_nodes() - 1)) if G.number_of_nodes() > 1 else 0

        return {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "node_types": node_types,
            "avg_degree": round(avg_degree, 2),
            "max_degree": max_degree,
            "density": round(density, 6),
            "persons": persons,
            "days": days,
        }
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Graph stats failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/insights", tags=["Graph Visualization"])
def get_graph_insights(days: int = 14):
    """
    Analyze the health graph and return key insights & implications.

    Returns structural analysis, nutritional patterns, sleep quality trends,
    activity correlations, and actionable recommendations.
    """
    try:
        G = build_demo_graph(days)
        insights = _analyze_graph_insights(G, days)
        return insights
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Graph insights failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


def _analyze_graph_insights(G, days: int) -> dict:
    """Extract meaningful insights from the health graph."""
    import statistics

    persons = {}
    sleep_data = {}       # person -> [duration, quality]
    meal_data = {}        # person -> [meal_count_per_day]
    food_freq = {}        # food_name -> count
    food_categories = {}  # category -> count
    activity_data = {}    # person -> [(name, duration, intensity, cal)]
    metric_data = {}      # person -> {metric_type -> [values]}
    day_count = {}        # person -> count

    for n in G.nodes():
        d = G.nodes[n]
        t = d.get("type")
        pid = d.get("person_id", "")

        if t == "person":
            persons[n] = d
        elif t == "sleep":
            sleep_data.setdefault(pid, []).append({
                "duration": d.get("duration_hours", 0),
                "quality": d.get("quality_score", 0),
                "date": d.get("date", ""),
            })
        elif t == "day":
            day_count[pid] = day_count.get(pid, 0) + 1
        elif t == "activity":
            activity_data.setdefault(pid, []).append({
                "name": d.get("name", ""),
                "duration_min": d.get("duration_min", 0),
                "intensity": d.get("intensity", ""),
                "calories": d.get("calories_burned", 0),
            })
        elif t == "metric":
            metric_data.setdefault(pid, {}).setdefault(
                d.get("metric_type", "unknown"), []
            ).append(d.get("value", 0))
        elif t == "food":
            fname = d.get("name", "unknown")
            fcat = d.get("category", "unknown")
            food_freq[fname] = food_freq.get(fname, 0) + 1
            food_categories[fcat] = food_categories.get(fcat, 0) + 1

    # Count meals per person
    for n in G.nodes():
        d = G.nodes[n]
        if d.get("type") == "meal":
            pid = d.get("person_id", "")
            meal_data.setdefault(pid, []).append(d.get("meal_type", ""))

    # ── Build insights ──
    result_insights = []
    person_summaries = []

    for pid, pdata in persons.items():
        pname = pdata.get("name", pid)
        pemoji = pdata.get("emoji", "👤")
        p_sleep = sleep_data.get(pid, [])
        p_activities = activity_data.get(pid, [])
        p_metrics = metric_data.get(pid, {})

        summary = {"name": pname, "emoji": pemoji, "id": pid, "findings": []}

        # ── Sleep Analysis ──
        if p_sleep:
            durations = [s["duration"] for s in p_sleep]
            qualities = [s["quality"] for s in p_sleep]
            avg_dur = statistics.mean(durations)
            avg_qual = statistics.mean(qualities)
            min_dur = min(durations)
            max_dur = max(durations)
            std_dur = statistics.stdev(durations) if len(durations) > 1 else 0

            sleep_emoji = "✅" if avg_dur >= 8 else ("⚠️" if avg_dur >= 6.5 else "🔴")
            summary["sleep"] = {
                "avg_duration": round(avg_dur, 1),
                "avg_quality": round(avg_qual, 1),
                "min_duration": round(min_dur, 1),
                "max_duration": round(max_dur, 1),
                "consistency": round(std_dur, 2),
            }
            # Generate findings
            if avg_dur < 7:
                summary["findings"].append({
                    "type": "warning",
                    "category": "sleep",
                    "title": f"{sleep_emoji} Insufficient Sleep Detected",
                    "detail": f"Average sleep {avg_dur:.1f}h - below the recommended 7-9 hours.",
                    "recommendation": "Consider moving your bedtime 30 minutes earlier.",
                })
            elif avg_dur >= 8.5:
                summary["findings"].append({
                    "type": "positive",
                    "category": "sleep",
                    "title": f"{sleep_emoji} Excellent Sleep Pattern",
                    "detail": f"Averaging {avg_dur:.1f}h of sleep - getting sufficient rest.",
                })
            if std_dur > 1.5:
                summary["findings"].append({
                    "type": "warning",
                    "category": "sleep",
                    "title": "⏰ Irregular Sleep Schedule",
                    "detail": f"Sleep deviation {std_dur:.1f}h - a consistent bedtime pattern is needed.",
                    "recommendation": "Try going to bed and waking up at the same time every day.",
                })
            if avg_qual < 60:
                summary["findings"].append({
                    "type": "warning",
                    "category": "sleep",
                    "title": "😴 Low Sleep Quality",
                    "detail": f"Average sleep quality {avg_qual:.0f}/100 - improvement needed.",
                    "recommendation": "Reduce screen time before bed and create a quiet environment.",
                })
            elif avg_qual >= 80:
                summary["findings"].append({
                    "type": "positive",
                    "category": "sleep",
                    "title": "😊 Good Sleep Quality",
                    "detail": f"Average sleep quality {avg_qual:.0f}/100 - good.",
                })

        # ── Activity Analysis ──
        if p_activities:
            total_cal = sum(a["calories"] for a in p_activities)
            total_min = sum(a["duration_min"] for a in p_activities)
            act_names = [a["name"] for a in p_activities]
            unique_activities = list(set(act_names))
            high_intensity = sum(1 for a in p_activities if a["intensity"] == "high")

            summary["activity"] = {
                "total_sessions": len(p_activities),
                "total_minutes": total_min,
                "total_calories": total_cal,
                "unique_activities": unique_activities,
                "high_intensity_count": high_intensity,
            }

            days_with_activity = len(p_activities)
            if days_with_activity / max(day_count.get(pid, 1), 1) < 0.4:
                summary["findings"].append({
                    "type": "warning",
                    "category": "activity",
                    "title": "🏃 Low Exercise Frequency",
                    "detail": f"{days_with_activity} workouts in {days} days - more frequent activity needed.",
                    "recommendation": "Aim for at least 30 minutes of light activity every day.",
                })
            else:
                summary["findings"].append({
                    "type": "positive",
                    "category": "activity",
                    "title": "💪 Active Exercise Habits",
                    "detail": f"{days_with_activity} workouts, {total_min} total minutes, {total_cal}kcal burned",
                })

            if unique_activities:
                summary["findings"].append({
                    "type": "info",
                    "category": "activity",
                    "title": "🎯 Main Activity Types",
                    "detail": ", ".join(unique_activities),
                })

        # ── Metric Analysis ──
        for mtype, values in p_metrics.items():
            avg_v = statistics.mean(values)
            if mtype == "energy_level" and avg_v < 50:
                summary["findings"].append({
                    "type": "warning",
                    "category": "wellness",
                    "title": "⚡ Low Energy Level",
                    "detail": f"Average energy {avg_v:.0f}/100 - diet/sleep improvement needed.",
                })
            elif mtype == "energy_level" and avg_v >= 70:
                summary["findings"].append({
                    "type": "positive",
                    "category": "wellness",
                    "title": "⚡ Good Energy Level",
                    "detail": f"Average energy {avg_v:.0f}/100",
                })
            if mtype == "mood" and avg_v < 50:
                summary["findings"].append({
                    "type": "warning",
                    "category": "wellness",
                    "title": "😔 Low Mood Tendency",
                    "detail": f"Average mood score {avg_v:.0f}/100 - attention needed.",
                    "recommendation": "Increased physical activity and adequate sleep can help improve mood.",
                })
            elif mtype == "mood" and avg_v >= 70:
                summary["findings"].append({
                    "type": "positive",
                    "category": "wellness",
                    "title": "😊 Good Mood Status",
                    "detail": f"Average mood score {avg_v:.0f}/100",
                })

            summary.setdefault("metrics", {})[mtype] = round(avg_v, 1)

        person_summaries.append(summary)

    # ── Overall Nutritional Analysis ──
    top_foods = sorted(food_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    cat_dist = sorted(food_categories.items(), key=lambda x: x[1], reverse=True)

    nutrition_insight = {
        "top_foods": [{"name": f, "count": c} for f, c in top_foods],
        "category_distribution": {k: v for k, v in cat_dist},
        "unique_foods": len(food_freq),
    }

    # Food diversity finding
    if len(food_freq) < 8:
        result_insights.append({
            "type": "warning",
            "category": "nutrition",
            "title": "🥗 Low Diet Diversity",
            "detail": f"Only {len(food_freq)} types of food consumed - more varied nutrition needed.",
            "recommendation": "Try adding new fruits, vegetables, and protein sources each week.",
        })
    elif len(food_freq) >= 15:
        result_insights.append({
            "type": "positive",
            "category": "nutrition",
            "title": "🥗 Excellent Diet Diversity",
            "detail": f"Consuming a diverse range of {len(food_freq)} types of food.",
        })

    # Category balance check
    vegetable_count = food_categories.get("vegetable", 0)
    if vegetable_count == 0:
        result_insights.append({
            "type": "warning",
            "category": "nutrition",
            "title": "🥬 Insufficient Vegetable Intake",
            "detail": "No vegetable intake recorded during this period.",
            "recommendation": "Add vegetables to every meal.",
        })

    # ── Graph Structure Insights ──
    degrees = dict(G.degree())
    hub_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    hub_info = []
    for n_id, deg in hub_nodes:
        nd = G.nodes[n_id]
        hub_info.append({
            "id": n_id,
            "label": nd.get("name") or nd.get("meal_type") or n_id.split(":")[-1],
            "type": nd.get("type", "unknown"),
            "degree": deg,
        })

    # Connected components
    if G.is_directed():
        import networkx as nx
        n_weakly = nx.number_weakly_connected_components(G)
        n_strongly = nx.number_strongly_connected_components(G)
        structure_info = {
            "weakly_connected_components": n_weakly,
            "strongly_connected_components": n_strongly,
        }
    else:
        import networkx as nx
        structure_info = {
            "connected_components": nx.number_connected_components(G),
        }

    structure_info["hub_nodes"] = hub_info
    structure_info["total_nodes"] = G.number_of_nodes()
    structure_info["total_edges"] = G.number_of_edges()

    if n_weakly > 1 if G.is_directed() else structure_info.get("connected_components", 1) > 1:
        result_insights.append({
            "type": "info",
            "category": "structure",
            "title": "🔗 Disconnected Graph Components Found",
            "detail": "The graph contains independent disconnected components. Some health data records may be missing.",
        })

    return {
        "period_days": days,
        "person_summaries": person_summaries,
        "nutrition": nutrition_insight,
        "structure": structure_info,
        "global_insights": result_insights,
        "generated_at": datetime.utcnow().isoformat(),
    }


@app.get("/ml/debug/graph-stats", tags=["Debug"])
def get_graph_stats(days: int = 60):
    """Get comprehensive graph statistics."""
    try:
        G = build_demo_graph(days)
        
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
    except HTTPException:
        raise
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
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Error generating example graph", exc=e)
        raise HTTPException(status_code=500, detail="Failed to generate example graph")


@app.get("/debug/synthetic-graph", tags=["Debug"])
def debug_synthetic_graph(days: int = 60):
    """Get synthetic household graph as JSON (node-link format)."""
    try:
        if days < 1 or days > 365:
            raise ValueError("days must be between 1 and 365")
        
        G = build_demo_graph(days)
        result = json_graph.node_link_data(G)
        api_logger.info("Synthetic graph served", days=days, nodes=len(G.nodes()), edges=len(G.edges()))
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
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
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Error generating SVG", exc=e)
        raise HTTPException(status_code=500, detail="Failed to generate SVG")


@app.get("/debug/synthetic-graph/svg", tags=["Debug"])
def debug_synthetic_graph_svg(days: int = 60):
    """Get synthetic graph as SVG visualization."""
    try:
        if days < 1 or days > 365:
            raise ValueError("days must be between 1 and 365")
        
        G = build_demo_graph(days)
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
    except HTTPException:
        raise
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
        
        G = build_demo_graph(days)
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
    except HTTPException:
        raise
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
        G = build_demo_graph(days)
        
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
    except HTTPException:
        raise
    except Exception as e:
        ml_logger.error("Error training embeddings", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ml/embeddings/daily/{person_id}/{date}", tags=["Embeddings"])
def get_daily_embedding(person_id: str, date: str, days: int = 60):
    """Get daily embedding for a specific person and date."""
    try:
        if not person_id or not date:
            raise ValueError("person_id and date are required")
        
        G = build_demo_graph(days)
        
        # Embeddings are transductive (one vector per node), so always fit on
        # the graph being queried instead of reusing a model from another window.
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
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        ml_logger.error("Error getting embedding", exc=e, person_id=person_id, date=date)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# ML Endpoints: Anomaly Detection
# ---------------------------------------------------------

def _fit_anomaly(G, person_id: str):
    detector = HealthAnomalyDetector(contamination=0.1)
    detector.fit(G, person_id)
    return detector


def _fit_predictor(G, person_id: str):
    predictor = HealthMetricPredictor(prediction_window=1)
    predictor.fit(G, person_id)
    return predictor


@app.post("/ml/anomalies/train")
def train_anomaly_detector(days: int = 60, person_id: str = "person:leo"):
    """Train anomaly detection models."""
    G = build_demo_graph(days)
    
    detector = HealthAnomalyDetector(contamination=0.1)
    detector.fit(G, person_id)
    
    # Save model
    joblib.dump(detector, MODEL_DIR / _model_file("anomaly_detector", person_id))
    
    return {
        "status": "success",
        "contamination": 0.1,
        "anomalies_detected": sum(1 for s in detector.anomaly_scores.values() if s > 0.15),
        "message": "Anomaly detector trained and saved"
    }


@app.get("/ml/anomalies/detect/{person_id}/{date}")
def detect_anomaly(person_id: str, date: str, days: int = 60):
    """Detect if a specific day is anomalous."""
    G = build_demo_graph(days)
    
    try:
        detector = load_or_fit(_model_file("anomaly_detector", person_id), lambda: _fit_anomaly(G, person_id), (person_id, days))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    result = detector.predict(G, person_id, date)
    return result


@app.get("/ml/anomalies/correlations/{person_id}")
def analyze_correlations(person_id: str, days: int = 60):
    """Analyze correlations between health behaviors and outcomes."""
    G = build_demo_graph(days)
    
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
def train_predictor(days: int = 60, person_id: str = "person:leo"):
    """Train health metric predictors."""
    G = build_demo_graph(days)
    
    predictor = HealthMetricPredictor(prediction_window=1)
    predictor.fit(G, person_id)
    
    # Save model
    joblib.dump(predictor, MODEL_DIR / _model_file("metric_predictor", person_id))
    
    return {
        "status": "success",
        "prediction_window": 1,
        "message": "Metric predictor trained and saved"
    }


@app.get("/ml/prediction/metrics/{person_id}/{date}")
def predict_metrics(person_id: str, date: str, days: int = 60):
    """Predict future health metrics."""
    G = build_demo_graph(days)
    
    try:
        predictor = load_or_fit(_model_file("metric_predictor", person_id), lambda: _fit_predictor(G, person_id), (person_id, days))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    result = predictor.predict(G, person_id, date)
    return result


@app.get("/ml/patterns/association-rules/{person_id}")
def discover_rules(person_id: str, days: int = 60):
    """Discover association rules in health behaviors."""
    G = build_demo_graph(days)
    
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
    G = build_demo_graph(days)
    
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
    G = build_demo_graph(days)
    
    def _fit():
        p = HealthLinkPredictor()
        p.fit(G)
        return p

    predictor = load_or_fit("link_predictor.pkl", _fit, days)
    
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
    G = build_demo_graph(days)
    
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
    G = build_demo_graph(days)
    
    def _fit():
        c = FoodNutritionClassifier()
        c.fit(G)
        return c

    clf = load_or_fit("food_classifier.pkl", _fit, days)
    
    result = clf.predict(G, food_id)
    return result


@app.get("/ml/classification/activity/{activity_id}")
def classify_activity(activity_id: str, days: int = 60):
    """Classify an activity's intensity."""
    G = build_demo_graph(days)
    
    def _fit():
        c = ActivityIntensityClassifier()
        c.fit(G)
        return c

    clf = load_or_fit("activity_classifier.pkl", _fit, days)
    
    result = clf.predict(G, activity_id)
    return result


@app.get("/ml/classification/daily-status/{person_id}/{date}")
def classify_daily_status(person_id: str, date: str, days: int = 60):
    """Classify overall health status for a day."""
    G = build_demo_graph(days)
    
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
    G = build_demo_graph(days)
    
    # Daily features
    day_features = extract_day_features(G, person_id, date)
    
    # Status
    status = HealthStatusClassifier.classify_day(day_features)
    
    # Anomaly detector and next-day predictor (saved model if present, else fit on demand)
    try:
        detector = load_or_fit(_model_file("anomaly_detector", person_id), lambda: _fit_anomaly(G, person_id), (person_id, days))
        anomaly = detector.predict(G, person_id, date)
        predictor = load_or_fit(_model_file("metric_predictor", person_id), lambda: _fit_predictor(G, person_id), (person_id, days))
        predictions = predictor.predict(G, person_id, date)
    except ValueError:  # unknown person or too few days to fit
        anomaly = {"anomaly_score": 0.0, "is_anomaly": False, "reasons": []}
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
    G = build_demo_graph(days)
    
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
        "lagged_correlations": correlations,
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
        G = build_demo_graph(days)
        report = ethics_engine.privacy.scan_graph(G)
        api_logger.info("PII scan completed", risk=report["risk_level"])
        return report
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("PII scan failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ethics/privacy/anonymize", tags=["Ethics - Privacy"])
def anonymize_graph_data(days: int = 60):
    """Create an anonymized version of the health graph."""
    try:
        G = build_demo_graph(days)
        anon_G = ethics_engine.privacy.anonymize_graph(G)

        return {
            "status": "success",
            "original_nodes": len(G.nodes()),
            "anonymized_nodes": len(anon_G.nodes()),
            "original_edges": len(G.edges()),
            "anonymized_edges": len(anon_G.edges()),
            "message": "Graph anonymized - person identifiers hashed, sensitive fields masked.",
        }
    except HTTPException:
        raise
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
    except HTTPException:
        raise
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
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Consent withdrawal failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ethics/consent/audit-trail", tags=["Ethics - Consent"])
def get_consent_audit_trail(person_id: str = None, limit: int = 50):
    """Get consent audit trail."""
    return ethics_engine.consent.get_audit_trail(person_id=person_id, limit=limit)


@app.get("/ethics/consent/{person_id}", tags=["Ethics - Consent"])
def get_consent_status(person_id: str):
    """Get consent summary for a person."""
    return ethics_engine.consent.get_consent_summary(person_id)


# ---------------------------------------------------------
# Ethics Endpoints: Minor Protection
# ---------------------------------------------------------

@app.get("/ethics/minors/detect", tags=["Ethics - Minor Protection"])
def detect_minors(days: int = 60):
    """Detect minor-aged members in the household graph."""
    try:
        G = build_demo_graph(days)
        minors = ethics_engine.minor_protection.detect_minors(G)
        return {
            "minors_found": len(minors),
            "details": minors,
        }
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Minor detection failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ethics/minors/setup-guardians", tags=["Ethics - Minor Protection"])
def setup_guardians(days: int = 60):
    """Automatically set up guardian relationships from the graph."""
    try:
        G = build_demo_graph(days)
        result = ethics_engine.minor_protection.setup_guardian_relationships(G)
        return result
    except HTTPException:
        raise
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
    except HTTPException:
        raise
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
        G = build_demo_graph(request.days)

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
    except HTTPException:
        raise
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
    except HTTPException:
        raise
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
        G = build_demo_graph(days)
        report = ethics_engine.governance.check_retention_policy(G)
        return report
    except HTTPException:
        raise
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
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Deletion request failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ethics/governance/deletion-requests", tags=["Ethics - Governance"])
def list_deletion_requests(status: str = None):
    """List data deletion requests."""
    return ethics_engine.governance.get_deletion_requests(status=status)


@app.get("/ethics/governance/data-inventory", tags=["Ethics - Governance"])
def get_data_inventory(days: int = 60):
    """Generate a data inventory (GDPR Art. 30 - Records of Processing)."""
    try:
        G = build_demo_graph(days)
        inventory = ethics_engine.governance.generate_data_inventory(G)
        return inventory
    except HTTPException:
        raise
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
        G = build_demo_graph(days)
        report = ethics_engine.full_ethics_audit(G)
        return report
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Full ethics audit failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# Apple Health & iOS Shortcuts Integration
# ---------------------------------------------------------

# In-memory graph for demo purposes (in production, use a database)
_demo_graph = None

def get_demo_graph():
    """Get or create demo graph."""
    global _demo_graph
    if _demo_graph is None:
        _demo_graph = GraphBuilder()
        # Add household
        household_id = "household:demo-family"
        _demo_graph.add_household(
            household_id,
            name="Demo family",
            timezone="America/Los_Angeles"
        )
        # Add Leo and Jordan persons
        _demo_graph.add_person(
            person_id="leo",
            household_id=household_id,
            name="Leo",
            age=8,
            emoji="👦",
            role="child",
        )
        _demo_graph.add_person(
            person_id="jordan",
            household_id=household_id,
            name="Jordan",
            age=35,
            emoji="👩",
            role="parent",
        )
    return _demo_graph


@app.post("/api/health/import-xml", tags=["Apple Health"])
async def import_apple_health_xml(
    file: UploadFile = File(...),
    person_name: str = Form(...)
):
    """
    Import Apple Health export.xml file and add data to the graph.
    
    Args:
        file: The export.xml file from Apple Health
        person_name: Person to associate the data with (leo or jordan)
    
    Returns:
        Import statistics and summary
    """
    try:
        # Validate person name
        person_id = person_name.lower()
        if person_id not in ["leo", "jordan"]:
            raise HTTPException(
                status_code=400,
                detail="person_name must be 'Leo' or 'Jordan'"
            )
        
        # Read XML file
        xml_content = await file.read()
        
        # Parse XML
        api_logger.info("Parsing Apple Health XML", person=person_id, size=len(xml_content))
        health_data = AppleHealthParser.parse_xml(xml_content)
        
        # Add to graph
        gb = get_demo_graph()
        stats = add_health_data_to_graph(gb, person_id, health_data)
        
        # Calculate summary statistics
        daily_steps = AppleHealthParser.aggregate_daily_steps(health_data["steps"])
        daily_hr = AppleHealthParser.calculate_average_heart_rate(health_data["heart_rate"])
        
        summary = {
            "person_name": person_name,
            "records_imported": {
                "steps_days": stats["steps_days"],
                "calories_days": stats["calories_days"],
                "heart_rate_days": stats["heart_rate_days"],
                "sleep_sessions": stats["sleep_sessions"],
            },
            "statistics": {
                "total_steps": int(sum(daily_steps.values())) if daily_steps else 0,
                "avg_steps_per_day": int(sum(daily_steps.values()) / len(daily_steps)) if daily_steps else 0,
                "avg_heart_rate": int(sum(daily_hr.values()) / len(daily_hr)) if daily_hr else 0,
                "total_sleep_sessions": stats["sleep_sessions"],
            }
        }
        
        api_logger.info("Apple Health import successful", person=person_id, stats=stats)
        return summary
        
    except ValueError as e:
        api_logger.error("XML parsing failed", exc=e)
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Import failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/health/steps", tags=["iOS Shortcuts"])
async def add_steps(
    person_name: str = Form(...),
    steps: int = Form(...),
    timestamp: str = Form(...)
):
    """
    Add step count data from iOS Shortcuts.
    
    Args:
        person_name: Person name (Leo or Jordan)
        steps: Number of steps
        timestamp: ISO 8601 timestamp
    """
    try:
        person_id = person_name.lower()
        if person_id not in ["leo", "jordan"]:
            raise HTTPException(status_code=400, detail="person_name must be 'Leo' or 'Jordan'")
        
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        date = dt.date().isoformat()
        
        gb = get_demo_graph()
        
        # Ensure day exists
        day_id = f"day:{person_id}:{date}"
        try:
            gb.add_day(
                day_id=day_id,
                person_id=person_id,
                date=date,
                weekday=dt.weekday(),
            )
        except Exception:
            pass
        
        # Add activity
        activity_id = f"activity:{person_id}:{date}:steps:{dt.hour}"
        gb.add_activity(
            activity_id=activity_id,
            person_id=person_id,
            date=date,
            name="Steps",
            day_id=day_id,
            category="walking",
            steps=steps,
        )
        
        api_logger.info("Steps added", person=person_id, steps=steps, date=date)
        return {"status": "success", "person": person_name, "steps": steps, "date": date}
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Add steps failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/health/heart-rate", tags=["iOS Shortcuts"])
async def add_heart_rate(
    person_name: str = Form(...),
    bpm: int = Form(...),
    timestamp: str = Form(...)
):
    """
    Add heart rate data from iOS Shortcuts.
    
    Args:
        person_name: Person name (Leo or Jordan)
        bpm: Heart rate in beats per minute
        timestamp: ISO 8601 timestamp
    """
    try:
        person_id = person_name.lower()
        if person_id not in ["leo", "jordan"]:
            raise HTTPException(status_code=400, detail="person_name must be 'Leo' or 'Jordan'")
        
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))  # validates the timestamp
        
        gb = get_demo_graph()
        metric_id = f"metric:{person_id}:{timestamp}:heart_rate"
        gb.add_metric(
            metric_id=metric_id,
            person_id=person_id,
            metric_type="heart_rate",
            value=float(bpm),
            unit="bpm",
            timestamp=timestamp,
        )
        
        api_logger.info("Heart rate added", person=person_id, bpm=bpm)
        return {"status": "success", "person": person_name, "bpm": bpm}
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Add heart rate failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/health/sleep", tags=["iOS Shortcuts"])
async def add_sleep(
    person_name: str = Form(...),
    start_time: str = Form(...),
    end_time: str = Form(...),
    quality: str = Form("good")
):
    """
    Add sleep data from iOS Shortcuts.
    
    Args:
        person_name: Person name (Leo or Jordan)
        start_time: Sleep start time (ISO 8601)
        end_time: Sleep end time (ISO 8601)
        quality: Sleep quality (good, fair, poor)
    """
    try:
        person_id = person_name.lower()
        if person_id not in ["leo", "jordan"]:
            raise HTTPException(status_code=400, detail="person_name must be 'Leo' or 'Jordan'")
        
        start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
        
        date = start_dt.date().isoformat()
        duration_hours = (end_dt - start_dt).total_seconds() / 3600
        
        # Map quality to score
        quality_scores = {"good": 85, "fair": 70, "poor": 50}
        quality_score = quality_scores.get(quality.lower(), 75)
        
        gb = get_demo_graph()
        
        # Ensure day exists
        day_id = f"day:{person_id}:{date}"
        try:
            gb.add_day(
                day_id=day_id,
                person_id=person_id,
                date=date,
                weekday=start_dt.weekday(),
            )
        except Exception:
            pass
        
        # Add sleep
        sleep_id = f"sleep:{person_id}:{date}"
        gb.add_sleep(
            sleep_id=sleep_id,
            person_id=person_id,
            date=date,
            start_time=start_dt.time().isoformat(),
            end_time=end_dt.time().isoformat(),
            day_id=day_id,
            duration_hours=duration_hours,
            quality_score=quality_score,
        )
        
        api_logger.info("Sleep added", person=person_id, duration=duration_hours, quality=quality)
        return {
            "status": "success",
            "person": person_name,
            "duration_hours": round(duration_hours, 2),
            "quality": quality
        }
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Add sleep failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/health/calories", tags=["iOS Shortcuts"])
async def add_calories(
    person_name: str = Form(...),
    calories: int = Form(...),
    type: str = Form("active"),
    timestamp: str = Form(...)
):
    """
    Add calorie data from iOS Shortcuts.
    
    Args:
        person_name: Person name (Leo or Jordan)
        calories: Calorie amount
        type: Type of calories (active or dietary)
        timestamp: ISO 8601 timestamp
    """
    try:
        person_id = person_name.lower()
        if person_id not in ["leo", "jordan"]:
            raise HTTPException(status_code=400, detail="person_name must be 'Leo' or 'Jordan'")
        
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        date = dt.date().isoformat()
        
        gb = get_demo_graph()
        
        # Ensure day exists
        day_id = f"day:{person_id}:{date}"
        try:
            gb.add_day(
                day_id=day_id,
                person_id=person_id,
                date=date,
                weekday=dt.weekday(),
            )
        except Exception:
            pass
        
        if type.lower() == "active":
            # Add as activity
            activity_id = f"activity:{person_id}:{date}:calories:{dt.hour}"
            gb.add_activity(
                activity_id=activity_id,
                person_id=person_id,
                date=date,
                name="Active Calories",
                day_id=day_id,
                category="calories",
                calories_burned=calories,
            )
        else:
            # Add as dietary metric
            metric_id = f"metric:{person_id}:{timestamp}:dietary_calories"
            gb.add_metric(
                metric_id=metric_id,
                person_id=person_id,
                metric_type="dietary_calories",
                value=float(calories),
                unit="kcal",
                timestamp=timestamp,
            )
        
        api_logger.info("Calories added", person=person_id, calories=calories, type=type)
        return {"status": "success", "person": person_name, "calories": calories, "type": type}
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Add calories failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health/shortcuts-guide", tags=["iOS Shortcuts"], response_class=HTMLResponse)
async def shortcuts_guide():
    """
    Return an HTML guide for setting up iOS Shortcuts.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>iOS Shortcuts Setup Guide</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 800px;
                margin: 40px auto;
                padding: 20px;
                line-height: 1.6;
            }
            h1 { color: #4A90E2; }
            h2 { color: #333; margin-top: 30px; }
            .endpoint {
                background: #f5f5f5;
                padding: 15px;
                border-radius: 8px;
                margin: 15px 0;
            }
            .url {
                background: #fff;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-family: monospace;
                word-break: break-all;
            }
            .example {
                background: #e8f4f8;
                padding: 10px;
                border-left: 4px solid #4A90E2;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <h1>🍎 iOS Shortcuts Setup Guide</h1>
        <p>Configure iOS Shortcuts to automatically send your Apple Health data to the Family Health Graph.</p>
        
        <h2>📍 Base URL</h2>
        <div class="url">http://your-server:8000</div>
        
        <h2>🚶 Steps Tracking</h2>
        <div class="endpoint">
            <strong>Endpoint:</strong> <code>POST /api/health/steps</code><br>
            <strong>Parameters:</strong>
            <ul>
                <li><code>person_name</code> - Your name (Leo or Jordan)</li>
                <li><code>steps</code> - Number of steps</li>
                <li><code>timestamp</code> - Current time in ISO 8601 format</li>
            </ul>
            <div class="example">
                <strong>Example:</strong><br>
                person_name: Leo<br>
                steps: 8500<br>
                timestamp: 2026-02-06T14:30:00Z
            </div>
        </div>
        
        <h2>❤️ Heart Rate</h2>
        <div class="endpoint">
            <strong>Endpoint:</strong> <code>POST /api/health/heart-rate</code><br>
            <strong>Parameters:</strong>
            <ul>
                <li><code>person_name</code> - Your name (Leo or Jordan)</li>
                <li><code>bpm</code> - Heart rate in beats per minute</li>
                <li><code>timestamp</code> - Current time in ISO 8601 format</li>
            </ul>
        </div>
        
        <h2>😴 Sleep Tracking</h2>
        <div class="endpoint">
            <strong>Endpoint:</strong> <code>POST /api/health/sleep</code><br>
            <strong>Parameters:</strong>
            <ul>
                <li><code>person_name</code> - Your name (Leo or Jordan)</li>
                <li><code>start_time</code> - Sleep start time (ISO 8601)</li>
                <li><code>end_time</code> - Sleep end time (ISO 8601)</li>
                <li><code>quality</code> - Sleep quality (good, fair, poor)</li>
            </ul>
        </div>
        
        <h2>🔥 Calorie Tracking</h2>
        <div class="endpoint">
            <strong>Endpoint:</strong> <code>POST /api/health/calories</code><br>
            <strong>Parameters:</strong>
            <ul>
                <li><code>person_name</code> - Your name (Leo or Jordan)</li>
                <li><code>calories</code> - Calorie amount</li>
                <li><code>type</code> - Type (active or dietary)</li>
                <li><code>timestamp</code> - Current time in ISO 8601 format</li>
            </ul>
        </div>
        
        <h2>📱 How to Create a Shortcut</h2>
        <ol>
            <li>Open the Shortcuts app on your iPhone</li>
            <li>Tap the + button to create a new shortcut</li>
            <li>Add "Get Health Sample" action to get data from Health app</li>
            <li>Add "Get Contents of URL" action</li>
            <li>Configure the URL with the endpoint above</li>
            <li>Set Method to POST</li>
            <li>Add form data with required parameters</li>
            <li>Save and run the shortcut</li>
        </ol>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# ---------------------------------------------------------
# LLM Chat Endpoints
# ---------------------------------------------------------

@app.get("/llm/status", tags=["LLM"])
def llm_status():
    """Check LLM engine configuration status."""
    llm = get_llm_engine()
    return {
        "configured": llm.is_configured,
        "model": llm.model,
        "vision_model": llm.vision_model,
        "message": "LLM ready" if llm.is_configured else "Set OPENAI_API_KEY environment variable to enable LLM features",
    }


@app.post("/llm/configure", tags=["LLM"])
def configure_llm(
    api_key: str = Form(None),
    base_url: str = Form(None),
    model: str = Form("gpt-4o-mini"),
    vision_model: str = Form("gpt-4o"),
):
    """Configure the LLM engine with API credentials."""
    try:
        llm = configure_llm_engine(
            api_key=api_key,
            base_url=base_url,
            model=model,
            vision_model=vision_model,
        )
        api_logger.info("LLM configured", model=model)
        return {
            "status": "success",
            "configured": llm.is_configured,
            "model": model,
            "vision_model": vision_model,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llm/chat", tags=["LLM"])
def llm_chat(
    message: str = Form(...),
    session_id: str = Form("default"),
    person_id: str = Form(None),
    use_rag: bool = Form(True),
    days: int = Form(60),
):
    """
    Chat with the health AI assistant.

    When use_rag=True, relevant health data is automatically retrieved
    and provided as context to the LLM.
    """
    try:
        context = None

        if use_rag:
            # Build graph and get RAG context
            rag = get_rag_engine()
            try:
                G = build_demo_graph(days)
                gb = get_demo_graph()
                if gb and gb.G.number_of_nodes() > 5:
                    G = gb.G

                # Ensure indexed
                if rag.vector_store.document_count == 0:
                    rag.index_graph(G, person_id)

                # Search for relevant context
                results = rag.vector_store.search(message, n_results=5)
                if results:
                    context = "\n---\n".join([r["text"] for r in results])
            except HTTPException:
                raise
            except Exception as e:
                api_logger.warning("RAG context retrieval failed", exc=str(e))

        llm = get_llm_engine()
        if not llm.is_configured:
            # Provide retrieval-only response
            if context:
                return {
                    "response": f"🔍 Related health data:\n\n{context}\n\n💡 Set OPENAI_API_KEY to enable AI analysis.",
                    "model": "none",
                    "session_id": session_id,
                    "llm_configured": False,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            return {
                "response": "LLM is not configured. Set the API key via POST /llm/configure or set the OPENAI_API_KEY environment variable.",
                "model": "none",
                "llm_configured": False,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
            }

        result = llm.chat(
            message=message,
            session_id=session_id,
            context=context,
        )
        result["llm_configured"] = True
        return result

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("LLM chat failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llm/chat/stream", tags=["LLM"])
async def llm_chat_stream(
    message: str = Form(...),
    session_id: str = Form("default"),
):
    """Stream a chat response (Server-Sent Events)."""
    llm = get_llm_engine()
    if not llm.is_configured:
        raise HTTPException(status_code=400, detail="LLM not configured")

    async def event_generator():
        async for chunk in llm.chat_stream(message, session_id):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/llm/history/{session_id}", tags=["LLM"])
def llm_get_history(session_id: str):
    """Get conversation history for a session."""
    llm = get_llm_engine()
    history = llm.get_history(session_id)
    return {"session_id": session_id, "messages": history, "count": len(history)}


@app.delete("/llm/history/{session_id}", tags=["LLM"])
def llm_clear_history(session_id: str):
    """Clear conversation history for a session."""
    llm = get_llm_engine()
    llm.clear_history(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.post("/llm/recommend/{person_id}", tags=["LLM"])
def llm_health_recommendation(
    person_id: str,
    question: str = Form(None),
    days: int = Form(60),
):
    """Generate personalized health recommendations using LLM."""
    try:
        G = build_demo_graph(days)
        context = GraphDocumentConverter.generate_person_summary(G, person_id)

        llm = get_llm_engine()
        if not llm.is_configured:
            return {
                "recommendation": context,
                "method": "graph_summary_only",
                "llm_configured": False,
                "person_id": person_id,
            }

        result = llm.generate_health_recommendation(
            person_key=person_id,
            graph_context=context,
            specific_question=question,
        )
        result["person_id"] = person_id
        return result

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Health recommendation failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# RAG Endpoints
# ---------------------------------------------------------

@app.post("/rag/index", tags=["RAG"])
def rag_index(
    person_id: str = Form(None),
    days: int = Form(60),
    use_demo_graph: bool = Form(False),
):
    """Index health graph data into the RAG vector store."""
    try:
        if use_demo_graph:
            gb = get_demo_graph()
            G = gb.G
        else:
            G = build_demo_graph(days)

        rag = get_rag_engine()
        stats = rag.index_graph(G, person_id)

        api_logger.info("RAG index completed", stats=stats)
        return {
            "status": "success",
            "documents_indexed": stats["indexed"],
            "total_in_store": stats["total_in_store"],
            "person_id": person_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("RAG indexing failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/query", tags=["RAG"])
def rag_query(
    question: str = Form(...),
    person_id: str = Form(None),
    session_id: str = Form("rag"),
    n_context: int = Form(5),
):
    """
    Ask a health question using RAG (Retrieval-Augmented Generation).

    The system retrieves relevant health data, then uses the LLM 
    to generate a contextual answer.
    """
    try:
        rag = get_rag_engine()

        # Auto-index if empty
        if rag.vector_store.document_count == 0:
            G = build_demo_graph(60)
            rag.index_graph(G, person_id)

        result = rag.query(
            question=question,
            person_id=person_id,
            n_context=n_context,
            session_id=session_id,
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("RAG query failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/search", tags=["RAG"])
def rag_search(
    query: str = Query(..., description="Search query"),
    n_results: int = Query(5, ge=1, le=20),
):
    """Search the health knowledge base without LLM generation."""
    try:
        rag = get_rag_engine()

        if rag.vector_store.document_count == 0:
            G = build_demo_graph(60)
            rag.index_graph(G)

        results = rag.vector_store.search(query, n_results=n_results)
        return {
            "query": query,
            "results": results,
            "count": len(results),
            "total_documents": rag.vector_store.document_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("RAG search failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/reindex", tags=["RAG"])
def rag_reindex(
    person_id: str = Form(None),
    days: int = Form(60),
):
    """Clear and re-index the RAG vector store."""
    try:
        G = build_demo_graph(days)
        rag = get_rag_engine()
        stats = rag.reindex(G, person_id)
        return {
            "status": "reindexed",
            "documents_indexed": stats["indexed"],
            "total_in_store": stats["total_in_store"],
        }
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("RAG reindex failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/stats", tags=["RAG"])
def rag_stats():
    """Get RAG vector store statistics."""
    rag = get_rag_engine()
    try:
        count = rag.vector_store.document_count
    except RuntimeError as e:  # optional chromadb dependency not installed
        return {"document_count": 0, "collection_name": "health_knowledge",
                "status": "unavailable", "detail": str(e)}
    return {
        "document_count": count,
        "collection_name": "health_knowledge",
        "status": "indexed" if count > 0 else "empty",
    }


# ---------------------------------------------------------
# Food OCR / Image Analysis Endpoints
# ---------------------------------------------------------

@app.post("/api/food/analyze-photo", tags=["Food Analysis"])
@app.post("/api/food/analyze-image", tags=["Food Analysis"])
async def analyze_food_image_endpoint(
    file: UploadFile = File(...),
    person_name: str = Form("unknown"),
    person_key: Optional[str] = Form(None),
    meal_date: Optional[str] = Form(None),
    add_to_graph: bool = Form(False),
):
    """
    Analyze a food image using OCR + LLM vision + color analysis.

    Pipeline:
    1. OCR text extraction (tesseract → easyocr → LLM vision)
    2. Food recognition from extracted text
    3. LLM vision analysis (if API key configured)
    4. Color-based food suggestion
    5. Nutrition lookup and aggregation
    6. Optionally add to health graph
    """
    person_name = person_key or person_name
    meal_date = meal_date or None
    try:
        image_bytes = await file.read()
        filename = file.filename or ""

        analysis = ocr_analyze_food_image(
            image_bytes=image_bytes,
            filename=filename,
            person_key=person_name.lower(),
            meal_date=meal_date,
        )

        # Optionally add to graph
        graph_stats = None
        if add_to_graph and analysis["food_count"] > 0:
            gb = get_demo_graph()
            graph_stats = add_meal_to_graph(gb, analysis)
            analysis["graph_update"] = graph_stats

        api_logger.info(
            "Food image analyzed",
            person=person_name,
            foods=analysis["food_count"],
            engine=analysis["ocr_engine"],
        )
        return analysis

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Food image analysis failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/food/analyze-text", tags=["Food Analysis"])
async def analyze_food_text_endpoint(
    text: str = Form(...),
    person_name: str = Form("unknown"),
    person_key: Optional[str] = Form(None),
    meal_date: Optional[str] = Form(None),
    add_to_graph: bool = Form(False),
):
    """
    Analyze food from text description.

    Example: "chicken breast with rice and broccoli"
    """
    person_name = person_key or person_name
    meal_date = meal_date or None
    try:
        analysis = analyze_food_text(
            text=text,
            person_key=person_name.lower(),
            meal_date=meal_date,
        )

        if add_to_graph and analysis["food_count"] > 0:
            gb = get_demo_graph()
            graph_stats = add_meal_to_graph(gb, analysis)
            analysis["graph_update"] = graph_stats

        return analysis

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Food text analysis failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/food/analyze-llm", tags=["Food Analysis"])
async def analyze_food_llm(
    file: UploadFile = File(...),
    person_name: str = Form("unknown"),
):
    """
    Analyze a food image exclusively using LLM vision (GPT-4o).
    Requires OPENAI_API_KEY to be configured.
    """
    try:
        llm = get_llm_engine()
        if not llm.is_configured:
            raise HTTPException(
                status_code=400,
                detail="LLM not configured. Set OPENAI_API_KEY or call POST /llm/configure",
            )

        image_bytes = await file.read()
        result = llm.analyze_food_image(
            image_bytes=image_bytes,
            person_key=person_name.lower(),
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("LLM food analysis failed", exc=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/food/search/{query}", tags=["Food Analysis"])
def search_food_db(query: str, limit: int = 20):
    """Search the built-in nutrition database by name or category."""
    q = query.strip().lower()
    results = [
        {"name": name, **info}
        for name, info in NUTRITION_DB.items()
        if q and (q in name or q == info.get("category"))
    ][:limit]
    return {"query": query, "count": len(results), "results": results}


# ---------------------------------------------------------
# Recommendations (ideal targets vs. current synthetic data)
# ---------------------------------------------------------

def _person_key(person: str) -> str:
    return person.split(":")[-1].lower()


@app.get("/api/recommendations/{person_key}", tags=["Recommendations"])
def get_recommendations(person_key: str, date: str = DEMO_END_DATE, days: int = 90):
    """Food, exercise, and lifestyle suggestions based on one day's gaps. Not medical advice."""
    key = _person_key(person_key)
    if key not in FAMILY_PROFILES:
        raise HTTPException(status_code=404, detail=f"Unknown person: {person_key}")
    G = build_demo_graph(days)
    return generate_recommendations(key, extract_day_features(G, key, date))


@app.get("/api/recommendations/{person_key}/compare/{date}", tags=["Recommendations"])
def compare_recommendations(person_key: str, date: str, days: int = 90):
    """Gap analysis: a day's synthetic data vs. age and activity based targets."""
    key = _person_key(person_key)
    if key not in FAMILY_PROFILES:
        raise HTTPException(status_code=404, detail=f"Unknown person: {person_key}")
    G = build_demo_graph(days)
    result = compare_current_vs_ideal(key, extract_day_features(G, key, date))
    result["date"] = date
    return result


# ---------------------------------------------------------
# Apple Health export import (kept local, git-ignored)
# ---------------------------------------------------------

IMPORT_DIR = Path(__file__).parent / "imported_graphs"
_imported_graphs: dict = {}


def _safe_key(person_key: str) -> str:
    key = "".join(ch for ch in person_key.lower() if ch.isalnum() or ch in "-_")
    if not key:
        raise HTTPException(status_code=400, detail="person_key must be alphanumeric")
    return key


def _load_imported(key: str):
    if key in _imported_graphs:
        return _imported_graphs[key]["graph"]
    path = IMPORT_DIR / f"{key}.json"
    if path.exists():
        data = json.loads(path.read_text())
        G = json_graph.node_link_graph(data["graph"], directed=True, edges="links")
        _imported_graphs[key] = {"graph": G, "meta": data["meta"]}
        return G
    raise HTTPException(status_code=404, detail=f"No imported graph for '{key}'")


@app.get("/api/apple-health/supported-types", tags=["Apple Health"])
def apple_health_supported_types():
    """HealthKit record types the streaming parser understands."""
    return {
        "quantity_types": sorted(QUANTITY_METRICS),
        "workout_types": sorted(WORKOUT_TYPES),
        "sleep": "HKCategoryTypeIdentifierSleepAnalysis",
        "total_supported": len(QUANTITY_METRICS) + len(WORKOUT_TYPES) + 1,
    }


@app.post("/api/apple-health/upload", tags=["Apple Health"])
async def apple_health_upload(
    file: UploadFile = File(...),
    person_key: str = Form(...),
    person_name: Optional[str] = Form(None),
    person_role: str = Form("child"),
    date_from: Optional[str] = Form(None),
    date_to: Optional[str] = Form(None),
    merge_with_synthetic: Optional[str] = Form(None),
):
    """
    Parse an Apple Health export.xml into a person graph. The result is stored
    only on this machine (backend/imported_graphs/, git-ignored).
    """
    key = _safe_key(person_key)
    if not (file.filename or "").lower().endswith(".xml"):
        raise HTTPException(status_code=400, detail="Please upload the export.xml file")
    raw = await file.read()
    gb = None
    if merge_with_synthetic in ("on", "true", "1"):
        gb = GraphBuilder()
        gb.G = build_demo_graph(90)
    try:
        G, stats = parse_apple_health_export(
            io.BytesIO(raw), person_key=key,
            date_from=date_from or None, date_to=date_to or None,
            person_name=person_name or None, person_role=person_role, gb=gb,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse export: {e}")
    meta = {
        "person_key": key,
        "imported_at": datetime.utcnow().isoformat(),
        "file_size_kb": round(len(raw) / 1024, 1),
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "stats": stats,
    }
    IMPORT_DIR.mkdir(exist_ok=True)
    (IMPORT_DIR / f"{key}.json").write_text(json.dumps(
        {"meta": meta, "graph": json_graph.node_link_data(G, edges="links")}, default=str))
    _imported_graphs[key] = {"graph": G, "meta": meta}
    return {"status": "success", "person_key": key, "stats": stats}


@app.get("/api/apple-health/imported", tags=["Apple Health"])
def apple_health_imported():
    """List imports stored on this machine."""
    imports = {k: v["meta"] for k, v in _imported_graphs.items()}
    if IMPORT_DIR.exists():
        for path in IMPORT_DIR.glob("*.json"):
            if path.stem not in imports:
                try:
                    imports[path.stem] = json.loads(path.read_text())["meta"]
                except Exception:
                    continue
    return {"imports": list(imports.values()), "count": len(imports)}


@app.get("/api/apple-health/graph/{person_key}", tags=["Apple Health"])
def apple_health_graph(person_key: str):
    """Node and edge counts for an imported graph."""
    G = _load_imported(_safe_key(person_key))
    node_types: dict = {}
    for _, attrs in G.nodes(data=True):
        t = attrs.get("type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1
    return {"person_key": person_key, "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(), "node_types": node_types}


@app.delete("/api/apple-health/graph/{person_key}", tags=["Apple Health"])
def apple_health_delete(person_key: str):
    """Delete an imported graph from memory and disk."""
    key = _safe_key(person_key)
    _imported_graphs.pop(key, None)
    path = IMPORT_DIR / f"{key}.json"
    existed = path.exists()
    if existed:
        path.unlink()
    return {"status": "deleted" if existed else "not_found", "person_key": key}


@app.post("/api/apple-health/analyze/{person_key}", tags=["Apple Health"])
def apple_health_analyze(person_key: str, date: Optional[str] = None):
    """Run anomaly detection, lagged correlations, and prediction on an imported graph."""
    key = _safe_key(person_key)
    G = _load_imported(key)
    person_id = f"person:{key}"
    days = sorted(n.split(":")[-1] for n, a in G.nodes(data=True)
                  if a.get("type") == "day" and n.startswith(f"day:{key}:"))
    if not days:
        raise HTTPException(status_code=404, detail="Imported graph has no days")
    if date not in days:
        date = days[-1]
    result: dict = {"person_key": key, "date": date, "days_available": len(days)}
    try:
        result["anomaly"] = _fit_anomaly(G, person_id).predict(G, person_id, date)
    except HTTPException:
        raise
    except Exception as e:
        result["anomaly"] = {"error": str(e)}
    try:
        analyzer = PatternCorrelationAnalyzer(lag_days=1)
        result["correlations"] = {"correlations": analyzer.analyze(G, person_id),
                                  "insights": analyzer.get_insights()}
    except HTTPException:
        raise
    except Exception as e:
        result["correlations"] = {"error": str(e)}
    try:
        result["prediction"] = _fit_predictor(G, person_id).predict(G, person_id, date)
    except HTTPException:
        raise
    except Exception as e:
        result["prediction"] = {"error": str(e)}
    return result
