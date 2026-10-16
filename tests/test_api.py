"""API smoke tests. Run with no saved models, so every model is fitted on demand."""

import pytest

DATE = "2025-07-10"
PEOPLE = ["person:leo", "person:mia", "person:alex", "person:jordan"]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_dashboard_page_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_graph_d3_and_stats(client):
    g = client.get("/api/graph/d3", params={"days": 14, "max_nodes": 300}).json()
    assert g["nodes"] and g["links"]
    stats = client.get("/api/graph/stats", params={"days": 14}).json()
    assert set(stats["persons"].values()) == {"Mia", "Leo", "Alex", "Jordan"}


def test_graph_insights_cover_every_member(client):
    data = client.get("/api/graph/insights", params={"days": 14}).json()
    assert len(data["person_summaries"]) == 4


@pytest.mark.parametrize("person_id", PEOPLE)
def test_daily_insights_without_saved_models(client, person_id):
    r = client.get(f"/ml/insights/daily/{person_id}/{DATE}", params={"days": 90})
    assert r.status_code == 200
    body = r.json()
    assert body["daily_features"]["meal_count"] >= 3
    assert set(body["next_day_predictions"]) == {
        "metric_energy_level", "metric_focus_score", "metric_mood"
    }
    assert 0.0 <= body["anomaly_detection"]["anomaly_score"] <= 1.0


def test_unknown_person_is_handled(client):
    r = client.get(f"/ml/insights/daily/person:nobody/{DATE}")
    assert r.status_code == 200
    assert r.json()["daily_features"] == {}
    assert client.get(f"/ml/anomalies/detect/person:nobody/{DATE}").status_code == 404


def test_invalid_days_rejected(client):
    assert client.get("/api/graph/stats", params={"days": 0}).status_code == 400


def test_train_then_use_saved_model(client):
    r = client.post("/ml/anomalies/train", params={"days": 60, "person_id": "person:mia"})
    assert r.status_code == 200
    from backend import main
    assert (main.MODEL_DIR / "anomaly_detector_mia.pkl").exists()
    r = client.get(f"/ml/anomalies/detect/person:mia/{DATE}", params={"days": 60})
    assert r.status_code == 200


def test_correlations_and_summary(client):
    corr = client.get("/ml/anomalies/correlations/person:leo", params={"days": 90}).json()
    assert corr["insights"], "expected at least one strong lagged correlation"
    summary = client.get("/ml/summary/person:leo", params={"days": 90}).json()
    assert summary["data_points"]["meals_recorded"] > 0


def test_embedding_and_link_prediction(client):
    emb = client.get(f"/ml/embeddings/daily/person:jordan/{DATE}", params={"days": 30}).json()
    assert emb["embedding_dim"] == len(emb["embedding"]) > 0
    links = client.get("/ml/link-prediction/suggest", params={"days": 30, "top_k": 3}).json()
    assert links["total_suggestions"] == 3


def test_recommendations(client):
    recs = client.get("/api/recommendations/leo", params={"date": DATE}).json()
    assert recs["profile"]["name"] == "Leo"
    cmp = client.get(f"/api/recommendations/mia/compare/{DATE}").json()
    assert "comparison" in cmp and "overall_score" in cmp
    assert client.get("/api/recommendations/nobody").status_code == 404


def test_ethics_audit_flags_minors(client):
    minors = client.get("/ethics/minors/detect", params={"days": 30}).json()
    assert minors["minors_found"] == 2
    assert client.get("/ethics/audit", params={"days": 30}).status_code == 200


def test_optional_features_degrade_gracefully(client):
    assert client.get("/llm/status").status_code == 200
    assert client.get("/rag/stats").status_code == 200


SYNTHETIC_EXPORT = """<?xml version="1.0" encoding="UTF-8"?>
<HealthData locale="en_US">
{records}
</HealthData>
"""


def _synthetic_export(days=12):
    rows = []
    for i in range(days):
        d = f"2025-06-{i + 1:02d}"
        rows.append(f'<Record type="HKQuantityTypeIdentifierStepCount" unit="count" '
                    f'startDate="{d} 09:00:00 -0700" endDate="{d} 10:00:00 -0700" value="{6000 + i * 150}"/>')
        rows.append(f'<Record type="HKQuantityTypeIdentifierRestingHeartRate" unit="count/min" '
                    f'startDate="{d} 07:00:00 -0700" endDate="{d} 07:00:00 -0700" value="{62 + i % 4}"/>')
        rows.append(f'<Workout workoutActivityType="HKWorkoutActivityTypeWalking" duration="{30 + i}" '
                    f'durationUnit="min" totalEnergyBurned="150" startDate="{d} 18:00:00 -0700" '
                    f'endDate="{d} 18:40:00 -0700"/>')
    return SYNTHETIC_EXPORT.format(records="\n".join(rows)).encode()


def test_apple_health_import_roundtrip(client):
    files = {"file": ("export.xml", _synthetic_export(), "application/xml")}
    r = client.post("/api/apple-health/upload", files=files,
                    data={"person_key": "sample", "person_role": "parent"})
    assert r.status_code == 200, r.text
    assert r.json()["stats"]["days_imported"] == 12

    listed = client.get("/api/apple-health/imported").json()
    assert [i["person_key"] for i in listed["imports"]] == ["sample"]
    assert client.get("/api/apple-health/graph/sample").json()["total_nodes"] > 12

    analysis = client.post("/api/apple-health/analyze/sample").json()
    assert analysis["days_available"] == 12

    assert client.delete("/api/apple-health/graph/sample").json()["status"] == "deleted"
    assert client.get("/api/apple-health/graph/sample").status_code == 404


def test_food_search_and_text_analysis(client):
    found = client.get("/api/food/search/rice").json()
    assert found["count"] >= 1
    r = client.post("/api/food/analyze-text",
                    data={"text": "chicken with rice and broccoli", "person_key": "leo", "meal_date": ""})
    assert r.status_code == 200
    assert r.json()["food_count"] >= 2


def test_consent_grant_and_audit_trail(client):
    r = client.post("/ethics/consent/grant", json={
        "person_id": "person:alex", "purposes": ["pattern_analysis"], "granted_by": "person:alex"})
    assert r.status_code == 200, r.text
    trail = client.get("/ethics/consent/audit-trail").json()
    # Must hit the audit-trail route, not be captured as person_id="audit-trail".
    assert not (isinstance(trail, dict) and trail.get("person_id") == "audit-trail")
