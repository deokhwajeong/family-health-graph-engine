"""
Comprehensive test script that generates large synthetic data and calls all API endpoints.

- Multi-household (3 households), multi-person (parents + children), 180 days of data
- All ML model training and inference
- Ethics/governance endpoint testing
"""

import requests
import json
import time
import sys
from datetime import date, timedelta

BASE_URL = "http://127.0.0.1:8000"
DAYS = 180  # 6 months of data

SEPARATOR = "=" * 70


def pprint(data, max_lines=30):
    """Pretty-print JSON, truncating if very long."""
    text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    lines = text.split("\n")
    if len(lines) > max_lines:
        print("\n".join(lines[:max_lines]))
        print(f"  ... ({len(lines) - max_lines} more lines)")
    else:
        print(text)


def timed_request(method, url, **kwargs):
    """Make a request and print timing info."""
    t0 = time.time()
    resp = getattr(requests, method)(url, **kwargs)
    elapsed = time.time() - t0
    return resp, elapsed


def section(title):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def test_health():
    section("1. System Health Check")
    resp, elapsed = timed_request("get", f"{BASE_URL}/health")
    print(f"  Status: {resp.status_code} ({elapsed:.2f}s)")
    pprint(resp.json())
    return resp.status_code == 200


def test_graph_stats():
    section(f"2. Graph Statistics ({DAYS} days)")
    resp, elapsed = timed_request("get", f"{BASE_URL}/ml/debug/graph-stats", params={"days": DAYS})
    print(f"  Status: {resp.status_code} ({elapsed:.2f}s)")
    pprint(resp.json())
    return resp.json()


def test_synthetic_graph():
    section(f"3. Synthetic Graph Generation ({DAYS} days)")
    resp, elapsed = timed_request("get", f"{BASE_URL}/debug/synthetic-graph", params={"days": DAYS})
    print(f"  Status: {resp.status_code} ({elapsed:.2f}s)")
    data = resp.json()
    print(f"  Nodes: {len(data.get('nodes', []))}")
    print(f"  Links: {len(data.get('links', []))}")
    # Show node type distribution
    node_types = {}
    for n in data.get("nodes", []):
        nt = n.get("type", "unknown")
        node_types[nt] = node_types.get(nt, 0) + 1
    print(f"  Node type distribution: {json.dumps(node_types, indent=4)}")
    return data


def test_train_embeddings():
    section(f"4. GraphSAGE Embedding Training ({DAYS} days)")
    resp, elapsed = timed_request("post", f"{BASE_URL}/ml/embeddings/train", params={"days": DAYS})
    print(f"  Status: {resp.status_code} ({elapsed:.2f}s)")
    pprint(resp.json())
    return resp.json()


def test_daily_embedding():
    section("5. Daily Embedding Query")
    # Test multiple dates
    start = date(2024, 11, 20)
    test_dates = [
        start + timedelta(days=10),
        start + timedelta(days=30),
        start + timedelta(days=60),
        start + timedelta(days=100),
        start + timedelta(days=150),
    ]
    results = []
    for d in test_dates:
        date_str = d.isoformat()
        resp, elapsed = timed_request(
            "get",
            f"{BASE_URL}/ml/embeddings/daily/person:mia/{date_str}",
            params={"days": DAYS},
        )
        print(f"  {date_str}: Status={resp.status_code} ({elapsed:.2f}s)", end="")
        if resp.status_code == 200:
            emb = resp.json()
            dim = emb.get("embedding_dim", "?")
            vec = emb.get("embedding", [])[:5]
            print(f" | dim={dim} | sample={[round(v,4) for v in vec]}")
            results.append(emb)
        else:
            print(f" | Error: {resp.text[:100]}")
    return results


def test_train_anomaly():
    section(f"6. Anomaly Detection Model Training ({DAYS} days)")
    resp, elapsed = timed_request("post", f"{BASE_URL}/ml/anomalies/train", params={"days": DAYS})
    print(f"  Status: {resp.status_code} ({elapsed:.2f}s)")
    pprint(resp.json())
    return resp.json()


def test_anomaly_detection():
    section("7. Anomaly Detection (Multiple Dates)")
    start = date(2024, 11, 20)
    test_dates = [start + timedelta(days=i * 20) for i in range(9)]
    anomalies_found = 0
    for d in test_dates:
        date_str = d.isoformat()
        resp, elapsed = timed_request(
            "get",
            f"{BASE_URL}/ml/anomalies/detect/person:mia/{date_str}",
            params={"days": DAYS},
        )
        if resp.status_code == 200:
            r = resp.json()
            is_anom = r.get("is_anomaly", False)
            score = r.get("anomaly_score", 0)
            reasons = r.get("reasons", [])
            flag = "🔴 ANOMALY" if is_anom else "🟢 Normal"
            print(f"  {date_str}: {flag} (score={score:.3f}) {elapsed:.2f}s")
            if reasons:
                for reason in reasons[:3]:
                    print(f"    - {reason}")
            if is_anom:
                anomalies_found += 1
        else:
            print(f"  {date_str}: Error({resp.status_code}) {elapsed:.2f}s")
    print(f"\n  Total anomalies found: {anomalies_found}/{len(test_dates)}")


def test_correlations():
    section("8. Health Behavior-Outcome Correlation Analysis")
    resp, elapsed = timed_request(
        "get",
        f"{BASE_URL}/ml/anomalies/correlations/person:mia",
        params={"days": DAYS},
    )
    print(f"  Status: {resp.status_code} ({elapsed:.2f}s)")
    if resp.status_code == 200:
        data = resp.json()
        insights = data.get("insights", [])
        print(f"  Correlation entries: {len(data.get('correlations', {}))}")
        print(f"  Key insights:")
        for ins in insights[:5]:
            print(f"    • {ins}")


def test_train_predictor():
    section(f"9. Health Metric Prediction Training ({DAYS} days)")
    resp, elapsed = timed_request("post", f"{BASE_URL}/ml/prediction/train", params={"days": DAYS})
    print(f"  Status: {resp.status_code} ({elapsed:.2f}s)")
    pprint(resp.json())
    return resp.json()


def test_predict_metrics():
    section("10. Health Metric Prediction (Future)")
    start = date(2024, 11, 20)
    # predict for dates throughout the range
    test_dates = [
        start + timedelta(days=30),
        start + timedelta(days=90),
        start + timedelta(days=150),
    ]
    for d in test_dates:
        date_str = d.isoformat()
        resp, elapsed = timed_request(
            "get",
            f"{BASE_URL}/ml/prediction/metrics/person:mia/{date_str}",
            params={"days": DAYS},
        )
        print(f"  {date_str}: Status={resp.status_code} ({elapsed:.2f}s)")
        if resp.status_code == 200:
            data = resp.json()
            preds = data.get("predictions", {})
            for k, v in preds.items():
                print(f"    {k}: {v:.2f}" if isinstance(v, (int, float)) else f"    {k}: {v}")


def test_association_rules():
    section("11. Health Association Rules Discovery")
    resp, elapsed = timed_request(
        "get",
        f"{BASE_URL}/ml/patterns/association-rules/person:mia",
        params={"days": DAYS},
    )
    print(f"  Status: {resp.status_code} ({elapsed:.2f}s)")
    if resp.status_code == 200:
        data = resp.json()
        rules = data.get("rules", [])
        print(f"  Rules discovered: {data.get('total_rules', 0)}")
        for rule in rules[:8]:
            print(f"    • {rule}")


def test_train_link_predictor():
    section(f"12. Link Prediction Model Training ({DAYS} days)")
    resp, elapsed = timed_request("post", f"{BASE_URL}/ml/link-prediction/train", params={"days": DAYS})
    print(f"  Status: {resp.status_code} ({elapsed:.2f}s)")
    pprint(resp.json())
    return resp.json()


def test_suggest_links():
    section("13. New Link Suggestions (Top 15)")
    resp, elapsed = timed_request(
        "get",
        f"{BASE_URL}/ml/link-prediction/suggest",
        params={"days": DAYS, "top_k": 15},
    )
    print(f"  Status: {resp.status_code} ({elapsed:.2f}s)")
    if resp.status_code == 200:
        data = resp.json()
        suggestions = data.get("suggested_links", [])
        print(f"  Suggested links: {data.get('total_suggestions', 0)}")
        for s in suggestions[:10]:
            print(f"    • {s}")


def test_train_classifiers():
    section(f"14. Node Classification Training ({DAYS} days)")
    resp, elapsed = timed_request("post", f"{BASE_URL}/ml/classification/train", params={"days": DAYS})
    print(f"  Status: {resp.status_code} ({elapsed:.2f}s)")
    pprint(resp.json())
    return resp.json()


def test_classify_food():
    section("15. Food Nutrition Classification")
    foods = [
        "food:greek-yogurt", "food:salmon", "food:broccoli",
        "food:rice", "food:almonds", "food:chicken",
        "food:spinach", "food:oatmeal",
    ]
    for food_id in foods:
        resp, elapsed = timed_request(
            "get",
            f"{BASE_URL}/ml/classification/food/{food_id}",
            params={"days": DAYS},
        )
        if resp.status_code == 200:
            data = resp.json()
            cls = data.get("classification", "?")
            conf = data.get("confidence", 0)
            print(f"  {food_id}: {cls} (confidence={conf:.2f}) {elapsed:.2f}s")
        else:
            print(f"  {food_id}: Error({resp.status_code}) {elapsed:.2f}s")


def test_classify_activity():
    section("16. Activity Intensity Classification")
    start = date(2024, 11, 20)
    # Try a few known activity patterns
    activities = [
        f"activity:mia:{(start + timedelta(days=1)).isoformat()}:fencing",  # Tue
        f"activity:mia:{(start + timedelta(days=3)).isoformat()}:fencing",  # Thu
        f"activity:mia:{(start + timedelta(days=5)).isoformat()}:family-walk",  # weekend
    ]
    for act_id in activities:
        resp, elapsed = timed_request(
            "get",
            f"{BASE_URL}/ml/classification/activity/{act_id}",
            params={"days": DAYS},
        )
        if resp.status_code == 200:
            data = resp.json()
            cls = data.get("classification", "?")
            conf = data.get("confidence", 0)
            print(f"  {act_id}: {cls} (confidence={conf:.2f}) {elapsed:.2f}s")
        else:
            print(f"  {act_id}: Error({resp.status_code}) {elapsed:.2f}s")


def test_daily_status():
    section("17. Daily Health Status Classification (Multiple Dates)")
    start = date(2024, 11, 20)
    test_dates = [start + timedelta(days=i * 25) for i in range(7)]
    for d in test_dates:
        date_str = d.isoformat()
        resp, elapsed = timed_request(
            "get",
            f"{BASE_URL}/ml/classification/daily-status/person:mia/{date_str}",
            params={"days": DAYS},
        )
        if resp.status_code == 200:
            data = resp.json()
            status = data.get("overall_status", data.get("classification", "?"))
            print(f"  {date_str}: {status} {elapsed:.2f}s")
        else:
            print(f"  {date_str}: Error({resp.status_code}) {elapsed:.2f}s")


def test_daily_insights():
    section("18. Comprehensive Daily Health Insights")
    start = date(2024, 11, 20)
    test_dates = [
        start + timedelta(days=15),
        start + timedelta(days=60),
        start + timedelta(days=120),
    ]
    for d in test_dates:
        date_str = d.isoformat()
        resp, elapsed = timed_request(
            "get",
            f"{BASE_URL}/ml/insights/daily/person:mia/{date_str}",
            params={"days": DAYS},
        )
        print(f"\n  📅 {date_str}: Status={resp.status_code} ({elapsed:.2f}s)")
        if resp.status_code == 200:
            data = resp.json()
            features = data.get("daily_features", {})
            status = data.get("health_status", {})
            anomaly = data.get("anomaly_detection", {})
            preds = data.get("next_day_predictions", {})
            
            print(f"    Sleep: {features.get('sleep_duration', '?')}h / Quality: {features.get('sleep_quality', '?')}")
            print(f"    Energy: {features.get('energy_level', '?')} | Focus: {features.get('focus_score', '?')} | Mood: {features.get('mood', '?')}")
            print(f"    Health Status: {status.get('overall_status', '?')}")
            print(f"    Anomaly: {'🔴 Yes' if anomaly.get('is_anomalous') else '🟢 No'} (score={anomaly.get('anomaly_score', 0):.3f})")
            if preds:
                print(f"    Next-day predictions:")
                for k, v in preds.items():
                    print(f"      {k}: {v:.2f}" if isinstance(v, (int, float)) else f"      {k}: {v}")


def test_person_summary():
    section(f"19. Person Summary ({DAYS} days)")
    resp, elapsed = timed_request(
        "get",
        f"{BASE_URL}/ml/summary/person:mia",
        params={"days": DAYS},
    )
    print(f"  Status: {resp.status_code} ({elapsed:.2f}s)")
    if resp.status_code == 200:
        data = resp.json()
        dp = data.get("data_points", {})
        print(f"  Days analyzed: {data.get('days_analyzed', '?')}")
        print(f"  Total nodes: {dp.get('total_nodes', '?')}")
        print(f"  Meals recorded: {dp.get('meals_recorded', '?')}")
        print(f"  Activities recorded: {dp.get('activities_recorded', '?')}")
        print(f"  Metrics recorded: {dp.get('metrics_recorded', '?')}")
        
        insights = data.get("behavior_insights", [])
        if insights:
            print(f"\n  Key behavioral insights:")
            for ins in insights[:5]:
                print(f"    • {ins}")
        
        rules = data.get("health_rules", [])
        if rules:
            print(f"\n  Health rules:")
            for r in rules[:5]:
                print(f"    • {r}")


def test_ethics_privacy_scan():
    section("20. PII Scan")
    resp, elapsed = timed_request("get", f"{BASE_URL}/ethics/privacy/scan", params={"days": DAYS})
    print(f"  Status: {resp.status_code} ({elapsed:.2f}s)")
    if resp.status_code == 200:
        pprint(resp.json())


def test_ethics_anonymize():
    section("21. Graph Anonymization")
    resp, elapsed = timed_request("post", f"{BASE_URL}/ethics/privacy/anonymize", params={"days": DAYS})
    print(f"  Status: {resp.status_code} ({elapsed:.2f}s)")
    if resp.status_code == 200:
        pprint(resp.json())


def test_ethics_consent():
    section("22. Consent Management (Grant & Check)")
    # Grant consent
    consent_data = {
        "person_id": "person:mia",
        "purposes": ["health_monitoring", "research"],
        "granted_by": "person:parent-guardian",
    }
    resp, elapsed = timed_request("post", f"{BASE_URL}/ethics/consent/grant", json=consent_data)
    print(f"  Consent granted: Status={resp.status_code} ({elapsed:.2f}s)")
    if resp.status_code == 200:
        pprint(resp.json())

    # Check consent
    resp, elapsed = timed_request("get", f"{BASE_URL}/ethics/consent/person:mia")
    print(f"\n  Consent status check: Status={resp.status_code} ({elapsed:.2f}s)")
    if resp.status_code == 200:
        pprint(resp.json())

    # Audit trail
    resp, elapsed = timed_request("get", f"{BASE_URL}/ethics/consent/audit-trail", params={"limit": 10})
    print(f"\n  Audit trail: Status={resp.status_code} ({elapsed:.2f}s)")
    if resp.status_code == 200:
        data = resp.json()
        if isinstance(data, list):
            print(f"  Audit entries: {len(data)}")
            for item in data[:5]:
                print(f"    • {item}")


def test_ethics_minors():
    section("23. Minor Protection")
    resp, elapsed = timed_request("get", f"{BASE_URL}/ethics/minors/detect", params={"days": DAYS})
    print(f"  Minor detection: Status={resp.status_code} ({elapsed:.2f}s)")
    if resp.status_code == 200:
        pprint(resp.json())

    resp, elapsed = timed_request("post", f"{BASE_URL}/ethics/minors/setup-guardians", params={"days": DAYS})
    print(f"\n  Guardian setup: Status={resp.status_code} ({elapsed:.2f}s)")
    if resp.status_code == 200:
        pprint(resp.json())

    resp, elapsed = timed_request(
        "get",
        f"{BASE_URL}/ethics/minors/evaluate/person:mia",
        params={"operation": "read", "purpose": "health_monitoring"},
    )
    print(f"\n  Access evaluation: Status={resp.status_code} ({elapsed:.2f}s)")
    if resp.status_code == 200:
        pprint(resp.json())


def test_ethics_transparency():
    section("24. Transparency Reporting")
    # Model card
    models = ["anomaly_detector", "link_predictor", "food_classifier"]
    for model in models:
        resp, elapsed = timed_request("get", f"{BASE_URL}/ethics/transparency/model-card/{model}")
        print(f"  Model card ({model}): Status={resp.status_code} ({elapsed:.2f}s)")
        if resp.status_code == 200:
            data = resp.json()
            print(f"    {json.dumps(data, indent=4, ensure_ascii=False)[:300]}...")

    # Compliance report
    resp, elapsed = timed_request("get", f"{BASE_URL}/ethics/transparency/compliance-report")
    print(f"\n  Compliance report: Status={resp.status_code} ({elapsed:.2f}s)")
    if resp.status_code == 200:
        pprint(resp.json())


def test_ethics_governance():
    section("25. Data Governance")
    resp, elapsed = timed_request("get", f"{BASE_URL}/ethics/governance/retention", params={"days": DAYS})
    print(f"  Retention policy check: Status={resp.status_code} ({elapsed:.2f}s)")
    if resp.status_code == 200:
        pprint(resp.json())

    resp, elapsed = timed_request("get", f"{BASE_URL}/ethics/governance/data-inventory", params={"days": DAYS})
    print(f"\n  Data inventory: Status={resp.status_code} ({elapsed:.2f}s)")
    if resp.status_code == 200:
        pprint(resp.json())


def test_full_ethics_audit():
    section(f"26. Comprehensive Ethics Audit ({DAYS} days)")
    resp, elapsed = timed_request("get", f"{BASE_URL}/ethics/audit", params={"days": DAYS})
    print(f"  Status: {resp.status_code} ({elapsed:.2f}s)")
    if resp.status_code == 200:
        pprint(resp.json(), max_lines=50)


def main():
    print("\n" + "🏥" * 35)
    print("  Family Health Graph Engine - Comprehensive Data Test")
    print(f"  Data: 180 days (6 months) | Server: {BASE_URL}")
    print("🏥" * 35)

    # Check server is running
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        if resp.status_code != 200:
            print("❌ Server is not responding properly.")
            sys.exit(1)
    except requests.ConnectionError:
        print("❌ Cannot connect to server. Please start the server first.")
        print(f"   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000")
        sys.exit(1)

    total_start = time.time()
    errors = []

    tests = [
        ("System Health", test_health),
        ("Graph Stats", test_graph_stats),
        ("Synthetic Graph", test_synthetic_graph),
        ("Embedding Training", test_train_embeddings),
        ("Daily Embedding", test_daily_embedding),
        ("Anomaly Training", test_train_anomaly),
        ("Anomaly Detection", test_anomaly_detection),
        ("Correlation Analysis", test_correlations),
        ("Prediction Training", test_train_predictor),
        ("Metric Prediction", test_predict_metrics),
        ("Association Rules", test_association_rules),
        ("Link Pred Training", test_train_link_predictor),
        ("Link Suggestion", test_suggest_links),
        ("Classification Training", test_train_classifiers),
        ("Food Classification", test_classify_food),
        ("Activity Classification", test_classify_activity),
        ("Daily Status", test_daily_status),
        ("Daily Insights", test_daily_insights),
        ("Person Summary", test_person_summary),
        ("PII Scan", test_ethics_privacy_scan),
        ("Graph Anonymize", test_ethics_anonymize),
        ("Consent Mgmt", test_ethics_consent),
        ("Minor Protection", test_ethics_minors),
        ("Transparency", test_ethics_transparency),
        ("Data Governance", test_ethics_governance),
        ("Ethics Audit", test_full_ethics_audit),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
            print(f"\n  ✅ {name} Done")
        except Exception as e:
            print(f"\n  ❌ {name} Failed: {e}")
            errors.append((name, str(e)))

    total_elapsed = time.time() - total_start

    section("📊 Final Results Summary")
    print(f"  Total tests: {len(tests)}")
    print(f"  Passed: {len(tests) - len(errors)}")
    print(f"  Failed: {len(errors)}")
    print(f"  Total time: {total_elapsed:.1f}s")
    
    if errors:
        print(f"\n  Failed tests:")
        for name, err in errors:
            print(f"    ❌ {name}: {err}")
    else:
        print(f"\n  🎉 All tests passed!")

    print()


if __name__ == "__main__":
    main()
