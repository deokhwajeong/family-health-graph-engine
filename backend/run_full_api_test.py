#!/usr/bin/env python3
"""
Comprehensive API test — exercises all ML + Ethics endpoints
against the 4-person Demo family graph (90 days, ~3400 nodes).
"""

import requests, json, sys, time

BASE = "http://localhost:8000"
DAYS = 90
PEOPLE = ["person:leo", "person:alex", "person:jordan", "person:mia"]
SAMPLE_DATE = "2025-07-15"  # mid-range date within 90-day window

ok = 0
fail = 0

def test(label, method, url, **kwargs):
    global ok, fail
    try:
        t0 = time.time()
        if method == "GET":
            r = requests.get(url, params=kwargs.get("params"), timeout=120)
        else:
            r = requests.post(url, params=kwargs.get("params"), json=kwargs.get("json_body"), timeout=120)
        elapsed = time.time() - t0
        if r.status_code == 200:
            data = r.json()
            summary = json.dumps(data, indent=2, default=str)[:300]
            print(f"  ✅ {label}  ({elapsed:.1f}s)")
            print(f"     {summary}\n")
            ok += 1
        else:
            print(f"  ❌ {label}  → {r.status_code}: {r.text[:200]}\n")
            fail += 1
    except Exception as e:
        print(f"  ❌ {label}  → {e}\n")
        fail += 1

print("=" * 70)
print("  FAMILY HEALTH GRAPH ENGINE — FULL API TEST")
print(f"  4 people × {DAYS} days  |  ~3400 nodes  |  ~7000 edges")
print("=" * 70)

# ── 1. Graph Stats ──
print("\n🔧 [1/10] Graph Statistics")
test("Graph Stats", "GET", f"{BASE}/ml/debug/graph-stats", params={"days": DAYS})

# ── 2. Train ALL models ──
print("🧠 [2/10] Training ML Models")
test("Train Embeddings",      "POST", f"{BASE}/ml/embeddings/train",       params={"days": DAYS})
test("Train Anomaly Detector (Alex)", "POST", f"{BASE}/ml/anomalies/train", params={"days": DAYS, "person_id": "person:alex"})
test("Train Anomaly Detector (Jordan)", "POST", f"{BASE}/ml/anomalies/train", params={"days": DAYS, "person_id": "person:jordan"})
test("Train Predictor (Mia)","POST", f"{BASE}/ml/prediction/train",      params={"days": DAYS, "person_id": "person:mia"})
test("Train Link Predictor",  "POST", f"{BASE}/ml/link-prediction/train",  params={"days": DAYS})
test("Train Classifiers",     "POST", f"{BASE}/ml/classification/train",   params={"days": DAYS})

# ── 3. Embeddings per person ──
print("📊 [3/10] Daily Embeddings")
for pid in PEOPLE:
    test(f"Embedding {pid}", "GET", f"{BASE}/ml/embeddings/daily/{pid}/{SAMPLE_DATE}", params={"days": DAYS})

# ── 4. Anomaly Detection per person ──
print("🔍 [4/10] Anomaly Detection")
for pid in PEOPLE:
    test(f"Anomaly {pid}", "GET", f"{BASE}/ml/anomalies/detect/{pid}/{SAMPLE_DATE}", params={"days": DAYS})

# ── 5. Correlations per person ──
print("🔗 [5/10] Behavior Correlations")
for pid in PEOPLE:
    test(f"Correlations {pid}", "GET", f"{BASE}/ml/anomalies/correlations/{pid}", params={"days": DAYS})

# ── 6. Metric Prediction ──
print("📈 [6/10] Metric Prediction")
for pid in PEOPLE:
    test(f"Predict {pid}", "GET", f"{BASE}/ml/prediction/metrics/{pid}/{SAMPLE_DATE}", params={"days": DAYS})

# ── 7. Association Rules ──
print("📋 [7/10] Association Rules")
for pid in PEOPLE:
    test(f"Rules {pid}", "GET", f"{BASE}/ml/patterns/association-rules/{pid}", params={"days": DAYS})

# ── 8. Link Prediction & Classification ──
print("🔮 [8/10] Link Prediction & Node Classification")
test("Link Suggestions", "GET", f"{BASE}/ml/link-prediction/suggest", params={"days": DAYS, "top_k": 10})
test("Classify food:ice-cream",  "GET", f"{BASE}/ml/classification/food/food:ice-cream",  params={"days": DAYS})
test("Classify food:broccoli",   "GET", f"{BASE}/ml/classification/food/food:broccoli",   params={"days": DAYS})
test("Classify food:beef-steak", "GET", f"{BASE}/ml/classification/food/food:beef-steak", params={"days": DAYS})

# ── 9. Daily Insights + Summary ──
print("🏥 [9/10] Daily Insights & Person Summary")
for pid in PEOPLE:
    test(f"Daily Insight {pid}", "GET", f"{BASE}/ml/insights/daily/{pid}/{SAMPLE_DATE}", params={"days": DAYS})
test("Summary Alex",     "GET", f"{BASE}/ml/summary/person:alex",     params={"days": DAYS})
test("Summary Jordan", "GET", f"{BASE}/ml/summary/person:jordan", params={"days": DAYS})

# ── 10. Ethics Endpoints ──
print("⚖️  [10/10] Ethics & Governance")
test("PII Scan",           "GET",  f"{BASE}/ethics/privacy/scan",            params={"days": DAYS})
test("Anonymize",          "POST", f"{BASE}/ethics/privacy/anonymize",       params={"days": DAYS})
test("Detect Minors",      "GET",  f"{BASE}/ethics/minors/detect",           params={"days": DAYS})
test("Setup Guardians",    "POST", f"{BASE}/ethics/minors/setup-guardians",  params={"days": DAYS})
test("Retention Policy",   "GET",  f"{BASE}/ethics/governance/retention",    params={"days": DAYS})
test("Data Inventory",     "GET",  f"{BASE}/ethics/governance/data-inventory", params={"days": DAYS})
test("Full Ethics Audit",  "GET",  f"{BASE}/ethics/audit",                   params={"days": DAYS})
test("Compliance Report",  "GET",  f"{BASE}/ethics/transparency/compliance-report")
test("Model Card (anomaly_detector)", "GET", f"{BASE}/ethics/transparency/model-card/anomaly_detector")

# ── Summary ──
print("=" * 70)
total = ok + fail
print(f"  RESULTS: {ok}/{total} passed, {fail} failed")
if fail == 0:
    print("  🎉 ALL TESTS PASSED!")
else:
    print(f"  ⚠️  {fail} test(s) failed — check output above")
print("=" * 70)

sys.exit(1 if fail else 0)
