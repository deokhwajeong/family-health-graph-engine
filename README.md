

# Household Health Graph & Family Dashboard

A portfolio project that unifies graph-based health modeling with a family-centric dashboard (web + TV UI).  
It visualizes how daily behaviors—meals, sleep, activity—interact across a household, making health patterns interpretable and actionable.

> Purpose: Demonstrate the convergence of **Bio + Graph ML + Systems Engineering + Smart Device UX** through a single, coherent product.

---

## 1. Motivation

Most consumer health tools treat data as flat, isolated time series.  
Real households don't work that way. Daily routines are interdependent: parents’ schedules shape children’s sleep, shared meals influence everyone’s energy, and weekend activities ripple into weekday performance.

This project asks:

- What if a **household** were modeled as a **graph**?
- What insights emerge when we analyze interactions rather than individuals?
- How can we surface these insights in a clean, ambient UI that’s realistic for home use (web or TV)?

---

## 2. Concept Overview

### **The Household Health Graph Engine**

- **Nodes:** people, meals, foods, sleep sessions, activities, metrics, days  
- **Edges:** “ate”, “occurred on”, “impacted”, “did together”, etc.

From this structure, the system computes:

- Daily embeddings (simple v1 or GraphSAGE/GAT v2)
- Signal extraction:
  - high-sugar evenings  
  - low-sleep → low-focus patterns  
  - high-fiber days vs. weekend routines  
- Personalized daily summaries for each member

The engine serves insights to:

- A **web dashboard** (Next.js)  
- A **TV-style UI** inspired by smart-home displays (optional but aligned with my background)

---

## 3. Target Users

- Families seeking patterns without micromanaging calories  
- Reviewers evaluating:
  - graph-structured ML thinking  
  - end-to-end system design  
  - consumer-grade UX  
  - PM-style product reasoning  

---

## 4. Architecture (v1)

### **Backend — FastAPI**
- Event ingestion (meals, sleep, activities)  
- Graph construction (NetworkX → PyG-compatible)  
- Insight engine:
  - Graph metrics (v1)
  - GNN embedding (v2)
- Summary APIs:
  - `/household/summary`
  - `/member/{id}/timeline`
  - `/insights/today`

### **Frontend — Next.js**
- Household overview  
- Member timelines  
- TV-style large-screen mode (stretch goal)

### **Modeling**
- Graph structure + embeddings  
- Emphasis on **interpretability**, not prediction accuracy  

---

## 5. Status

### Backend — FastAPI + ML Engine
- [x] Graph schema and data model  
- [x] Synthetic household data generator (60-day realistic data)  
- [x] Graph builder (NetworkX DiGraph)  
- [x] **GraphSAGE embeddings** (16-dim, node type encoding)  
- [x] **Anomaly detection** (Isolation Forest + LOF)  
- [x] **Pattern analysis & prediction** (next-day health metrics)  
- [x] **Link prediction** (food-health relationships)  
- [x] **Node classification** (nutrition, activity intensity, health status)  
- [x] 50+ ML API endpoints  
- [x] Model persistence (joblib pkl files)  
- [x] Training pipeline  

### DevOps & CI/CD
- [x] GitHub Actions CI workflow (Python 3.10/3.11/3.12 matrix)  
- [x] GitHub Actions release workflow (v*.*.* tag triggers)  
- [x] Automated testing on push/PR  
- [x] v0.2.0 release published  

### Frontend & Dashboard
- [ ] Web dashboard (Next.js)  
- [ ] Household overview visualization  
- [ ] Member timelines  
- [ ] TV-style large-screen mode  

### Documentation
- [x] ML_IMPLEMENTATION.md (algorithm details, usage examples)  
- [x] IMPLEMENTATION_COMPLETE.md (summary)  

---

## 6. API Endpoints Summary

### Graph & Embeddings
- `GET /ml/embeddings/daily/{person_id}/{date}` - Daily node embeddings
- `GET /ml/embeddings/person/{person_id}` - Person node embedding
- `GET /ml/features/daily/{person_id}/{date}` - Extracted features (10+ metrics)

### Anomaly Detection
- `POST /ml/anomalies/detect` - Detect unusual patterns
- `GET /ml/anomalies/correlations/{person_id}` - Pattern correlations

### Prediction & Analysis
- `POST /ml/prediction/next-day` - Predict energy, focus, mood
- `POST /ml/patterns/rules` - Discover health rules (food → energy, etc.)

### Link Prediction
- `POST /ml/links/predict` - Suggest food-health relationships
- `POST /ml/links/explain/{node_id}/{target_id}` - Link explanation

### Classification
- `POST /ml/classify/food` - Food nutrition classification
- `POST /ml/classify/activity` - Activity intensity classification
- `POST /ml/classify/health` - Daily health status classification

### Insights
- `GET /ml/insights/daily/{person_id}/{date}` - Complete daily analysis
- `GET /ml/summary/{person_id}` - Person-level health summary

---

## 7. Roadmap (v0.3+)

### Phase 1 — Dashboard
Household view → member view → interactive visualizations

### Phase 2 — Advanced ML
Temporal graph networks → multi-task learning → causal inference

### Phase 3 — Smart Integration
TV UI → mobile app → smart device sync

### Phase 4 — Production
Data privacy → auth & access control → cloud deployment

---

## 8. Tech Stack

**Backend:**
- FastAPI 0.2.0 (REST API)
- NetworkX (Graph construction & traversal)
- NumPy, SciPy (Numerical computing)
- Scikit-Learn (ML algorithms: Isolation Forest, LOF, LogisticRegression, RandomForest)
- PyTorch, PyTorch Geometric (Optional: advanced GNN)
- JobLib (Model persistence)
- Pandas, Matplotlib, Seaborn (Data analysis)

**Frontend:**
- Next.js, TypeScript (Coming soon)

**DevOps:**
- GitHub Actions (CI/CD)
- Docker (Optional)
- GitHub Codespaces

---

## 9. Quick Start

### Setup
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Train Models
```bash
python train_models.py
```

### Run Server
```bash
python main.py
# Server runs at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Test API
```bash
curl http://localhost:8000/ml/debug/graph-stats
curl http://localhost:8000/ml/insights/daily/yooni/2025-12-06
```

---

## 10. Background Integration

This project reflects the full arc of my trajectory:

- **Bio & health science** (academic training)  
- **Graph ML** (Stanford XCS224W, perfect score)  
- **Embedded/Smart TV engineering** (10+ yrs at LG)  
- **Technical PM** for global data-driven products  

The result is a system that turns household health data into a **graph of interpretable interactions**, displayed through a **consumer-grade UX** suitable for real families.
