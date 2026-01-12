echo "- Jan 11: API stabilization and rebase cleanup" >> README.md
git add README.md
git commit -m "Chore: daily progress note"
git push


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

- [x] Basic nutrition analysis API (FastAPI)  
- [x] Simple demo UI (fiber scoring)  
- [ ] Graph schema  
- [ ] Synthetic household data generator  
- [ ] Graph engine + embeddings  
- [ ] Insight APIs  
- [ ] Web dashboard  
- [ ] TV layout  
- [ ] Final polish + slide-ready visuals  

---

## 6. Roadmap

### Week 1 — Modeling
Graph schema, synthetic data, ingestion → graph builder

### Week 2 — Graph Engine
Graph metrics → embeddings → insight rules → APIs

### Week 3 — Dashboard
Household view → member view → interactions

### Week 4 — Polish
TV UI, documentation, diagrams, and screenshots

---

## 7. Tech Stack

Python, FastAPI  
NetworkX, PyTorch Geometric (optional)  
Next.js, TypeScript  
GitHub Codespaces

---

## 8. Background Integration

This project reflects the full arc of my trajectory:

- **Bio & health science** (academic training)  
- **Graph ML** (Stanford XCS224W, perfect score)  
- **Embedded/Smart TV engineering** (10+ yrs at LG)  
- **Technical PM** for global data-driven products  

The result is a system that turns household health data into a **graph of interpretable interactions**, displayed through a **consumer-grade UX** suitable for real families.
