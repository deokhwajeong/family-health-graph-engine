
# 🏠 Family Health Graph Engine

> **Graph-based family health modeling + AI-powered dashboard**
>
> A full-stack portfolio project that models family health data as a knowledge graph and provides insights using ML/LLM/RAG

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.0+-orange)](https://networkx.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📑 Table of Contents

- [Motivation](#-motivation)
- [Key Concepts](#-key-concepts)
- [Screenshots](#-screenshots)
- [Architecture](#-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [API Endpoints](#-api-endpoints)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Roadmap](#-roadmap)

---

## 💡 Motivation

Family members' health data (sleep, diet, activity, vitals) is closely interconnected. Traditional table-based approaches struggle to capture these **relationships**.

This project:
- Models relationships between health data using a **Knowledge Graph**
- Discovers patterns through graph ML such as **GNN / Link Prediction / Anomaly Detection**
- Provides natural language health consultation and recommendations via **LLM (GPT-4o)**
- Enables graph-based semantic search with **RAG (ChromaDB)**
- Automatically analyzes food photos using **OCR + Vision AI**
- Manages privacy, consent, fairness, and governance through an **Ethics Module**

---

## 🧠 Key Concepts

```
Household ──has_member──▶ Person
Person ──has_day──▶ Day
Day ──has_meal──▶ Meal ──contains_food──▶ Food
Day ──has_sleep──▶ Sleep
Day ──has_metric──▶ Metric (heart rate, energy, mood, focus)
Day ──has_activity──▶ Activity
```

**8 Node Types**: Household · Person · Day · Meal · Food · Sleep · Metric · Activity

---

## 📸 Screenshots

### Dashboard Overview
![Dashboard Overview](docs/images/dashboard_overview.png)

### Graph Visualization (D3.js Force-Directed)
![Graph Visualization](docs/images/graph_visualization.png)

### Graph Insights & Analysis
![Graph Insights](docs/images/graph_insights.png)

### AI Chat (GPT-4o)
![AI Chat](docs/images/ai_chat.png)

### Family Analysis
![Family Analysis](docs/images/family_analysis.png)

### ML Prediction
![ML Prediction](docs/images/ml_prediction.png)

### Ethics & Governance
![Ethics Governance](docs/images/ethics_governance.png)

### API Documentation (Swagger)
![API Docs](docs/images/api_docs.png)

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend (SPA)                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────────┐  │
│  │Dashboard │ │D3 Graph  │ │AI Chat   │ │RAG Search         │  │
│  │Overview  │ │Force-Dir │ │GPT-4o    │ │Semantic Query     │  │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────────┐  │
│  │Person    │ │Anomaly   │ │ML Predict│ │Ethics Dashboard   │  │
│  │Insights  │ │Detection │ │& Classify│ │Privacy & Consent  │  │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │ REST API (67+ endpoints)
┌─────────────────────────┴───────────────────────────────────────┐
│                    FastAPI Backend                               │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Graph Engine (NetworkX DiGraph)                            │  │
│  │  • graph_builder   • synthetic_data   • graph_viz          │  │
│  │  • graph_embeddings  • pattern_analysis                    │  │
│  └────────────────────────────────────────────────────────────┘  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │ ML Pipeline  │ │ LLM Engine   │ │ RAG Engine               │ │
│  │ • Anomaly    │ │ • GPT-4o     │ │ • ChromaDB               │ │
│  │ • Prediction │ │ • Streaming  │ │ • Sentence Embeddings    │ │
│  │ • Link Pred  │ │ • Vision AI  │ │ • Semantic Search        │ │
│  │ • GNN Embed  │ │ • Sessions   │ │ • Auto-Indexing          │ │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘ │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │ Food OCR     │ │ Apple Health │ │ Ethics Module            │ │
│  │ • Tesseract  │ │ • XML Import │ │ • Privacy / Consent      │ │
│  │ • LLM Vision │ │ • Shortcuts  │ │ • Fairness / Governance  │ │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✅ Features

### Core Graph Engine
- [x] Health knowledge graph based on NetworkX DiGraph
- [x] 8 node types with relationship definitions
- [x] Synthetic data generator (14-day multi-person household)
- [x] D3.js Force-Directed graph visualization
- [x] Graph insight analysis (sleep, activity, nutrition, structure)

### ML / AI Pipeline
- [x] Node2Vec graph embeddings (128 dimensions)
- [x] Isolation Forest anomaly detection
- [x] Health metric prediction (Ridge Regression)
- [x] Link Prediction (common neighbors, Jaccard coefficient)
- [x] Food/Activity classifier (Random Forest)
- [x] Association rule mining (Apriori)

### LLM & RAG
- [x] GPT-4o / GPT-4o-mini chat engine
- [x] Streaming responses (SSE)
- [x] Session management and conversation history
- [x] Health recommendation generation (graph context-based)
- [x] ChromaDB vector store
- [x] Semantic search and auto-indexing

### Food Analysis
- [x] Tesseract OCR (Korean + English)
- [x] LLM Vision analysis (GPT-4o)
- [x] Text-based food analysis
- [x] Automatic graph node insertion

### Apple Health Integration
- [x] Apple Health XML import
- [x] iOS Shortcuts integration (steps, heart rate, sleep, calories)
- [x] Shortcuts guide page

### Ethics & Governance
- [x] PII scanning and anonymization
- [x] Consent management (grant/withdraw/audit trail)
- [x] Minor protection (guardian approval)
- [x] Fairness audit (anomaly detection bias check)
- [x] Transparency (model cards, decision logs)
- [x] Governance (data retention, deletion requests)

---

## 🛠 Tech Stack

| Category | Technologies |
|----------|-------------|
| **Backend** | Python 3.10+, FastAPI, Uvicorn, Pydantic |
| **Graph** | NetworkX, Node2Vec, Graph Embeddings |
| **ML** | PyTorch, PyG (torch-geometric), scikit-learn, NumPy, SciPy, Pandas |
| **LLM** | OpenAI API (GPT-4o), tiktoken |
| **RAG** | ChromaDB, Sentence Embeddings |
| **OCR** | Tesseract (pytesseract), Pillow |
| **Frontend** | Vanilla JS (SPA), D3.js v7, Chart.js 4.4 |
| **Visualization** | Matplotlib, Seaborn |
| **Dev Tools** | pytest, black, flake8, mypy |

---

## 📡 API Endpoints (67+)

### System
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard (SPA) |
| GET | `/health` | Health check |

### Graph Visualization
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/graph/d3` | D3.js force graph data |
| GET | `/api/graph/stats` | Graph statistics (density, degree, etc.) |
| GET | `/api/graph/insights` | Health insight analysis |

### ML & Embeddings
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ml/embeddings/train` | Train graph embeddings |
| GET | `/ml/embeddings/daily/{person_id}/{date}` | Daily embedding vector |
| POST | `/ml/anomalies/train` | Train anomaly detection model |
| GET | `/ml/anomalies/detect/{person_id}/{date}` | Detect anomalies |
| GET | `/ml/anomalies/correlations/{person_id}` | Correlation analysis |
| POST | `/ml/prediction/train` | Train prediction model |
| GET | `/ml/prediction/metrics/{person_id}/{date}` | Metric prediction |
| GET | `/ml/patterns/association-rules/{person_id}` | Association rules |
| POST | `/ml/link-prediction/train` | Train link prediction |
| GET | `/ml/link-prediction/suggest` | Link prediction suggestions |
| POST | `/ml/classification/train` | Train classification model |
| GET | `/ml/classification/food/{food_id}` | Food classification |
| GET | `/ml/classification/activity/{activity_id}` | Activity classification |
| GET | `/ml/classification/daily-status/{person_id}/{date}` | Daily status |
| GET | `/ml/insights/daily/{person_id}/{date}` | Daily comprehensive insights |
| GET | `/ml/summary/{person_id}` | Personal health summary |

### LLM (GPT-4o)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/llm/status` | LLM configuration status |
| POST | `/llm/configure` | Configure LLM (API key) |
| POST | `/llm/chat` | Chat (synchronous) |
| POST | `/llm/chat/stream` | Chat (streaming SSE) |
| GET | `/llm/history/{session_id}` | Retrieve conversation history |
| DELETE | `/llm/history/{session_id}` | Delete conversation history |
| POST | `/llm/recommend/{person_id}` | Generate health recommendations |

### RAG (Semantic Search)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/rag/index` | Index graph data |
| POST | `/rag/query` | RAG query |
| GET | `/rag/search` | Semantic search |
| POST | `/rag/reindex` | Reindex data |
| GET | `/rag/stats` | RAG statistics |

### Food Analysis
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/food/analyze-image` | Food photo OCR analysis |
| POST | `/api/food/analyze-text` | Text-based food analysis |
| POST | `/api/food/analyze-llm` | LLM vision food analysis |

### Apple Health / iOS Shortcuts
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/health/import-xml` | Import Apple Health XML |
| POST | `/api/health/steps` | Record steps |
| POST | `/api/health/heart-rate` | Record heart rate |
| POST | `/api/health/sleep` | Record sleep |
| POST | `/api/health/calories` | Record calories |
| GET | `/api/health/shortcuts-guide` | iOS Shortcuts guide |

### Ethics & Governance (17 endpoints)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/ethics/privacy/scan` | PII scan |
| POST | `/ethics/privacy/anonymize` | Data anonymization |
| POST | `/ethics/consent/grant` | Grant consent |
| POST | `/ethics/consent/withdraw` | Withdraw consent |
| GET | `/ethics/consent/{person_id}` | Consent status |
| GET | `/ethics/consent/audit-trail` | Audit trail |
| GET | `/ethics/minors/detect` | Detect minors |
| POST | `/ethics/minors/setup-guardians` | Set up guardians |
| GET | `/ethics/minors/evaluate/{person_id}` | Evaluate minor |
| POST | `/ethics/fairness/audit-anomalies` | Fairness audit |
| GET | `/ethics/fairness/history` | Fairness history |
| GET | `/ethics/transparency/model-card/{model_name}` | Model card |
| GET | `/ethics/transparency/compliance-report` | Compliance report |
| GET | `/ethics/transparency/decisions` | Decision log |
| GET | `/ethics/governance/retention` | Data retention policy |
| POST | `/ethics/governance/deletion-request` | Deletion request |
| GET | `/ethics/governance/data-inventory` | Data inventory |
| GET | `/ethics/audit` | Comprehensive ethics audit |

### Debug
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/ml/debug/graph-stats` | Graph ML statistics |
| GET | `/debug/example-graph` | Example graph |
| GET | `/debug/synthetic-graph` | Synthetic graph |
| GET | `/debug/example-graph/svg` | Example graph SVG |
| GET | `/debug/synthetic-graph/svg` | Synthetic graph SVG |
| GET | `/debug/synthetic-graph/html` | Synthetic graph HTML |

> 💡 Full API documentation: After starting the server, visit `http://localhost:8000/docs` (Swagger UI)

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/deokhwajeong/family-health-graph-engine.git
cd family-health-graph-engine

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r backend/requirements.txt
```

### 2. Environment Variables (Optional)

To use LLM/RAG features, create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-your-api-key
OPENAI_MODEL=gpt-4o           # Default: gpt-4o-mini
```

### 3. Run Server

```bash
cd family-health-graph-engine
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open Dashboard

```
http://localhost:8000
```

### 5. API Documentation

```
http://localhost:8000/docs      # Swagger UI
http://localhost:8000/redoc     # ReDoc
```

---

## 📁 Project Structure

```
family-health-graph-engine/
├── backend/
│   ├── main.py                    # FastAPI app (67+ endpoints)
│   ├── requirements.txt           # Python dependencies
│   ├── train_models.py            # Model training script
│   ├── app/
│   │   ├── graph_builder.py       # NetworkX graph construction
│   │   ├── synthetic_data.py      # Synthetic health data generator
│   │   ├── graph_viz.py           # Graph visualization (Matplotlib)
│   │   ├── graph_embeddings.py    # Node2Vec embeddings
│   │   ├── anomaly_detection.py   # Isolation Forest anomaly detection
│   │   ├── pattern_analysis.py    # Association rules & patterns
│   │   ├── link_prediction.py     # Link prediction algorithms
│   │   ├── node_classifier.py     # Food/Activity classification
│   │   ├── health_recommendations.py  # Rule-based recommendations
│   │   ├── llm_engine.py          # OpenAI GPT-4o chat engine
│   │   ├── rag_engine.py          # ChromaDB RAG pipeline
│   │   ├── food_ocr.py            # Tesseract + LLM Vision OCR
│   │   ├── apple_health.py        # Apple Health XML parser
│   │   ├── apple_health_parser.py # Apple Health data mapper
│   │   ├── ethics.py              # Ethics & governance module
│   │   ├── schemas.py             # Pydantic models
│   │   ├── config.py              # Configuration
│   │   └── logger.py              # Logging setup
│   ├── static/
│   │   └── index.html             # SPA Dashboard (3300+ lines)
│   ├── models/                    # Trained model files
│   ├── uploads/                   # Uploaded health data
│   └── imported_graphs/           # Imported graph JSON files
├── frontend/                      # Alternative frontend
│   ├── index.html
│   └── static/
│       ├── css/styles.css
│       └── js/
│           ├── app.js
│           ├── dashboard.js
│           └── apple-health.js
├── docs/
│   ├── graph-schema.md            # Graph schema documentation
│   └── images/                    # Screenshots & diagrams
└── README.md
```

---

## 🗺 Roadmap

### Phase 1 — Core Graph Engine ✅
- [x] Health knowledge graph modeling
- [x] Synthetic data generation
- [x] Graph visualization (SVG, D3.js)
- [x] Basic CRUD API

### Phase 2 — ML & AI ✅
- [x] Graph embeddings (Node2Vec)
- [x] Anomaly detection (Isolation Forest)
- [x] Health metric prediction
- [x] LLM chat engine (GPT-4o)
- [x] RAG semantic search (ChromaDB)
- [x] Food photo OCR analysis
- [x] Ethics & Governance module
- [x] Apple Health integration

### Phase 3 — Production Ready 🔜
- [ ] User authentication (OAuth2 / JWT)
- [ ] PostgreSQL / Neo4j persistence
- [ ] TV UI ambient display
- [ ] Mobile PWA optimization
- [ ] Multi-household support
- [ ] CI/CD pipeline
- [ ] Docker Compose deployment

---

## 📄 License

This project is licensed under the MIT License.

---

<p align="center">
  <sub>Built with ❤️ for family health insights</sub>
</p>
