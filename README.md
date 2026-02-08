
# 🏠 Family Health Graph Engine

> **Graph-based family health modeling + AI-powered dashboard**
>
> 가족 건강 데이터를 지식 그래프로 모델링하고, ML/LLM/RAG를 활용하여 인사이트를 제공하는 풀스택 포트폴리오 프로젝트

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

가족 구성원의 건강 데이터(수면, 식단, 활동, 바이탈)는 서로 밀접하게 연결되어 있습니다. 기존의 테이블 기반 접근으로는 이러한 **관계성**을 포착하기 어렵습니다.

이 프로젝트는:
- **Knowledge Graph**로 건강 데이터 간 관계를 모델링
- **GNN / Link Prediction / Anomaly Detection** 등 그래프 ML로 패턴 발견
- **LLM (GPT-4o)** 으로 자연어 건강 상담 및 추천
- **RAG (ChromaDB)** 로 그래프 기반 시맨틱 검색
- **OCR + Vision AI**로 음식 사진 자동 분석
- **Ethics Module**로 개인정보 보호·동의·공정성·거버넌스 관리

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

**8가지 노드 타입**: Household · Person · Day · Meal · Food · Sleep · Metric · Activity

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
- [x] NetworkX DiGraph 기반 건강 지식 그래프
- [x] 8가지 노드 타입, 관계 정의
- [x] 합성 데이터 생성기 (14일 다인 가구)
- [x] D3.js Force-Directed 그래프 시각화
- [x] 그래프 인사이트 분석 (수면, 활동, 영양, 구조)

### ML / AI Pipeline
- [x] Node2Vec 그래프 임베딩 (128차원)
- [x] Isolation Forest 이상 탐지
- [x] 건강 지표 예측 (Ridge Regression)
- [x] Link Prediction (공통 이웃, Jaccard 계수)
- [x] 음식/활동 분류기 (Random Forest)
- [x] 연관 규칙 분석 (Apriori)

### LLM & RAG
- [x] GPT-4o / GPT-4o-mini 채팅 엔진
- [x] 스트리밍 응답 (SSE)
- [x] 세션 관리 및 대화 이력
- [x] 건강 추천 생성 (그래프 컨텍스트 기반)
- [x] ChromaDB 벡터 스토어
- [x] 시맨틱 검색 및 자동 인덱싱

### Food Analysis
- [x] Tesseract OCR (한국어 + 영어)
- [x] LLM Vision 분석 (GPT-4o)
- [x] 텍스트 기반 음식 분석
- [x] 자동 그래프 노드 추가

### Apple Health Integration
- [x] Apple Health XML 가져오기
- [x] iOS Shortcuts 연동 (걸음수, 심박수, 수면, 칼로리)
- [x] Shortcuts 가이드 페이지

### Ethics & Governance
- [x] 개인정보 스캔 및 익명화
- [x] 동의 관리 (부여/철회/감사 추적)
- [x] 미성년자 보호 (후견인 승인)
- [x] 공정성 감사 (이상탐지 편향 검사)
- [x] 투명성 (모델 카드, 의사결정 로그)
- [x] 거버넌스 (데이터 보존, 삭제 요청)

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
| GET | `/api/graph/d3` | D3.js 포스 그래프 데이터 |
| GET | `/api/graph/stats` | 그래프 통계 (밀도, 차수 등) |
| GET | `/api/graph/insights` | 건강 인사이트 분석 |

### ML & Embeddings
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ml/embeddings/train` | 그래프 임베딩 학습 |
| GET | `/ml/embeddings/daily/{person_id}/{date}` | 일별 임베딩 벡터 |
| POST | `/ml/anomalies/train` | 이상탐지 모델 학습 |
| GET | `/ml/anomalies/detect/{person_id}/{date}` | 이상 탐지 |
| GET | `/ml/anomalies/correlations/{person_id}` | 상관관계 분석 |
| POST | `/ml/prediction/train` | 예측 모델 학습 |
| GET | `/ml/prediction/metrics/{person_id}/{date}` | 지표 예측 |
| GET | `/ml/patterns/association-rules/{person_id}` | 연관 규칙 |
| POST | `/ml/link-prediction/train` | 링크 예측 학습 |
| GET | `/ml/link-prediction/suggest` | 링크 예측 제안 |
| POST | `/ml/classification/train` | 분류 모델 학습 |
| GET | `/ml/classification/food/{food_id}` | 음식 분류 |
| GET | `/ml/classification/activity/{activity_id}` | 활동 분류 |
| GET | `/ml/classification/daily-status/{person_id}/{date}` | 일일 상태 |
| GET | `/ml/insights/daily/{person_id}/{date}` | 일일 종합 인사이트 |
| GET | `/ml/summary/{person_id}` | 개인 건강 요약 |

### LLM (GPT-4o)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/llm/status` | LLM 설정 상태 |
| POST | `/llm/configure` | LLM 설정 (API 키) |
| POST | `/llm/chat` | 채팅 (동기) |
| POST | `/llm/chat/stream` | 채팅 (스트리밍 SSE) |
| GET | `/llm/history/{session_id}` | 대화 이력 조회 |
| DELETE | `/llm/history/{session_id}` | 대화 이력 삭제 |
| POST | `/llm/recommend/{person_id}` | 건강 추천 생성 |

### RAG (Semantic Search)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/rag/index` | 그래프 데이터 인덱싱 |
| POST | `/rag/query` | RAG 질의 |
| GET | `/rag/search` | 시맨틱 검색 |
| POST | `/rag/reindex` | 데이터 재인덱싱 |
| GET | `/rag/stats` | RAG 통계 |

### Food Analysis
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/food/analyze-image` | 음식 사진 OCR 분석 |
| POST | `/api/food/analyze-text` | 텍스트 음식 분석 |
| POST | `/api/food/analyze-llm` | LLM 비전 음식 분석 |

### Apple Health / iOS Shortcuts
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/health/import-xml` | Apple Health XML 가져오기 |
| POST | `/api/health/steps` | 걸음수 기록 |
| POST | `/api/health/heart-rate` | 심박수 기록 |
| POST | `/api/health/sleep` | 수면 기록 |
| POST | `/api/health/calories` | 칼로리 기록 |
| GET | `/api/health/shortcuts-guide` | iOS Shortcuts 가이드 |

### Ethics & Governance (17 endpoints)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/ethics/privacy/scan` | 개인정보 스캔 |
| POST | `/ethics/privacy/anonymize` | 데이터 익명화 |
| POST | `/ethics/consent/grant` | 동의 부여 |
| POST | `/ethics/consent/withdraw` | 동의 철회 |
| GET | `/ethics/consent/{person_id}` | 동의 상태 조회 |
| GET | `/ethics/consent/audit-trail` | 감사 추적 |
| GET | `/ethics/minors/detect` | 미성년자 탐지 |
| POST | `/ethics/minors/setup-guardians` | 후견인 설정 |
| GET | `/ethics/minors/evaluate/{person_id}` | 미성년자 평가 |
| POST | `/ethics/fairness/audit-anomalies` | 공정성 감사 |
| GET | `/ethics/fairness/history` | 공정성 이력 |
| GET | `/ethics/transparency/model-card/{model_name}` | 모델 카드 |
| GET | `/ethics/transparency/compliance-report` | 컴플라이언스 리포트 |
| GET | `/ethics/transparency/decisions` | 의사결정 로그 |
| GET | `/ethics/governance/retention` | 데이터 보존 정책 |
| POST | `/ethics/governance/deletion-request` | 삭제 요청 |
| GET | `/ethics/governance/data-inventory` | 데이터 인벤토리 |
| GET | `/ethics/audit` | 종합 윤리 감사 |

### Debug
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/ml/debug/graph-stats` | 그래프 ML 통계 |
| GET | `/debug/example-graph` | 예제 그래프 |
| GET | `/debug/synthetic-graph` | 합성 그래프 |
| GET | `/debug/example-graph/svg` | 예제 그래프 SVG |
| GET | `/debug/synthetic-graph/svg` | 합성 그래프 SVG |
| GET | `/debug/synthetic-graph/html` | 합성 그래프 HTML |

> 💡 전체 API 문서: 서버 실행 후 `http://localhost:8000/docs` (Swagger UI)

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

LLM/RAG 기능을 사용하려면 `.env` 파일을 프로젝트 루트에 생성:

```env
OPENAI_API_KEY=sk-your-api-key
OPENAI_MODEL=gpt-4o           # 기본값: gpt-4o-mini
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
- [x] 건강 지식 그래프 모델링
- [x] 합성 데이터 생성
- [x] 그래프 시각화 (SVG, D3.js)
- [x] 기본 CRUD API

### Phase 2 — ML & AI ✅
- [x] 그래프 임베딩 (Node2Vec)
- [x] 이상 탐지 (Isolation Forest)
- [x] 건강 지표 예측
- [x] LLM 채팅 엔진 (GPT-4o)
- [x] RAG 시맨틱 검색 (ChromaDB)
- [x] 음식 사진 OCR 분석
- [x] Ethics & Governance 모듈
- [x] Apple Health 연동

### Phase 3 — Production Ready 🔜
- [ ] 사용자 인증 (OAuth2 / JWT)
- [ ] PostgreSQL / Neo4j 영속성
- [ ] TV UI 앰비언트 디스플레이
- [ ] Mobile PWA 최적화
- [ ] Multi-household 지원
- [ ] CI/CD 파이프라인
- [ ] Docker Compose 배포

---

## 📄 License

This project is licensed under the MIT License.

---

<p align="center">
  <sub>Built with ❤️ for family health insights</sub>
</p>
