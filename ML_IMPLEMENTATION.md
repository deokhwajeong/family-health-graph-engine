# Household Health Graph Engine - ML Implementation Guide

## 개요

이 프로젝트는 가족의 건강 패턴을 그래프 기반으로 분석하고 머신러닝으로 인사이트를 도출하는 시스템입니다.

**Data Add → Learn → Predict** 구조로 설계되었습니다:
1. **Add**: 매일의 식사, 수면, 활동 데이터를 그래프 노드로 추가
2. **Learn**: 데이터로부터 패턴을 학습
3. **Predict**: 미래의 건강 상태를 예측하고 이상을 탐지

---

## 구현된 ML 기능

### 1. **Graph Embedding (GraphSAGE)**
```python
GET /ml/embeddings/daily/{person_id}/{date}
```
- **목적**: 매일의 건강 행동을 벡터 임베딩으로 표현
- **알고리즘**: 이웃 노드 정보를 2개 레이어로 집계하는 GraphSAGE 변형
- **특징**:
  - 식사, 수면, 활동 정보를 종합한 일일 임베딩
  - 노드 타입별 특정 피처 추출 (weekday, sleep quality, calories 등)
  - 의미있는 건강 패턴 벡터 표현

**사용 예**:
```bash
curl http://localhost:8000/ml/embeddings/daily/person:yooni/2024-12-15
```

---

### 2. **Anomaly Detection (Isolation Forest + LOF)**
```python
GET /ml/anomalies/detect/{person_id}/{date}
POST /ml/anomalies/train
```
- **목적**: 개인의 평소 패턴과 다른 비정상 날을 탐지
- **알고리즘**: 
  - **Isolation Forest**: 전역 아웃라이어 탐지 (극단적인 행동)
  - **Local Outlier Factor**: 국소 밀도 기반 이상 탐지
- **탐지 지표**:
  - 매우 낮은 수면 (< 6시간)
  - 극단적인 칼로리 섭취 (< 1000 또는 > 4000 kcal)
  - 높은 설탕 섭취 (> 100g)
  - 활동과 영양의 불균형

**반환 예**:
```json
{
  "anomaly_score": 0.75,
  "is_anomaly": true,
  "reasons": [
    "Very low sleep duration",
    "High sugar consumption"
  ]
}
```

**상관관계 분석**:
```python
GET /ml/anomalies/correlations/{person_id}
```
- 수면 질과 집중력의 관계
- 단백질 섭취와 에너지 레벨의 관계
- 활동 빈도와 수면 시간의 관계

---

### 3. **Pattern Analysis & Prediction**

#### 3.1 메트릭 예측
```python
GET /ml/prediction/metrics/{person_id}/{date}
```
- **목적**: 현재 일의 행동으로부터 다음날 에너지, 집중력, 기분 예측
- **알고리즘**: 다변량 선형회귀
- **입력 피처**:
  - 총 칼로리, 단백질, 섬유소, 설탕
  - 수면 시간, 수면 질
  - 활동 횟수, 활동 강도, 소모 칼로리
- **출력**: 
  - 예상 에너지 레벨 (0-100)
  - 예상 집중력 점수 (0-100)
  - 예상 기분 (0-100)
  - 피처 중요도

**반환 예**:
```json
{
  "predictions": {
    "metric_energy_level": 78.5,
    "metric_focus_score": 82.1,
    "metric_mood": 75.3
  },
  "feature_importance": {
    "sleep_quality": 0.35,
    "total_protein_g": 0.28,
    "activity_count": 0.22
  }
}
```

#### 3.2 Association Rule Discovery
```python
GET /ml/patterns/association-rules/{person_id}
```
- **목적**: "높은 단백질 + 충분한 수면 → 높은 집중력" 같은 규칙 발견
- **알고리즘**: 이진 특징에 대한 빈도 기반 규칙 마이닝
- **평가 지표**:
  - **Support**: 규칙이 성립하는 일의 비율
  - **Confidence**: 조건이 참일 때 결론이 참일 확률

**반환 예**:
```json
{
  "rules": [
    {
      "if": "sleep_quality_high AND total_protein_high",
      "then": "focus_high",
      "support": 0.35,
      "confidence": 0.87
    }
  ]
}
```

---

### 4. **Link Prediction**
```python
GET /ml/link-prediction/suggest
POST /ml/link-prediction/train
```
- **목적**: 아직 연결되지 않은 노드 간의 숨은 관계 발견
- **예시**:
  - "브로콜리" → "고집중력" (아직 추가 안 됨)
  - "야간 고설탕 식사" → "수면 부족" 

- **알고리즘**: 로지스틱 회귀 + 노드 쌍 특징:
  - 타입 호환성 (식품-메트릭이 연결되기 쉬움)
  - 공통 이웃 수
  - 영양학적 적합성 (고단백 → 높은 집중력)
  - 그래프 거리
  - 시간적 근접도

**반환 예**:
```json
{
  "suggested_links": [
    {
      "source": "food:almonds",
      "source_type": "food",
      "target": "metric:energy_level",
      "target_type": "metric",
      "probability": 0.82,
      "reasoning": "High protein content likely improves energy"
    }
  ]
}
```

---

### 5. **Node Classification**

#### 5.1 음식 영양 분류
```python
GET /ml/classification/food/{food_id}
```
- **카테고리**: "healthy" / "balanced" / "indulgent"
- **기준**:
  - 단백질 함량
  - 섬유소 함량
  - 설탕 함량
  - 영양 밀도

**반환 예**:
```json
{
  "food_id": "food:salmon",
  "name": "Salmon",
  "classification": "healthy",
  "confidence": 0.94,
  "nutritional_profile": {
    "protein_g": 25,
    "fiber_g": 0,
    "sugar_g": 0,
    "fat_g": 17,
    "kcal": 280
  }
}
```

#### 5.2 활동 강도 분류
```python
GET /ml/classification/activity/{activity_id}
```
- **카테고리**: "light" / "moderate" / "vigorous"
- **기준**:
  - 지속 시간
  - 강도 레벨
  - 소모 칼로리

#### 5.3 일일 건강 상태 분류
```python
GET /ml/classification/daily-status/{person_id}/{date}
```
- **카테고리**: "optimal" / "good" / "fair" / "concerning"
- **평가 차원**:
  - 수면 (시간, 질)
  - 영양 (칼로리, 단백질, 섬유소)
  - 활동 (빈도, 강도)
  - 결과 메트릭 (에너지, 집중력, 기분)

**반환 예**:
```json
{
  "classification": "good",
  "overall_score": 12,
  "max_score": 15,
  "breakdown": {
    "sleep": 3,
    "nutrition": 2,
    "activity": 3,
    "outcomes": 3
  },
  "weak_areas": ["nutrition"],
  "recommendation": "You're doing well. Minor improvements possible."
}
```

---

## 통합 대시보드 엔드포인트

### 일일 종합 인사이트
```python
GET /ml/insights/daily/{person_id}/{date}
```
모든 ML 모델의 결과를 종합한 일일 대시보드:
- 일일 피처 (식사, 수면, 활동)
- 현재 건강 상태 분류
- 이상 탐지 여부 및 원인
- 내일 예상 지표
- 중요 영향 요인

### 개인 종합 요약
```python
GET /ml/summary/{person_id}
```
장기 패턴 분석:
- 상위 5개 행동 인사이트
- 상위 5개 건강 규칙
- 기록된 데이터 통계

---

## 빠른 시작

### 1. 환경 설정
```bash
# 의존성 설치
pip install -r backend/requirements.txt

# 모든 ML 모델 학습
python backend/train_models.py
```

출력 예:
```
[1/7] Training GraphSAGE embeddings...
✓ GraphSAGE model saved
[2/7] Training anomaly detector...
✓ Anomaly detector trained (5 anomalies detected)
...
ML PIPELINE TRAINING COMPLETE
```

### 2. API 실행
```bash
cd backend
uvicorn main:app --reload
```

### 3. 테스트
```bash
# 일일 인사이트 조회
curl "http://localhost:8000/ml/insights/daily/person:yooni/2024-12-15"

# 음식 분류
curl "http://localhost:8000/ml/classification/food/food:salmon"

# 이상 탐지
curl "http://localhost:8000/ml/anomalies/detect/person:yooni/2024-12-15"

# 다음날 메트릭 예측
curl "http://localhost:8000/ml/prediction/metrics/person:yooni/2024-12-15"
```

---

## 데이터 흐름

```
User Input (Meal, Sleep, Activity)
          ↓
    [Add to Graph]
          ↓
  GraphX DiGraph Structure
          ↓
    [ML Processing]
    ├─ GraphSAGE → Embedding
    ├─ Isolation Forest + LOF → Anomalies
    ├─ LinearRegression → Predictions
    ├─ Logistic Regression → Link Prediction
    └─ RandomForest → Classification
          ↓
   [Generate Insights]
   ├─ Daily Status
   ├─ Anomaly Alert
   ├─ Next Day Forecast
   └─ Recommendations
          ↓
   Dashboard / API Response
```

---

## 모델 저장 및 로드

모든 ML 모델은 `backend/models/` 디렉토리에 joblib 형식으로 저장됩니다:

```
backend/models/
├── graph_sage.pkl              # GraphSAGE 임베딩 모델
├── anomaly_detector.pkl        # 이상 탐지 모델
├── metric_predictor.pkl        # 메트릭 예측 모델
├── link_predictor.pkl          # 링크 예측 모델
├── food_classifier.pkl         # 음식 분류기
└── activity_classifier.pkl     # 활동 분류기
```

API는 모델이 없으면 자동으로 학습하고 저장합니다.

---

## 알고리즘 상세

### GraphSAGE
```python
# 간소화된 구현
def aggregate_neighbors(node, layer):
    neighbor_embeddings = [embed[n] for n in neighbors(node)]
    return mean(neighbor_embeddings)

embedding[node] = (embedding[node] + aggregate_neighbors(node)) / 2
```

### Anomaly Detection
- **Isolation Forest**: 의사결정 트리로 이상치를 고립시킴
- **LOF (Local Outlier Factor)**: 주변 점들의 밀도와 비교

### 메트릭 예측
```
y (next day energy) = β₀ + β₁·sleep_quality + β₂·protein + β₃·activity + ...
```

---

## 확장 가능성

### 다음 단계
1. **Graph Neural Networks (GNN)** 업그레이드
   - PyTorch Geometric 사용
   - Graph Attention Networks (GAT)
   
2. **시계열 예측**
   - LSTM / Transformer 기반 다중 단계 예측
   
3. **개인화 추천**
   - 협필터링 + 그래프 기반 추천
   
4. **실시간 학습**
   - 온라인 학습 알고리즘
   - 새 데이터 추가 시 점진적 모델 업데이트

---

## 파일 구조

```
backend/
├── main.py                  # FastAPI 애플리케이션 + 모든 엔드포인트
├── train_models.py          # ML 모델 학습 파이프라인
├── requirements.txt         # Python 의존성
├── app/
│   ├── __init__.py
│   ├── graph_builder.py     # 그래프 구조 정의
│   ├── synthetic_data.py    # 60일 합성 데이터 생성
│   ├── graph_embeddings.py  # GraphSAGE 임베딩
│   ├── anomaly_detection.py # 이상 탐지 + 상관관계 분석
│   ├── pattern_analysis.py  # 메트릭 예측 + 규칙 발견
│   ├── link_prediction.py   # 링크 예측
│   └── node_classifier.py   # 노드 분류 (음식, 활동, 상태)
└── models/                  # 저장된 ML 모델 (자동 생성)
    ├── graph_sage.pkl
    ├── anomaly_detector.pkl
    ├── metric_predictor.pkl
    ├── link_predictor.pkl
    ├── food_classifier.pkl
    └── activity_classifier.pkl
```

---

## 참고 문헌

- GraphSAGE: Hamilton et al. "Inductive Representation Learning on Large Graphs"
- Isolation Forest: Liu et al. "Isolation Forest"
- Local Outlier Factor: Breunig et al.

---

**작성자**: AI Assistant  
**마지막 업데이트**: 2025-01-18  
**버전**: 0.2.0 (ML enabled)
