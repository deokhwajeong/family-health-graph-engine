# Household Health Graph Engine - ML Implementation Guide

## Overview

This project analyzes family health patterns using a graph-based approach and derives insights through machine learning.

Designed with a **Data Add → Learn → Predict** architecture:
1. **Add**: Daily meals, sleep, and activity data are added as graph nodes
2. **Learn**: Patterns are extracted from accumulated data
3. **Predict**: Future health states are predicted and anomalies detected

---

## Implemented ML Features

### 1. **Graph Embedding (GraphSAGE)**
```python
GET /ml/embeddings/daily/{person_id}/{date}
```
- **Purpose**: Represent daily health behaviors as vector embeddings
- **Algorithm**: GraphSAGE variant that aggregates neighbor node information across 2 layers
- **Features**:
  - Daily embeddings combining meal, sleep, and activity information
  - Type-specific feature extraction per node type (weekday, sleep quality, calories, etc.)
  - Meaningful health pattern vector representation

**Usage Example**:
```bash
curl http://localhost:8000/ml/embeddings/daily/person:mia/2024-12-15
```

---

### 2. **Anomaly Detection (Isolation Forest + LOF)**
```python
GET /ml/anomalies/detect/{person_id}/{date}
POST /ml/anomalies/train
```
- **Purpose**: Detect days that deviate from a person's normal patterns
- **Algorithms**:
  - **Isolation Forest**: Global outlier detection (extreme behaviors)
  - **Local Outlier Factor**: Local density-based anomaly detection
- **Detection Criteria**:
  - Very low sleep duration (< 6 hours)
  - Extreme calorie intake (< 1000 or > 4000 kcal)
  - High sugar consumption (> 100g)
  - Imbalance between activity and nutrition

**Response Example**:
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

**Correlation Analysis**:
```python
GET /ml/anomalies/correlations/{person_id}
```
- Relationship between sleep quality and focus
- Relationship between protein intake and energy level
- Relationship between activity frequency and sleep duration

---

### 3. **Pattern Analysis & Prediction**

#### 3.1 Metric Prediction
```python
GET /ml/prediction/metrics/{person_id}/{date}
```
- **Purpose**: Predict next-day energy, focus, and mood from current-day behaviors
- **Algorithm**: Multivariate Linear Regression
- **Input Features**:
  - Total calories, protein, fiber, sugar
  - Sleep duration, sleep quality
  - Activity count, activity intensity, calories burned
- **Output**:
  - Predicted energy level (0-100)
  - Predicted focus score (0-100)
  - Predicted mood (0-100)
  - Feature importance

**Response Example**:
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
- **Purpose**: Discover rules like "high protein + sufficient sleep → high focus"
- **Algorithm**: Frequency-based rule mining on binary features
- **Evaluation Metrics**:
  - **Support**: Proportion of days where the rule holds
  - **Confidence**: Probability of the conclusion being true when the condition is true

**Response Example**:
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
- **Purpose**: Discover hidden relationships between unconnected nodes
- **Examples**:
  - "Broccoli" → "High focus" (not yet connected)
  - "Late-night high-sugar meal" → "Sleep deficit"

- **Algorithm**: Logistic Regression + node pair features:
  - Type compatibility (food-metric pairs are more likely to connect)
  - Common neighbor count
  - Nutritional compatibility (high protein → high focus)
  - Graph distance
  - Temporal proximity

**Response Example**:
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

#### 5.1 Food Nutrition Classification
```python
GET /ml/classification/food/{food_id}
```
- **Categories**: "healthy" / "balanced" / "indulgent"
- **Criteria**:
  - Protein content
  - Fiber content
  - Sugar content
  - Nutritional density

**Response Example**:
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

#### 5.2 Activity Intensity Classification
```python
GET /ml/classification/activity/{activity_id}
```
- **Categories**: "light" / "moderate" / "vigorous"
- **Criteria**:
  - Duration
  - Intensity level
  - Calories burned

#### 5.3 Daily Health Status Classification
```python
GET /ml/classification/daily-status/{person_id}/{date}
```
- **Categories**: "optimal" / "good" / "fair" / "concerning"
- **Evaluation Dimensions**:
  - Sleep (duration, quality)
  - Nutrition (calories, protein, fiber)
  - Activity (frequency, intensity)
  - Outcome metrics (energy, focus, mood)

**Response Example**:
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

## Integrated Dashboard Endpoints

### Daily Comprehensive Insight
```python
GET /ml/insights/daily/{person_id}/{date}
```
A daily dashboard combining results from all ML models:
- Daily features (meals, sleep, activity)
- Current health status classification
- Anomaly detection results and causes
- Next-day predicted metrics
- Key influencing factors

### Personal Summary
```python
GET /ml/summary/{person_id}
```
Long-term pattern analysis:
- Top 5 behavioral insights
- Top 5 health rules
- Recorded data statistics

---

## Quick Start

### 1. Environment Setup
```bash
# Install dependencies
pip install -r backend/requirements.txt

# Train all ML models
python backend/train_models.py
```

Expected output:
```
[1/7] Training GraphSAGE embeddings...
 GraphSAGE model saved
[2/7] Training anomaly detector...
 Anomaly detector trained (5 anomalies detected)
...
ML PIPELINE TRAINING COMPLETE
```

### 2. Run API Server
```bash
cd backend
uvicorn main:app --reload
```

### 3. Test
```bash
# Daily insight query
curl "http://localhost:8000/ml/insights/daily/person:mia/2024-12-15"

# Food classification
curl "http://localhost:8000/ml/classification/food/food:salmon"

# Anomaly detection
curl "http://localhost:8000/ml/anomalies/detect/person:mia/2024-12-15"

# Next-day metric prediction
curl "http://localhost:8000/ml/prediction/metrics/person:mia/2024-12-15"
```

---

## Data Flow

```
User Input (Meal, Sleep, Activity)
          |
    [Add to Graph]
          |
  NetworkX DiGraph Structure
          |
    [ML Processing]
    |- GraphSAGE -> Embedding
    |- Isolation Forest + LOF -> Anomalies
    |- LinearRegression -> Predictions
    |- Logistic Regression -> Link Prediction
    +- RandomForest -> Classification
          |
   [Generate Insights]
   |- Daily Status
   |- Anomaly Alert
   |- Next Day Forecast
   +- Recommendations
          |
   Dashboard / API Response
```

---

## Model Storage and Loading

All ML models are saved in joblib format under the `backend/models/` directory:

```
backend/models/
|- graph_sage.pkl              # GraphSAGE embedding model
|- anomaly_detector.pkl        # Anomaly detection model
|- metric_predictor.pkl        # Metric prediction model
|- link_predictor.pkl          # Link prediction model
|- food_classifier.pkl         # Food classifier
+- activity_classifier.pkl     # Activity classifier
```

The API automatically trains and saves models if they don't already exist.

---

## Algorithm Details

### GraphSAGE
```python
# Simplified implementation
def aggregate_neighbors(node, layer):
    neighbor_embeddings = [embed[n] for n in neighbors(node)]
    return mean(neighbor_embeddings)

embedding[node] = (embedding[node] + aggregate_neighbors(node)) / 2
```

### Anomaly Detection
- **Isolation Forest**: Isolates anomalies using decision tree partitioning
- **LOF (Local Outlier Factor)**: Compares local density against neighboring points

### Metric Prediction
```
y (next day energy) = b0 + b1*sleep_quality + b2*protein + b3*activity + ...
```

---

## Future Roadmap

### Next Steps
1. **Graph Neural Networks (GNN)** Upgrade
   - PyTorch Geometric integration
   - Graph Attention Networks (GAT)

2. **Time Series Forecasting**
   - LSTM / Transformer-based multi-step prediction

3. **Personalized Recommendations**
   - Collaborative filtering + graph-based recommendations

4. **Online Learning**
   - Incremental learning algorithms
   - Gradual model updates as new data is added

---

## File Structure

```
backend/
|- main.py                  # FastAPI application + all endpoints
|- train_models.py          # ML model training pipeline
|- requirements.txt         # Python dependencies
|- app/
|   |- __init__.py
|   |- graph_builder.py     # Graph schema definition
|   |- synthetic_data.py    # 90-day synthetic data generation
|   |- graph_embeddings.py  # GraphSAGE embeddings
|   |- anomaly_detection.py # Anomaly detection + correlation analysis
|   |- pattern_analysis.py  # Metric prediction + rule discovery
|   |- link_prediction.py   # Link prediction
|   +- node_classifier.py   # Node classification (food, activity, status)
+- models/                  # Saved ML models (auto-generated)
    |- graph_sage.pkl
    |- anomaly_detector.pkl
    |- metric_predictor.pkl
    |- link_predictor.pkl
    |- food_classifier.pkl
    +- activity_classifier.pkl
```

---

## References

- GraphSAGE: Hamilton et al. "Inductive Representation Learning on Large Graphs"
- Isolation Forest: Liu et al. "Isolation Forest"
- Local Outlier Factor: Breunig et al.

---

**Author**: AI Assistant
**Last Updated**: 2026-02-07
**Version**: 0.2.0 (ML enabled)
