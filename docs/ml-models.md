# ML models

All models are small, CPU-only, and fit in well under a second on the synthetic
Demo family graph. The repo ships **without** trained `.pkl` files: every
inference endpoint fits the model it needs on demand and keeps it in memory.
You can still persist models with the `/ml/*/train` endpoints or
`python backend/train_models.py`; saved files go to `backend/models/` (git-ignored)
and are used when present.

Every model consumes the same per-day feature vector produced by
`extract_day_features(G, person_id, date)` in `backend/app/graph_embeddings.py`,
which walks the graph from a `day` node to its meals, foods, sleep, activities,
and metrics:

| Feature | Source in the graph |
|---|---|
| `total_calories`, `total_protein_g`, `total_fiber_g`, `total_sugar_g`, `meal_count` | `day -> meal -> food` |
| `sleep_duration_h`, `sleep_quality` | `day -> sleep` |
| `activity_count`, `total_calories_burned`, `max_intensity` | `day -> activity` |
| `metric_energy_level`, `metric_focus_score`, `metric_mood`, `metric_steps`, `metric_resting_heart_rate` | `day -> metric` |

## 1. Graph embeddings (GraphSAGE-style)

`GET /ml/embeddings/daily/{person_id}/{date}`

A simplified, NumPy-only GraphSAGE: each node starts with a type one-hot plus
type-specific attributes (sleep quality, activity intensity, food nutrients),
then two rounds of mean-aggregation over neighbours. The `day` node's vector
summarises that person's day. Because the embedding is transductive, it is
always fitted on the graph being queried.

## 2. Anomaly detection (Isolation Forest + LOF)

`GET /ml/anomalies/detect/{person_id}/{date}`, `POST /ml/anomalies/train`

An ensemble of scikit-learn `IsolationForest` (global outliers) and
`LocalOutlierFactor` (local density) over the scaled day features of one
person. Results include a 0 to 1 score and human-readable reasons, for example
very short sleep, very high sugar, or high activity with low intake.

## 3. Lagged correlations

`GET /ml/anomalies/correlations/{person_id}`

Pearson correlation between a behaviour on day *d* and an outcome on day *d+1*
(for example `total_sugar_g -> sleep_quality(+1d)`). Correlations with
|r| > 0.5 are turned into plain-language insights. The synthetic generator
deliberately carries yesterday's sugar, activity, and sleep into today so these
cross-day effects are present and discoverable.

## 4. Next-day metric prediction

`GET /ml/prediction/metrics/{person_id}/{date}`, `POST /ml/prediction/train`

Multivariate linear regression from today's features to tomorrow's energy,
focus, and mood, with per-feature importance.

## 5. Association rules

`GET /ml/patterns/association-rules/{person_id}`

Features are discretised into low / medium / high and frequent
"if A and B then C" rules are mined with support and confidence thresholds.

## 6. Link prediction

`GET /ml/link-prediction/suggest`, `POST /ml/link-prediction/train`

Logistic regression over node-pair features (type compatibility, common
neighbours, graph distance, nutritional fit) to suggest edges that are not in
the graph yet, such as a food that tends to co-occur with high-focus days.

## 7. Node classification

| Endpoint | Model | Classes |
|---|---|---|
| `GET /ml/classification/food/{food_id}` | Random forest on nutrients | healthy / balanced / indulgent |
| `GET /ml/classification/activity/{activity_id}` | Random forest on duration, intensity, kcal | light / moderate / vigorous |
| `GET /ml/classification/daily-status/{person_id}/{date}` | Transparent scoring over sleep, nutrition, activity, outcomes | optimal / good / fair / concerning |

## 8. Combined daily insight

`GET /ml/insights/daily/{person_id}/{date}` combines the day's features,
status classification, anomaly result, and next-day predictions in one call.
This is what the dashboard and person views use.

## Limits

- Models are fitted per person on a few months of synthetic days; they are
  demonstrations of the pipeline, not validated health models.
- Recommendations and classifications are non-diagnostic and are not medical
  advice.
